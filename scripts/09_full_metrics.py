#!/usr/bin/env python3
"""
Calcula as metricas que faltam para as tabelas do paper, sobre N
embeddings (media +- desvio), para os dois classicos:
  XGBoost (Word2Vec) e ExtraTrees (fastText), k=6.
Para cada embedding: treina no pool 80% ZOVER, prediz probabilidades
nos 82 virus + 7542 transcritos negativos v2, e calcula no ratio natural:
ROC-AUC, PR-AUC, e recall por familia. Precisao/F1/MCC sao derivadas
analiticamente de recall+especificidade depois (nao precisam de probs).
"""
import time, json
import os
from pathlib import Path
import numpy as np, pandas as pd
from Bio import SeqIO
from gensim.models import Word2Vec, FastText
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier

E=Path(__file__).resolve().parents[1]; VSC=Path(os.environ.get("VSC_ROOT", E.parent/"Viral-Sequence-Classification"))
K=6; N_EMB=10
def km(s): s=str(s).upper(); return [s[i:i+K] for i in range(len(s)-K+1)]
def load(p,l): return [(str(r.seq).upper(),l) for r in SeqIO.parse(str(p),"fasta")]
def vec(s,m):
    v=[m.wv[k] for k in km(s) if k in m.wv]
    return np.mean(v,axis=0) if v else np.zeros(m.vector_size,dtype=np.float32)

# familias dos 82 (do metadata)
meta=pd.read_csv(E/"data/accessions.csv")
fam_by_acc=dict(zip(meta.accession,meta.family))
def acc(h): return h.split()[0].split(".")[0]

data=load(VSC/"src/data/training/viral.fasta",1)+load(VSC/"src/data/training/nonviral.fasta",0)
seqs=[d[0] for d in data]; lab=[d[1] for d in data]
ip,_,_,_=train_test_split(list(range(len(seqs))),lab,test_size=0.2,random_state=42,stratify=lab)
pool=[seqs[i] for i in ip]; yp=np.array([lab[i] for i in ip])
pool_km=[km(s) for s in pool]

pos_recs=list(SeqIO.parse(str(E/"data/amazon_viruses.fasta"),"fasta"))
pos=[str(r.seq).upper() for r in pos_recs]
pos_fam=[fam_by_acc.get(acc(r.id),"?") for r in pos_recs]
neg=[str(r.seq).upper() for r in SeqIO.parse(str(E/"data/amazon_negatives_v2.fasta"),"fasta")]
y_true=np.r_[np.ones(len(pos)),np.zeros(len(neg))]

configs={"XGBoost":("w2v",lambda:XGBClassifier(n_estimators=300,max_depth=5,learning_rate=0.1,
            subsample=0.8,scale_pos_weight=8.8,eval_metric="logloss",random_state=42,n_jobs=-1)),
         "ExtraTrees":("ft",lambda:ExtraTreesClassifier(n_estimators=200,max_depth=10,
            min_samples_split=2,class_weight="balanced",random_state=42,n_jobs=-1))}

res={m:{"roc_auc":[],"pr_auc":[],"fam":{}} for m in configs}
for run in range(1,N_EMB+1):
    w2v=Word2Vec(sentences=pool_km,vector_size=100,window=5,min_count=5,sg=1)
    ft=FastText(sentences=pool_km,vector_size=100,window=5,min_count=5,sg=1)
    embs={"w2v":w2v,"ft":ft}
    for m,(ek,mk) in configs.items():
        emb=embs[ek]
        clf=mk(); clf.fit(np.vstack([vec(s,emb) for s in pool]),yp)
        Xpos=np.vstack([vec(s,emb) for s in pos]); Xneg=np.vstack([vec(s,emb) for s in neg])
        pp=clf.predict_proba(Xpos)[:,1]; pn=clf.predict_proba(Xneg)[:,1]
        score=np.r_[pp,pn]
        res[m]["roc_auc"].append(roc_auc_score(y_true,score))
        res[m]["pr_auc"].append(average_precision_score(y_true,score))
        pred_pos=(pp>=0.5).astype(int)
        for f,pr in zip(pos_fam,pred_pos):
            res[m]["fam"].setdefault(f,[]).append(pr)
    print(f"  emb {run}/{N_EMB} ok",flush=True)

out={}
for m in configs:
    out[m]={"roc_auc_mean":float(np.mean(res[m]["roc_auc"])),"roc_auc_std":float(np.std(res[m]["roc_auc"])),
            "pr_auc_mean":float(np.mean(res[m]["pr_auc"])),"pr_auc_std":float(np.std(res[m]["pr_auc"]))}
    print(f"\n{m}: ROC-AUC={out[m]['roc_auc_mean']:.3f}+-{out[m]['roc_auc_std']:.3f}  "
          f"PR-AUC={out[m]['pr_auc_mean']:.3f}+-{out[m]['pr_auc_std']:.3f}")
# recall por familia (media), so precisa de uma contagem por familia
fam_order=["Phasmaviridae","Rhabdoviridae","Iflaviridae","Togaviridae","Flaviviridae",
           "Mesoniviridae","Totiviridae","Phenuiviridae","Xinmoviridae","Peribunyaviridae"]
print("\nRecall por familia (media sobre embeddings):")
print(f"{'Familia':18s} {'n':>3s} {'XGBoost':>8s} {'ExtraTrees':>11s}")
fam_table={}
for f in fam_order:
    n=pos_fam.count(f)
    xg=np.mean(res["XGBoost"]["fam"].get(f,[0]))
    et=np.mean(res["ExtraTrees"]["fam"].get(f,[0]))
    fam_table[f]={"n":n,"xgboost":round(xg,3),"extratrees":round(et,3)}
    print(f"{f:18s} {n:>3d} {xg:>7.1%} {et:>10.1%}")
out["familias"]=fam_table
json.dump(out,open(E/"results/full_metrics.json","w"),indent=2)
print("\nsalvo: results/full_metrics.json")
