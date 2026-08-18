#!/usr/bin/env python3
"""Tudo num lugar so, n=20, mesmas rodadas para todos os numeros:
recall, especificidade, ROC-AUC, PR-AUC e recall por familia, para
XGBoost(Word2Vec) e ExtraTrees(fastText). Garante consistencia interna."""
import json, numpy as np
import os
from pathlib import Path
from Bio import SeqIO
from gensim.models import Word2Vec, FastText
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
import csv
E=Path(__file__).resolve().parents[1]
VSC=Path(os.environ.get("VSC_ROOT", E.parent/"Viral-Sequence-Classification"))
K=6; N=20
def km(s): s=str(s).upper(); return [s[i:i+K] for i in range(len(s)-K+1)]
def load(p,l): return [(str(r.seq).upper(),l) for r in SeqIO.parse(str(p),"fasta")]
def vec(s,m):
    v=[m.wv[k] for k in km(s) if k in m.wv]
    return np.mean(v,axis=0) if v else np.zeros(m.vector_size,dtype=np.float32)
fam={r['accession']:r['family'] for r in csv.DictReader(open('data/accessions.csv'))}
data=load(VSC/"src/data/training/viral.fasta",1)+load(VSC/"src/data/training/nonviral.fasta",0)
seqs=[d[0] for d in data]; lab=[d[1] for d in data]
ip,_,_,_=train_test_split(list(range(len(seqs))),lab,test_size=0.2,random_state=42,stratify=lab)
pool=[seqs[i] for i in ip]; yp=np.array([lab[i] for i in ip]); pk=[km(s) for s in pool]
precs=list(SeqIO.parse(str(E/"data/amazon_viruses.fasta"),"fasta"))
pos=[str(r.seq).upper() for r in precs]; posfam=[fam.get(r.id.split('.')[0],'?') for r in precs]
neg=[str(r.seq).upper() for r in SeqIO.parse(str(E/"data/amazon_negatives_v2.fasta"),"fasta")]
yt=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
cfgs={"XGBoost":("w2v",lambda:XGBClassifier(n_estimators=300,max_depth=5,learning_rate=0.1,subsample=0.8,scale_pos_weight=8.8,eval_metric="logloss",random_state=42,n_jobs=-1)),
      "ExtraTrees":("ft",lambda:ExtraTreesClassifier(n_estimators=200,max_depth=10,min_samples_split=2,class_weight="balanced",random_state=42,n_jobs=-1))}
R={m:{"rec":[],"spec":[],"roc":[],"pr":[],"fam":{}} for m in cfgs}
for run in range(1,N+1):
    embs={"w2v":Word2Vec(sentences=pk,vector_size=100,window=5,min_count=5,sg=1),
          "ft":FastText(sentences=pk,vector_size=100,window=5,min_count=5,sg=1)}
    for m,(ek,mk) in cfgs.items():
        em=embs[ek]; clf=mk(); clf.fit(np.vstack([vec(s,em) for s in pool]),yp)
        pp=clf.predict_proba(np.vstack([vec(s,em) for s in pos]))[:,1]
        pn=clf.predict_proba(np.vstack([vec(s,em) for s in neg]))[:,1]
        R[m]["rec"].append((pp>=.5).mean()); R[m]["spec"].append((pn<.5).mean())
        R[m]["roc"].append(roc_auc_score(yt,np.r_[pp,pn])); R[m]["pr"].append(average_precision_score(yt,np.r_[pp,pn]))
        for f,pr in zip(posfam,(pp>=.5).astype(int)): R[m]["fam"].setdefault(f,[]).append(pr)
    print(f"run {run}/{N}",flush=True)
out={}
for m in cfgs:
    out[m]={k:[round(float(np.mean(R[m][k])),4),round(float(np.std(R[m][k])),4)] for k in ["rec","spec","roc","pr"]}
    out[m]["fam"]={f:round(float(np.mean(v)),3) for f,v in R[m]["fam"].items()}
json.dump(out,open("results/all_metrics_n20.json","w"),indent=2)
for m in cfgs:
    o=out[m]; print(f"{m}: rec={o['rec'][0]:.3f}+-{o['rec'][1]:.3f} spec={o['spec'][0]:.3f}+-{o['spec'][1]:.3f} roc={o['roc'][0]:.3f} pr={o['pr'][0]:.3f}")
print("OK")
