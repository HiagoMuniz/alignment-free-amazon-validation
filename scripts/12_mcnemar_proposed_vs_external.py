#!/usr/bin/env python3
"""
Paired McNemar test between each proposed model, over 20 embedding
repetitions, and each reference tool, on the 82 Amazonian viral sequences.
Reports the median and the range of the p-values and how many repetitions fall
below 0.05.

Output: results/mcnemar_proposed_vs_external.json
"""
import json, os
import numpy as np
from pathlib import Path
from Bio import SeqIO
from gensim.models import Word2Vec, FastText
from sklearn.model_selection import train_test_split
from statsmodels.stats.contingency_tables import mcnemar
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
import pandas as pd

E=Path(__file__).resolve().parents[1]
VSC=Path(os.environ.get("VSC_ROOT", E.parent/"Viral-Sequence-Classification"))
RW=E/"external_tools"
K=6; N=20
def km(s): s=str(s).upper(); return [s[i:i+K] for i in range(len(s)-K+1)]
def load(p,l): return [(str(r.seq).upper(),l) for r in SeqIO.parse(str(p),"fasta")]
def vec(s,m):
    v=[m.wv[k] for k in km(s) if k in m.wv]
    return np.mean(v,axis=0) if v else np.zeros(m.vector_size,dtype=np.float32)
def acc(s): return str(s).split('|')[0].split()[0].split('.')[0]

# the 82 positives, in fixed order
precs=list(SeqIO.parse(str(E/"data/amazon_viruses.fasta"),"fasta"))
pos=[str(r.seq).upper() for r in precs]; pos_acc=[acc(r.id) for r in precs]

# reference tools: one binary prediction per sequence, 1 = viral
dvf=pd.read_csv(RW/"amazon82/dvf_amazon82.txt",sep="\t"); dvf['a']=dvf['name'].map(acc)
dvf_set=set(dvf[(dvf.score>=0.5)&(dvf.pvalue<0.05)]['a'])
vs2_set=set(pd.read_csv(RW/"amazon82/vs2_amazon82.tsv",sep="\t")['seqname'].map(acc))
gen_set=set(pd.read_csv(RW/"amazon82/genomad_amazon82.tsv",sep="\t")['seq_name'].map(acc))
ext={"geNomad":np.array([1 if a in gen_set else 0 for a in pos_acc]),
     "DeepVirFinder":np.array([1 if a in dvf_set else 0 for a in pos_acc]),
     "VirSorter2":np.array([1 if a in vs2_set else 0 for a in pos_acc])}

# training pool
data=load(VSC/"src/data/training/viral.fasta",1)+load(VSC/"src/data/training/nonviral.fasta",0)
seqs=[d[0] for d in data]; lab=[d[1] for d in data]
ip,_,_,_=train_test_split(list(range(len(seqs))),lab,test_size=0.2,random_state=42,stratify=lab)
pool=[seqs[i] for i in ip]; yp=np.array([lab[i] for i in ip]); pk=[km(s) for s in pool]
cfgs={"XGBoost":("w2v",lambda:XGBClassifier(n_estimators=300,max_depth=5,learning_rate=0.1,subsample=0.8,scale_pos_weight=8.8,eval_metric="logloss",random_state=42,n_jobs=-1)),
      "ExtraTrees":("ft",lambda:ExtraTreesClassifier(n_estimators=200,max_depth=10,min_samples_split=2,class_weight="balanced",random_state=42,n_jobs=-1))}

pvals={f"{m} vs {t}":[] for m in cfgs for t in ext}
for run in range(1,N+1):
    embs={"w2v":Word2Vec(sentences=pk,vector_size=100,window=5,min_count=5,sg=1),
          "ft":FastText(sentences=pk,vector_size=100,window=5,min_count=5,sg=1)}
    for m,(ek,mk) in cfgs.items():
        em=embs[ek]; clf=mk(); clf.fit(np.vstack([vec(s,em) for s in pool]),yp)
        pred=clf.predict(np.vstack([vec(s,em) for s in pos]))
        for t,ev in ext.items():
            b=int(np.sum((pred==1)&(ev==0))); c=int(np.sum((pred==0)&(ev==1)))
            p=mcnemar([[0,b],[c,0]],exact=True).pvalue
            pvals[f"{m} vs {t}"].append(p)
    print(f"run {run}/{N}",flush=True)

out={}
print("\n=== McNemar, proposed vs reference tool (82 positives, 20 repetitions) ===")
for k,v in pvals.items():
    v=np.array(v); out[k]={"median_p":round(float(np.median(v)),3),"min_p":round(float(v.min()),3),"max_p":round(float(v.max()),3),"n_signif":int((v<0.05).sum())}
    print(f"  {k:26s}: median p={np.median(v):.3f}  range [{v.min():.3f},{v.max():.3f}]  significant(p<0.05): {int((v<0.05).sum())}/{N}")
json.dump(out,open(E/"results/mcnemar_proposed_vs_external.json","w"),indent=2)
print("OK -> results/mcnemar_proposed_vs_external.json")
