#!/usr/bin/env python3
"""
Variancia do embedding para o ExtraTrees (fastText, k=6) — espelho do
05b mas com fastText (embedding do ExtraTrees no BRACIS) e classificador
ExtraTrees. 20 rodadas: F1 in-dist, recall OOD (82 virus), especificidade
OOD (7542 transcritos v2).
"""
import argparse, time
import os
from pathlib import Path
import numpy as np, pandas as pd
from Bio import SeqIO
from gensim.models import FastText
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import ExtraTreesClassifier

E=Path(__file__).resolve().parents[1]; VSC=Path(os.environ.get("VSC_ROOT", E.parent/"Viral-Sequence-Classification"))
VIRAL=VSC/"src/data/training/viral.fasta"; NONVIRAL=VSC/"src/data/training/nonviral.fasta"
AMAZON_POS=E/"data/amazon_viruses.fasta"; AMAZON_NEG=E/"data/amazon_negatives_v2.fasta"
K=6; SEED_SPLIT=42

def kmers(s): s=str(s).upper(); return [s[i:i+K] for i in range(len(s)-K+1)]
def load(p,l): return [(str(r.seq).upper(),l) for r in SeqIO.parse(str(p),"fasta")]
def mean_vec(s,m):
    v=[m.wv[k] for k in kmers(s) if k in m.wv]
    return np.mean(v,axis=0) if v else np.zeros(m.vector_size,dtype=np.float32)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=20); a=ap.parse_args()
    data=load(VIRAL,1)+load(NONVIRAL,0)
    seqs=[d[0] for d in data]; labels=[d[1] for d in data]
    idx=list(range(len(seqs)))
    ip,it,_,_=train_test_split(idx,labels,test_size=0.2,random_state=SEED_SPLIT,stratify=labels)
    pool=[seqs[i] for i in ip]; test=[seqs[i] for i in it]
    yp=np.array([labels[i] for i in ip]); yt=np.array([labels[i] for i in it])
    pos=[str(r.seq).upper() for r in SeqIO.parse(str(AMAZON_POS),"fasta")]
    neg=[str(r.seq).upper() for r in SeqIO.parse(str(AMAZON_NEG),"fasta")]
    print(f"pool={len(pool)} test={len(test)} | OOD {len(pos)} virus {len(neg)} host",flush=True)
    pool_km=[kmers(s) for s in pool]
    rows=[]
    for run in range(1,a.n+1):
        t0=time.time()
        emb=FastText(sentences=pool_km,vector_size=100,window=5,min_count=5,sg=1)
        Xp=np.vstack([mean_vec(s,emb) for s in pool])
        clf=ExtraTreesClassifier(n_estimators=200,max_depth=10,min_samples_split=2,
            class_weight="balanced",random_state=42,n_jobs=-1)
        clf.fit(Xp,yp)
        f1id=f1_score(yt,clf.predict(np.vstack([mean_vec(s,emb) for s in test])),zero_division=0)
        rec=clf.predict(np.vstack([mean_vec(s,emb) for s in pos])).mean()
        spec=1-clf.predict(np.vstack([mean_vec(s,emb) for s in neg])).mean()
        rows.append(dict(run=run,f1_indist=f1id,recall_ood=rec,specificity_ood=spec,secs=time.time()-t0))
        print(f"  run {run:2d}/{a.n}: F1id={f1id:.4f} recall_OOD={rec:.4f} spec_OOD={spec:.4f} ({time.time()-t0:.0f}s)",flush=True)
        pd.DataFrame(rows).to_csv(E/"results/embedding_variance_extratrees.csv",index=False)
    df=pd.DataFrame(rows)
    print("\n=== RESUMO ExtraTrees (fastText) ===")
    for c,n in [("f1_indist","F1 in-dist"),("recall_ood","Recall OOD"),("specificity_ood","Espec OOD")]:
        print(f"  {n:14s}: media={df[c].mean():.4f} std={df[c].std():.4f} min={df[c].min():.4f} max={df[c].max():.4f}")
if __name__=="__main__": main()
