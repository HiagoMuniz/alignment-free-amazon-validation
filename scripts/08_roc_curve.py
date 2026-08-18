#!/usr/bin/env python3
"""
Gera curva ROC e Precision-Recall do classificador proposto (XGBoost +
Word2Vec k=6) nos dados amazonicos (82 virus + 7542 transcritos negativos).

ILUSTRATIVO: usa UM embedding (o desempenho OOD varia entre embeddings,
ver embedding_variance_v2.csv). A curva mostra o trade-off de thresholds;
nao serve para escolher threshold (isso seria vies se feito no teste).
Ferramentas externas (VS2, geNomad) dao decisao binaria -> ponto unico,
nao curva; por isso a ROC e so do modelo proposto.
"""
import os
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from Bio import SeqIO
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from xgboost import XGBClassifier

E=Path(__file__).resolve().parents[1]; VSC=Path(os.environ.get("VSC_ROOT", E.parent/"Viral-Sequence-Classification"))
K=6
def km(s): s=str(s).upper(); return [s[i:i+K] for i in range(len(s)-K+1)]
def load(p,l): return [(str(r.seq).upper(),l) for r in SeqIO.parse(str(p),"fasta")]
def vec(s,m):
    v=[m.wv[k] for k in km(s) if k in m.wv]
    return np.mean(v,axis=0) if v else np.zeros(m.vector_size,dtype=np.float32)

data=load(VSC/"src/data/training/viral.fasta",1)+load(VSC/"src/data/training/nonviral.fasta",0)
seqs=[d[0] for d in data]; lab=[d[1] for d in data]
idx=list(range(len(seqs)))
ip,_,_,_=train_test_split(idx,lab,test_size=0.2,random_state=42,stratify=lab)
pool=[seqs[i] for i in ip]; yp=np.array([lab[i] for i in ip])
print("treinando embedding+xgb (1 rodada ilustrativa)...",flush=True)
emb=Word2Vec(sentences=[km(s) for s in pool],vector_size=100,window=5,min_count=5,sg=1)
clf=XGBClassifier(n_estimators=300,max_depth=5,learning_rate=0.1,subsample=0.8,
    scale_pos_weight=8.8,eval_metric="logloss",random_state=42,n_jobs=-1)
clf.fit(np.vstack([vec(s,emb) for s in pool]),yp)

pos=[str(r.seq).upper() for r in SeqIO.parse(str(E/"data/amazon_viruses.fasta"),"fasta")]
neg=[str(r.seq).upper() for r in SeqIO.parse(str(E/"data/amazon_negatives_v2.fasta"),"fasta")]
Xp=np.vstack([vec(s,emb) for s in pos]); Xn=np.vstack([vec(s,emb) for s in neg])
y_true=np.r_[np.ones(len(pos)),np.zeros(len(neg))]
y_score=np.r_[clf.predict_proba(Xp)[:,1],clf.predict_proba(Xn)[:,1]]

fpr,tpr,_=roc_curve(y_true,y_score); roc_auc=auc(fpr,tpr)
prec,rec,_=precision_recall_curve(y_true,y_score); ap=average_precision_score(y_true,y_score)
print(f"ROC-AUC={roc_auc:.3f}  AP(PR-AUC)={ap:.3f}  (razao {len(pos)}:{len(neg)})")

fig,(a1,a2)=plt.subplots(1,2,figsize=(10,4))
a1.plot(fpr,tpr,color="#1f77b4",lw=2,label=f"XGBoost (AUC={roc_auc:.3f})")
a1.plot([0,1],[0,1],"--",color="gray",lw=1)
# marca o ponto do threshold 0.5
from sklearn.metrics import confusion_matrix
pred05=(y_score>=0.5).astype(int)
tn,fp,fn,tp=confusion_matrix(y_true,pred05).ravel()
a1.plot(fp/(fp+tn),tp/(tp+fn),"o",color="red",ms=9,label="threshold=0.5 (usado)")
a1.set_xlabel("Taxa de falso positivo (1 - especificidade)"); a1.set_ylabel("Recall (sensibilidade)")
a1.set_title("Curva ROC"); a1.legend(loc="lower right",frameon=False); a1.grid(alpha=.3)
a2.plot(rec,prec,color="#1f77b4",lw=2,label=f"XGBoost (AP={ap:.3f})")
a2.plot(tp/(tp+fn),tp/(tp+fp),"o",color="red",ms=9,label="threshold=0.5")
a2.set_xlabel("Recall"); a2.set_ylabel("Precisao")
a2.set_title(f"Curva Precisao-Recall (razao {len(pos)}:{len(neg)})")
a2.legend(loc="upper right",frameon=False); a2.grid(alpha=.3)
fig.tight_layout()
out=E/"results/roc_pr_curve_ilustrativa.png"
fig.savefig(out,dpi=130,bbox_inches="tight")
print(f"salvo: {out}")
