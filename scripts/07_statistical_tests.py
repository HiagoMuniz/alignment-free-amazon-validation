import pandas as pd, numpy as np
from pathlib import Path
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import mcnemar

ROOT = Path(__file__).resolve().parents[1]
RW = ROOT / "external_tools"
def acc(s): return str(s).split('|')[0].split()[0].split('.')[0]

# accessions dos 82 positivos
pos_acc = [acc(l[1:].split()[0]) for l in open(ROOT/"data/amazon_viruses.fasta") if l.startswith(">")]
pos_acc = list(dict.fromkeys(pos_acc))
N=len(pos_acc)

# --- predicoes por metodo nos positivos (1=viral) ---
dvf=pd.read_csv(RW/"amazon82/dvf_amazon82.txt",sep="\t")
dvf['a']=dvf['name'].map(acc); dvf_set=set(dvf[(dvf.score>=0.5)&(dvf.pvalue<0.05)]['a'])
vs2=pd.read_csv(RW/"amazon82/vs2_amazon82.tsv",sep="\t"); vs2_set=set(vs2['seqname'].map(acc))
gen=pd.read_csv(RW/"amazon82/genomad_amazon82.tsv",sep="\t"); gen_set=set(gen['seq_name'].map(acc))

methods={"DeepVirFinder":dvf_set,"VirSorter2":vs2_set,"geNomad":gen_set}
pred={m:np.array([1 if a in s else 0 for a in pos_acc]) for m,s in methods.items()}

print("=== RECALL com IC de Wilson 95% (n=82 positivos) ===")
for m,s in methods.items():
    tp=sum(1 for a in pos_acc if a in s)
    lo,hi=proportion_confint(tp,N,method="wilson")
    print(f"  {m:16s}: {tp}/{N} = {tp/N:.1%}  IC95% [{lo:.1%}, {hi:.1%}]")
# classico: media+-std de 20 embeddings
ev=pd.read_csv(ROOT/"results/embedding_variance_v2.csv")
m_=ev.recall_ood.mean(); s_=ev.recall_ood.std()
print(f"  {'XGBoost (20 emb)':16s}: {m_:.1%} +/- {s_:.1%}  (faixa {ev.recall_ood.min():.1%}-{ev.recall_ood.max():.1%})")

print("\n=== McNemar pareado entre ferramentas externas (positivos) ===")
names=list(methods)
for i in range(len(names)):
    for j in range(i+1,len(names)):
        a,b=names[i],names[j]
        n01=int(np.sum((pred[a]==0)&(pred[b]==1))); n10=int(np.sum((pred[a]==1)&(pred[b]==0)))
        tb=[[0,n01],[n10,0]]
        p=mcnemar(tb,exact=True).pvalue
        sig="SIM" if p<0.05 else "nao"
        print(f"  {a} vs {b}: discordam {a}+:{n10} {b}+:{n01}  p={p:.3f}  signif={sig}")

# --- especificidade nos 7542 negativos v2 ---
print("\n=== ESPECIFICIDADE com IC Wilson 95% (n=7542 negativos transcritos) ===")
NEG=7542
dvfn=pd.read_csv(RW/"negatives/dvf_negv2.txt",sep="\t")
dvf_fp=int(((dvfn.score>=0.5)&(dvfn.pvalue<0.05)).sum())
vs2n=pd.read_csv(RW/"negatives/vs2_negv2.tsv",sep="\t"); vs2_fp=vs2n['seqname'].map(acc).nunique()
genn=RW/"negatives/genomad_negv2.tsv"
genn=pd.read_csv(genn,sep="\t"); gen_fp=genn['seq_name'].map(acc).nunique() if len(genn) else 0
for m,fp in [("DeepVirFinder",dvf_fp),("VirSorter2",vs2_fp),("geNomad",gen_fp)]:
    tn=NEG-fp; lo,hi=proportion_confint(tn,NEG,method="wilson")
    print(f"  {m:16s}: {tn}/{NEG} = {tn/NEG:.1%}  IC95% [{lo:.1%}, {hi:.1%}]  (FP={fp})")
m_=ev.specificity_ood.mean(); s_=ev.specificity_ood.std()
print(f"  {'XGBoost (20 emb)':16s}: {m_:.1%} +/- {s_:.1%}")
