#!/usr/bin/env python3
"""
Wilson 95% confidence intervals and paired McNemar tests for the three
reference tools, over the 82 Amazonian viruses and the 7,542 host transcripts.

Reads only the committed raw tool outputs plus results/all_metrics_n20.json, so
it runs in seconds and needs no retraining. The figures for the proposed
classifiers come from that same JSON, which is the source of Table 1, so the two
cannot diverge.
"""
import json
import numpy as np, pandas as pd
from pathlib import Path
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.contingency_tables import mcnemar

ROOT = Path(__file__).resolve().parents[1]
RW = ROOT / "external_tools"
def acc(s): return str(s).split('|')[0].split()[0].split('.')[0]

# accessions of the 82 positives
pos_acc = [acc(l[1:].split()[0]) for l in open(ROOT/"data/amazon_viruses.fasta") if l.startswith(">")]
pos_acc = list(dict.fromkeys(pos_acc))
N=len(pos_acc)

# --- per-method predictions on the positives, 1 = viral ---
dvf=pd.read_csv(RW/"amazon82/dvf_amazon82.txt",sep="\t")
dvf['a']=dvf['name'].map(acc); dvf_set=set(dvf[(dvf.score>=0.5)&(dvf.pvalue<0.05)]['a'])
vs2=pd.read_csv(RW/"amazon82/vs2_amazon82.tsv",sep="\t"); vs2_set=set(vs2['seqname'].map(acc))
gen=pd.read_csv(RW/"amazon82/genomad_amazon82.tsv",sep="\t"); gen_set=set(gen['seq_name'].map(acc))

methods={"DeepVirFinder":dvf_set,"VirSorter2":vs2_set,"geNomad":gen_set}
pred={m:np.array([1 if a in s else 0 for a in pos_acc]) for m,s in methods.items()}

print("=== RECALL with 95% Wilson CI (n=82 positives) ===")
for m,s in methods.items():
    tp=sum(1 for a in pos_acc if a in s)
    lo,hi=proportion_confint(tp,N,method="wilson")
    print(f"  {m:18s}: {tp}/{N} = {tp/N:.1%}  95% CI [{lo:.1%}, {hi:.1%}]")
# proposed classifiers: mean +- sd over the same 20 embedding repetitions that
# produce Table 1, read from all_metrics_n20.json so that this file cannot drift
# from the published tables
prop=json.load(open(ROOT/"results/all_metrics_n20.json"))
for m in ("ExtraTrees","XGBoost"):
    mu,sd=prop[m]["rec"]
    print(f"  {m+' (20 emb)':18s}: {mu*100:.1f}% +/- {sd*100:.2f} pp")

print("\n=== Paired McNemar among the reference tools, on the positives ===")
names=list(methods)
for i in range(len(names)):
    for j in range(i+1,len(names)):
        a,b=names[i],names[j]
        n01=int(np.sum((pred[a]==0)&(pred[b]==1))); n10=int(np.sum((pred[a]==1)&(pred[b]==0)))
        tb=[[0,n01],[n10,0]]
        p=mcnemar(tb,exact=True).pvalue
        sig="yes" if p<0.05 else "no"
        print(f"  {a} vs {b}: discordant {a}+:{n10} {b}+:{n01}  p={p:.3f}  significant={sig}")

# --- specificity over the 7,542 host transcripts ---
print("\n=== SPECIFICITY with 95% Wilson CI (n=7,542 host transcripts) ===")
NEG=7542
dvfn=pd.read_csv(RW/"negatives/dvf_negv2.txt",sep="\t")
dvf_fp=int(((dvfn.score>=0.5)&(dvfn.pvalue<0.05)).sum())
vs2n=pd.read_csv(RW/"negatives/vs2_negv2.tsv",sep="\t"); vs2_fp=vs2n['seqname'].map(acc).nunique()
genn=RW/"negatives/genomad_negv2.tsv"
genn=pd.read_csv(genn,sep="\t"); gen_fp=genn['seq_name'].map(acc).nunique() if len(genn) else 0
for m,fp in [("DeepVirFinder",dvf_fp),("VirSorter2",vs2_fp),("geNomad",gen_fp)]:
    tn=NEG-fp; lo,hi=proportion_confint(tn,NEG,method="wilson")
    print(f"  {m:18s}: {tn}/{NEG} = {tn/NEG:.1%}  95% CI [{lo:.1%}, {hi:.1%}]  (FP={fp})")
for m in ("ExtraTrees","XGBoost"):
    mu,sd=prop[m]["spec"]
    print(f"  {m+' (20 emb)':18s}: {mu*100:.1f}% +/- {sd*100:.2f} pp")
