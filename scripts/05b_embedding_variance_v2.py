#!/usr/bin/env python3
"""
Quantifica a variancia do desempenho causada pela estocasticidade do
embedding Word2Vec (gensim sem seed fixa, workers default).

Para N rodadas independentes:
  1. treina Word2Vec fresco no pool de 80% do ZOVER (k=6)
  2. treina XGBoost (mesmos hiperparametros do paper) sobre esse embedding
  3. mede:
     - F1 in-distribution no test set 20% do ZOVER
     - Recall OOD nos 82 virus amazonicos
     - Especificidade OOD nos 7542 transcritos de hospedeiro (classe negativa v2)

Hipotese: F1 in-distribution e estavel entre rodadas, mas o recall OOD
oscila (foi o que vimos: 67/82 vs 81/82 em duas rodadas distintas).

Saida: results/embedding_variance_v2.csv (uma linha por rodada) + resumo.

Uso:
    python3 05_embedding_variance.py --n 20
"""

import argparse
import time
import os
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

ENIAC = Path(__file__).resolve().parents[1]
VSC = Path(os.environ.get("VSC_ROOT", ENIAC.parent / "Viral-Sequence-Classification"))
VIRAL = VSC / "src/data/training/viral.fasta"
NONVIRAL = VSC / "src/data/training/nonviral.fasta"
AMAZON_POS = ENIAC / "data/amazon_viruses.fasta"
AMAZON_NEG = ENIAC / "data/amazon_negatives_v2.fasta"

K = 6
SEED_SPLIT = 42  # split fixo, igual ao BRACIS, para todas as rodadas


def kmers(seq):
    seq = str(seq).upper()
    return [seq[i:i+K] for i in range(len(seq)-K+1)]


def load_fasta(path, label):
    return [(str(r.seq).upper(), label) for r in SeqIO.parse(str(path), "fasta")]


def mean_vec(seq, model):
    vs = [model.wv[k] for k in kmers(seq) if k in model.wv]
    return np.mean(vs, axis=0) if vs else np.zeros(model.vector_size, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="numero de rodadas de embedding")
    args = ap.parse_args()

    print("[DATA] carregando ZOVER...", flush=True)
    data = load_fasta(VIRAL, 1) + load_fasta(NONVIRAL, 0)
    seqs = [d[0] for d in data]
    labels = [d[1] for d in data]

    # split 80/20 fixo (mesmo de sempre)
    idx = list(range(len(seqs)))
    idx_pool, idx_test, y_pool, y_test = train_test_split(
        idx, labels, test_size=0.2, random_state=SEED_SPLIT, stratify=labels)
    pool_seqs = [seqs[i] for i in idx_pool]
    test_seqs = [seqs[i] for i in idx_test]
    y_pool = np.array([labels[i] for i in idx_pool])
    y_test = np.array([labels[i] for i in idx_test])
    print(f"[DATA] pool={len(pool_seqs)} test={len(test_seqs)}", flush=True)

    print("[DATA] carregando amazonicos (OOD)...", flush=True)
    amazon_pos = [str(r.seq).upper() for r in SeqIO.parse(str(AMAZON_POS), "fasta")]
    amazon_neg = [str(r.seq).upper() for r in SeqIO.parse(str(AMAZON_NEG), "fasta")]
    print(f"[DATA] OOD: {len(amazon_pos)} virus, {len(amazon_neg)} host", flush=True)

    # pre-computa k-mers do pool (reusado em cada rodada para treinar embedding)
    pool_kmers = [kmers(s) for s in pool_seqs]

    rows = []
    for run in range(1, args.n + 1):
        t0 = time.time()
        # embedding fresco: SEM seed fixa, workers default -> captura a
        # nao-determinancia real do pipeline
        emb = Word2Vec(sentences=pool_kmers, vector_size=100, window=5,
                       min_count=5, sg=1)

        Xp = np.vstack([mean_vec(s, emb) for s in pool_seqs])
        clf = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                            subsample=0.8, scale_pos_weight=8.8,
                            eval_metric="logloss", random_state=42, n_jobs=-1)
        clf.fit(Xp, y_pool)

        # in-distribution F1
        Xt = np.vstack([mean_vec(s, emb) for s in test_seqs])
        f1_id = f1_score(y_test, clf.predict(Xt), zero_division=0)

        # OOD recall (positivos)
        Xpos = np.vstack([mean_vec(s, emb) for s in amazon_pos])
        rec_ood = clf.predict(Xpos).mean()

        # OOD especificidade (negativos)
        Xneg = np.vstack([mean_vec(s, emb) for s in amazon_neg])
        spec_ood = 1 - clf.predict(Xneg).mean()

        dt = time.time() - t0
        rows.append(dict(run=run, f1_indist=f1_id, recall_ood=rec_ood,
                         specificity_ood=spec_ood, secs=dt))
        print(f"  run {run:2d}/{args.n}: F1_indist={f1_id:.4f}  "
              f"recall_OOD={rec_ood:.4f}  spec_OOD={spec_ood:.4f}  ({dt:.0f}s)",
              flush=True)
        # salva incremental
        pd.DataFrame(rows).to_csv(ENIAC/"results/embedding_variance_v2.csv", index=False)

    df = pd.DataFrame(rows)
    print("\n=== RESUMO (", args.n, "rodadas) ===")
    for col, name in [("f1_indist", "F1 in-distribution"),
                      ("recall_ood", "Recall OOD (82 virus)"),
                      ("specificity_ood", "Especificidade OOD (8200 host)")]:
        print(f"  {name:32s}: media={df[col].mean():.4f}  std={df[col].std():.4f}  "
              f"min={df[col].min():.4f}  max={df[col].max():.4f}")
    print(f"\n[OK] salvo em results/embedding_variance_v2.csv")


if __name__ == "__main__":
    main()
