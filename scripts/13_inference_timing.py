#!/usr/bin/env python3
"""
Measures inference wall-clock time over the full validation set of 7,624
sequences (82 viral plus 7,542 host transcripts), separately for each proposed
model, on CPU.

The two models do not share an embedding: ExtraTrees uses fastText and XGBoost
uses Word2Vec. Vectorization therefore cannot be reused between them and is
timed once per model.

Model loading is timed apart from inference, since loading is a one-off cost
that does not scale with the number of input sequences.

Requires the classifiers and embeddings produced by script 02, which are not
distributed with this repository. Point MODELS_DIR at them.

Usage:
  MODELS_DIR=/path/to/models python3 scripts/13_inference_timing.py [--repeats 3]
"""

import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
from Bio import SeqIO
from gensim.models import FastText, Word2Vec

E = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("MODELS_DIR", E / "models"))
K = 6

CONFIGS = [
    ("ExtraTrees", "fastText", FastText,
     "ExtraTrees_k6_fast_text.model", "extratrees_k6_fasttext.joblib"),
    ("XGBoost", "Word2Vec", Word2Vec,
     "XGBoost_k6_word2vec.model", "xgboost_k6_word2vec.joblib"),
]


def kmers(seq):
    return [seq[i:i + K] for i in range(len(seq) - K + 1)]


def mean_vec(seq, emb):
    v = [emb.wv[k] for k in kmers(seq) if k in emb.wv]
    return np.mean(v, axis=0) if v else np.zeros(emb.vector_size, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3,
                    help="Timed repetitions per model; the median is reported.")
    args = ap.parse_args()

    pos = [str(r.seq).upper() for r in SeqIO.parse(str(E / "data/amazon_viruses.fasta"), "fasta")]
    neg = [str(r.seq).upper() for r in SeqIO.parse(str(E / "data/amazon_negatives_v2.fasta"), "fasta")]
    seqs = pos + neg
    print(f"Input: {len(seqs)} sequences ({len(pos)} viral, {len(neg)} host)", flush=True)
    print(f"Models: {MODELS_DIR}", flush=True)
    print(f"Repeats per model: {args.repeats}\n", flush=True)

    out = {"n_sequences": len(seqs), "repeats": args.repeats, "models": {}}

    for clf_name, emb_name, loader, emb_file, clf_file in CONFIGS:
        t0 = time.perf_counter()
        emb = loader.load(str(MODELS_DIR / emb_file))
        clf = joblib.load(MODELS_DIR / clf_file)
        t_load = time.perf_counter() - t0

        t_vec, t_pred = [], []
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            X = np.vstack([mean_vec(s, emb) for s in seqs])
            t_vec.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            clf.predict(X)
            t_pred.append(time.perf_counter() - t0)

        vec_med = float(np.median(t_vec))
        pred_med = float(np.median(t_pred))
        out["models"][f"{clf_name} ({emb_name})"] = {
            "load_s": round(t_load, 2),
            "vectorize_s": round(vec_med, 2),
            "predict_s": round(pred_med, 3),
            "inference_s": round(vec_med + pred_med, 2),
            "vectorize_all_runs_s": [round(t, 2) for t in t_vec],
        }
        print(f"{clf_name} ({emb_name}):", flush=True)
        print(f"  load        {t_load:7.2f} s  (one-off, does not scale with input)", flush=True)
        print(f"  vectorize   {vec_med:7.2f} s  (median of {args.repeats}: "
              f"{', '.join(f'{t:.1f}' for t in t_vec)})", flush=True)
        print(f"  predict     {pred_med:7.3f} s", flush=True)
        print(f"  inference   {vec_med + pred_med:7.2f} s  (vectorize + predict)\n", flush=True)

    total = sum(m["inference_s"] for m in out["models"].values())
    out["both_models_inference_s"] = round(total, 2)
    for name, m in out["models"].items():
        print(f"Inference only, {name}: {m['inference_s']:.1f} s", flush=True)
    print(f"Inference only, both models: {total:.1f} s", flush=True)

    dest = E / "results/inference_timing.json"
    json.dump(out, open(dest, "w"), indent=2)
    print(f"\nOK -> {dest}", flush=True)


if __name__ == "__main__":
    main()
