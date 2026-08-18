#!/usr/bin/env python3
"""
Runs inference with the two ZOVER-trained classifiers over the 82 Amazonian
viral sequences of Fuques et al. 2026.

Outputs:
- results/amazon_predictions.csv: one row per sequence, with columns
  accession, species, family, novel, length_bp,
  xgb_proba, xgb_pred, et_proba, et_pred, agree
- results/amazon_summary.csv: aggregates by family and by model

Requires the classifiers and embeddings produced by script 02, which are not
distributed with this repository.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from Bio import SeqIO
from gensim.models import Word2Vec, FastText


ENIAC_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ENIAC_ROOT / "models"
DATA_DIR = ENIAC_ROOT / "data"
RES_DIR = ENIAC_ROOT / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)

K = 6
THRESHOLD = 0.5


def embed_mean(seq: str, model, k: int) -> np.ndarray:
    kmers = [seq[i:i + k] for i in range(len(seq) - k + 1)]
    vecs = [model.wv[km] for km in kmers if km in model.wv]
    if not vecs:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0)


def main() -> int:
    print("[LOAD] Loading sequences and metadata...", flush=True)
    fasta = DATA_DIR / "amazon_viruses.fasta"
    meta = pd.read_csv(DATA_DIR / "amazon_viruses_metadata.csv")

    records = list(SeqIO.parse(str(fasta), "fasta"))
    # Map id -> seq
    seq_by_id = {rec.id.split(".")[0]: str(rec.seq).upper() for rec in records}
    meta["seq"] = meta["accession"].map(seq_by_id)
    missing = meta["seq"].isna().sum()
    if missing:
        print(f"  warning: {missing} accession(s) without sequence",
              file=sys.stderr)
    print(f"  {len(meta)} sequences loaded", flush=True)

    print("[LOAD] Word2Vec embedding (XGBoost)...", flush=True)
    w2v = Word2Vec.load(str(MODELS_DIR / "XGBoost_k6_word2vec.model"))
    print("[LOAD] fastText embedding (ExtraTrees)...", flush=True)
    ft = FastText.load(str(MODELS_DIR / "ExtraTrees_k6_fast_text.model"))

    print("[LOAD] Classifiers...", flush=True)
    xgb = joblib.load(MODELS_DIR / "xgboost_k6_word2vec.joblib")
    et = joblib.load(MODELS_DIR / "extratrees_k6_fasttext.joblib")

    print("[EMBED] Vectorizing sequences with both embeddings...", flush=True)
    X_w2v = np.vstack([embed_mean(s, w2v, K) for s in meta["seq"]])
    X_ft = np.vstack([embed_mean(s, ft, K) for s in meta["seq"]])

    print("[PRED] Running inference...", flush=True)
    xgb_proba = xgb.predict_proba(X_w2v)[:, 1]
    xgb_pred = (xgb_proba >= THRESHOLD).astype(int)
    et_proba = et.predict_proba(X_ft)[:, 1]
    et_pred = (et_proba >= THRESHOLD).astype(int)

    out = meta[["accession", "species", "family", "novel", "length_bp"]].copy()
    out["xgb_proba"] = xgb_proba
    out["xgb_pred"] = xgb_pred
    out["et_proba"] = et_proba
    out["et_pred"] = et_pred
    out["agree"] = (xgb_pred == et_pred).astype(int)

    pred_csv = RES_DIR / "amazon_predictions.csv"
    out.to_csv(pred_csv, index=False)
    print(f"  per-sequence predictions: {pred_csv}", flush=True)

    # Summary
    summary_rows = []
    for model in ("xgb", "et"):
        pcol, predcol = f"{model}_proba", f"{model}_pred"
        for fam, sub in out.groupby("family"):
            n = len(sub)
            n_pos = int(sub[predcol].sum())
            recall = n_pos / n
            summary_rows.append({
                "model": model.upper(),
                "family": fam,
                "n": n,
                "detected_viral": n_pos,
                "recall": recall,
                "mean_proba": sub[pcol].mean(),
            })
        # Global summary
        n = len(out)
        n_pos = int(out[predcol].sum())
        summary_rows.append({
            "model": model.upper(),
            "family": "ALL",
            "n": n,
            "detected_viral": n_pos,
            "recall": n_pos / n,
            "mean_proba": out[pcol].mean(),
        })

    summary = pd.DataFrame(summary_rows)
    summary_csv = RES_DIR / "amazon_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"  summary by family: {summary_csv}", flush=True)

    # Print overall
    print(f"\n=== Overall recall (threshold={THRESHOLD}) ===")
    for model in ("XGB", "ET"):
        global_row = summary[(summary["model"] == model) & (summary["family"] == "ALL")].iloc[0]
        print(f"  {model}: {int(global_row['detected_viral'])}/{int(global_row['n'])} "
              f"({global_row['recall']:.1%}), mean_proba={global_row['mean_proba']:.3f}")

    print(f"\n=== Recall by family ===")
    for fam, sub in out.groupby("family"):
        xgb_n = int(sub["xgb_pred"].sum())
        et_n = int(sub["et_pred"].sum())
        n = len(sub)
        print(f"  {fam:18s} (n={n:2d}): XGB={xgb_n}/{n} ({xgb_n/n:.0%})  "
              f"ET={et_n}/{n} ({et_n/n:.0%})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
