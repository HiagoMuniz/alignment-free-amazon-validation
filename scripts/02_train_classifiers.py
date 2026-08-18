#!/usr/bin/env python3
"""
Retrains the two best classical classifiers of the prior work (XGBoost with
Word2Vec k=6, ExtraTrees with fastText k=6) on the 80% ZOVER pool, using the
stratified split with seed 42.

Hyperparameters are the ones selected by grid search in the prior work:
- XGBoost: n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.8
- ExtraTrees: n_estimators=200, max_depth=10, min_samples_split=2

The embeddings are loaded from the directory given by EMB_DIR, which holds the
embeddings trained on the same 80% pool in the prior work and is not
distributed with this repository. This single-run path is not the source of any
number in the paper; see scripts 11 and 12.
"""

import sys
import time
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from gensim.models import Word2Vec, FastText
from Bio import SeqIO


ENIAC_ROOT = Path(__file__).resolve().parents[1]
BRACIS_ROOT = ENIAC_ROOT.parent / "paper_bracis"
VSC_ROOT = Path(os.environ.get("VSC_ROOT", ENIAC_ROOT.parent / "Viral-Sequence-Classification"))

VIRAL_FA = VSC_ROOT / "src" / "data" / "training" / "viral.fasta"
NONVIRAL_FA = VSC_ROOT / "src" / "data" / "training" / "nonviral.fasta"
EMB_DIR = Path(os.environ.get("EMB_DIR", BRACIS_ROOT / "embeddings_test_full80"))
OUT_DIR = ENIAC_ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


def load_fasta(path: Path, label: int) -> list[dict]:
    return [
        {"seq": str(rec.seq).upper(), "label": label, "id": rec.id}
        for rec in SeqIO.parse(str(path), "fasta")
    ]


def embed_mean(seq: str, model, k: int) -> np.ndarray:
    kmers = [seq[i:i + k] for i in range(len(seq) - k + 1)]
    vecs = [model.wv[km] for km in kmers if km in model.wv]
    if not vecs:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0)


def split_pool(records: list[dict]):
    """Stratified 80/20 split (BRACIS protocol, seed=42)."""
    labels = [r["label"] for r in records]
    idx = list(range(len(records)))
    idx_pool, idx_test, _, _ = train_test_split(
        idx, labels, test_size=0.2, random_state=SEED, stratify=labels,
    )
    return [records[i] for i in idx_pool], [records[i] for i in idx_test]


def train_xgboost(X_pool, y_pool, save_path: Path):
    print("[XGBoost] Training...", flush=True)
    t0 = time.time()
    clf = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.8,
        scale_pos_weight=8.8, eval_metric="logloss",
        random_state=SEED, n_jobs=-1,
    )
    clf.fit(X_pool, y_pool)
    elapsed = time.time() - t0
    print(f"[XGBoost] Done in {elapsed:.1f}s", flush=True)
    joblib.dump(clf, save_path)
    print(f"[XGBoost] Saved: {save_path}", flush=True)
    return clf


def train_extratrees(X_pool, y_pool, save_path: Path):
    print("[ExtraTrees] Training...", flush=True)
    t0 = time.time()
    clf = ExtraTreesClassifier(
        n_estimators=200, max_depth=10, min_samples_split=2,
        class_weight="balanced",
        random_state=SEED, n_jobs=-1,
    )
    clf.fit(X_pool, y_pool)
    elapsed = time.time() - t0
    print(f"[ExtraTrees] Done in {elapsed:.1f}s", flush=True)
    joblib.dump(clf, save_path)
    print(f"[ExtraTrees] Saved: {save_path}", flush=True)
    return clf


def evaluate_on_test(clf, X_test, y_test, name: str):
    from sklearn.metrics import (
        f1_score, matthews_corrcoef, precision_score, recall_score,
        accuracy_score, roc_auc_score,
    )
    y_pred = clf.predict(X_test)
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = None
    f1 = f1_score(y_test, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
    p = precision_score(y_test, y_pred, zero_division=0)
    r = recall_score(y_test, y_pred, zero_division=0)
    a = accuracy_score(y_test, y_pred)
    print(f"\n[{name}] Test set sanity check (80/20 from ZOVER):")
    print(f"  F1={f1:.4f}  MCC={mcc:.4f}  P={p:.4f}  R={r:.4f}  Acc={a:.4f}  ROC-AUC={auc:.4f}" if auc else
          f"  F1={f1:.4f}  MCC={mcc:.4f}  P={p:.4f}  R={r:.4f}  Acc={a:.4f}")
    return {"f1": f1, "mcc": mcc, "precision": p, "recall": r, "accuracy": a, "roc_auc": auc}


def main() -> int:
    np.random.seed(SEED)

    print(f"[DATA] Loading ZOVER FASTA...", flush=True)
    virals = load_fasta(VIRAL_FA, label=1)
    nonvirals = load_fasta(NONVIRAL_FA, label=0)
    records = virals + nonvirals
    print(f"  {len(virals)} viral, {len(nonvirals)} non-viral, total={len(records)}",
          flush=True)

    pool, test = split_pool(records)
    print(f"[SPLIT] pool={len(pool)} (80%), test={len(test)} (20%)", flush=True)

    # XGBoost (k=6, word2vec) and ExtraTrees (k=6, fast_text) share k=6 but
    # use different embeddings.
    w2v_path = EMB_DIR / "XGBoost_k6_word2vec.model"
    ft_path = EMB_DIR / "ExtraTrees_k6_fast_text.model"
    print(f"[EMB] Loading Word2Vec from {w2v_path.name}", flush=True)
    w2v = Word2Vec.load(str(w2v_path))
    print(f"[EMB] Loading fastText from {ft_path.name}", flush=True)
    ft = FastText.load(str(ft_path))

    # Vectorize with each embedding
    print(f"[VEC] Vectorizing with Word2Vec (XGBoost) ...", flush=True)
    X_pool_w2v = np.vstack([embed_mean(r["seq"], w2v, 6) for r in pool])
    X_test_w2v = np.vstack([embed_mean(r["seq"], w2v, 6) for r in test])
    print(f"[VEC] Vectorizing with fastText (ExtraTrees) ...", flush=True)
    X_pool_ft = np.vstack([embed_mean(r["seq"], ft, 6) for r in pool])
    X_test_ft = np.vstack([embed_mean(r["seq"], ft, 6) for r in test])

    y_pool = np.array([r["label"] for r in pool])
    y_test = np.array([r["label"] for r in test])

    # Train + save
    xgb = train_xgboost(X_pool_w2v, y_pool, OUT_DIR / "xgboost_k6_word2vec.joblib")
    et = train_extratrees(X_pool_ft, y_pool, OUT_DIR / "extratrees_k6_fasttext.joblib")

    # Sanity check on test set: should match paper numbers
    # (paper: XGBoost F1=0.891, ExtraTrees F1=0.894)
    evaluate_on_test(xgb, X_test_w2v, y_test, "XGBoost")
    evaluate_on_test(et, X_test_ft, y_test, "ExtraTrees")

    # Mirror embeddings to models so the inference script
    # finds everything in one place
    import shutil
    for src in [w2v_path, ft_path,
                w2v_path.with_suffix(".model.syn1neg.npy"),
                w2v_path.with_suffix(".model.wv.vectors.npy"),
                ft_path.with_suffix(".model.wv.vectors_ngrams.npy")]:
        if src.exists():
            shutil.copy(src, OUT_DIR / src.name)
            print(f"[COPY] {src.name} -> {OUT_DIR}", flush=True)

    print(f"\n[OK] Models saved to {OUT_DIR}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
