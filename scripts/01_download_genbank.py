#!/usr/bin/env python3
"""
Download viral sequences from GenBank for the ENIAC paper.

Reads accessions from data/accessions.csv (extracted from Table 1 of
Fuques et al. 2026, PeerJ 14:e20880, Amazon Mosquito Viroma) and writes
them to data/amazon_viruses.fasta with metadata in
data/amazon_viruses_metadata.csv.

Usage:
    python3 scripts/01_download_genbank.py --email YOUR_EMAIL@ufpel.edu.br
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from Bio import Entrez, SeqIO


ROOT = Path(__file__).resolve().parents[1]
ACC_CSV = ROOT / "data" / "accessions.csv"
OUT_FASTA = ROOT / "data" / "amazon_viruses.fasta"
OUT_META = ROOT / "data" / "amazon_viruses_metadata.csv"


def fetch_one(accession: str, retries: int = 3, sleep_s: float = 1.0):
    """Fetch a single GenBank record via Entrez efetch."""
    for attempt in range(retries):
        try:
            handle = Entrez.efetch(
                db="nucleotide",
                id=accession,
                rettype="fasta",
                retmode="text",
            )
            record = SeqIO.read(handle, "fasta")
            handle.close()
            return record
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = sleep_s * (attempt + 1)
            print(f"  retry {attempt + 1}/{retries} for {accession} after {wait}s ({e})",
                  flush=True)
            time.sleep(wait)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Download GenBank accessions.")
    parser.add_argument("--email", required=True,
                        help="NCBI Entrez email (required by NCBI policy).")
    parser.add_argument("--sleep", type=float, default=0.35,
                        help="Seconds between requests (NCBI limit: 3 req/s).")
    args = parser.parse_args()

    Entrez.email = args.email

    if not ACC_CSV.exists():
        print(f"error: accessions file not found: {ACC_CSV}", file=sys.stderr)
        return 1

    df = pd.read_csv(ACC_CSV)
    accessions = df["accession"].tolist()
    print(f"Will fetch {len(accessions)} accessions.", flush=True)

    records = []
    meta_rows = []
    failed = []

    for i, acc in enumerate(accessions, 1):
        print(f"[{i:3d}/{len(accessions)}] {acc} ...", end="", flush=True)
        try:
            record = fetch_one(acc)
        except Exception as e:
            print(f" FAILED ({e})", flush=True)
            failed.append(acc)
            continue
        if record is None:
            print(" FAILED (no record)", flush=True)
            failed.append(acc)
            continue
        records.append(record)
        row = df[df["accession"] == acc].iloc[0]
        meta_rows.append({
            "accession": acc,
            "species": row["species"],
            "family": row["family"],
            "novel": row["novel"],
            "pool": row["pool"],
            "length_bp": len(record.seq),
            "description": record.description,
        })
        print(f" OK ({len(record.seq)} bp)", flush=True)
        time.sleep(args.sleep)

    print(f"\nFetched {len(records)} of {len(accessions)} accessions.", flush=True)
    if failed:
        print(f"Failed: {failed}", flush=True)

    OUT_FASTA.parent.mkdir(parents=True, exist_ok=True)
    SeqIO.write(records, OUT_FASTA, "fasta")
    print(f"Wrote FASTA: {OUT_FASTA}", flush=True)

    pd.DataFrame(meta_rows).to_csv(OUT_META, index=False)
    print(f"Wrote metadata: {OUT_META}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
