#!/usr/bin/env python3
"""
Downloads a diverse, non-redundant pool of real host transcripts from NCBI to
serve as the negative class of the validation set.

Each negative is one distinct transcript, used exactly once, with no
fragmentation and no overlap. They are transcripts (mRNA, rRNA and genes),
which matches the molecule type of the RNA viruses of the reference study.

Every sequence carries a component label in its header, so that the
composition can be reweighted later without downloading again.

Note that the components are queried by organism, not by accession, so a fresh
download will not return exactly the set used in the paper as the database
grows. The authoritative list is data/amazon_negatives_v2_metadata.csv.

Output:
  data/negatives_v2/<component>.fasta   one file per component
  data/amazon_negatives_v2.fasta        combined
  data/amazon_negatives_v2_metadata.csv accession, component, length

Usage:
  python3 scripts/06_download_negatives_v2.py --email YOUR_EMAIL
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from Bio import Entrez, SeqIO

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "data" / "negatives_v2"
OUTDIR.mkdir(parents=True, exist_ok=True)
COMBINED = ROOT / "data" / "amazon_negatives_v2.fasta"
META = ROOT / "data" / "amazon_negatives_v2_metadata.csv"

# component -> (query, target number of sequences)
COMPONENTS = {
    "Anopheles_darlingi_mRNA": ('"Anopheles darlingi"[Organism] AND biomol_mrna[PROP]', 3000),
    "Anopheles_genus_mRNA":    ('"Anopheles"[Organism] AND biomol_mrna[PROP] AND refseq[filter]', 1500),
    "Culex_mRNA":              ('"Culex"[Organism] AND biomol_mrna[PROP] AND refseq[filter]', 1000),
    "Aedes_aegypti_mRNA":      ('"Aedes aegypti"[Organism] AND biomol_mrna[PROP] AND refseq[filter]', 1000),
    "Wolbachia_genes":         ('"Wolbachia"[Organism] AND refseq[filter] AND biomol_genomic[PROP]', 1000),
    "Culicidae_rRNA":          ('"Culicidae"[Organism] AND biomol_rrna[PROP]', 700),
}

BATCH = 300


def fetch_component(name, query, target):
    print(f"\n[{name}] esearch (target {target})...", flush=True)
    h = Entrez.esearch(db="nucleotide", term=query, retmax=target)
    r = Entrez.read(h); h.close()
    ids = r["IdList"]
    print(f"[{name}] {len(ids)} ids retrieved (of {r["Count"]} available)", flush=True)
    if not ids:
        return []

    # epost, to download in batches
    post = Entrez.read(Entrez.epost(db="nucleotide", id=",".join(ids)))
    webenv, qkey = post["WebEnv"], post["QueryKey"]

    records = []
    for start in range(0, len(ids), BATCH):
        for attempt in range(3):
            try:
                h = Entrez.efetch(db="nucleotide", rettype="fasta", retmode="text",
                                  retstart=start, retmax=BATCH,
                                  webenv=webenv, query_key=qkey)
                recs = list(SeqIO.parse(h, "fasta")); h.close()
                records.extend(recs)
                print(f"  [{name}] {len(records)}/{len(ids)}", flush=True)
                break
            except Exception as e:
                print(f"  [{name}] retry {attempt+1}: {e}", flush=True)
                time.sleep(3)
        time.sleep(0.4)

    SeqIO.write(records, str(OUTDIR / f"{name}.fasta"), "fasta")
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    args = ap.parse_args()
    Entrez.email = args.email

    all_recs = []
    meta_rows = []
    for name, (query, target) in COMPONENTS.items():
        recs = fetch_component(name, query, target)
        for rec in recs:
            all_recs.append(rec)
            meta_rows.append({"accession": rec.id, "component": name,
                              "length_bp": len(rec.seq)})

    # deduplicate by accession, in case queries overlap
    seen = set(); uniq = []
    for rec, row in zip(all_recs, meta_rows):
        if rec.id in seen:
            continue
        seen.add(rec.id); uniq.append((rec, row))
    recs_u = [u[0] for u in uniq]
    SeqIO.write(recs_u, str(COMBINED), "fasta")
    pd.DataFrame([u[1] for u in uniq]).to_csv(META, index=False)

    print(f"\n[OK] {len(recs_u)} distinct transcripts written to {COMBINED}")
    df = pd.DataFrame([u[1] for u in uniq])
    print("\nFinal composition:")
    print(df.groupby("component").agg(n=("accession","size"),
          comp_medio=("length_bp","mean")).round(0).to_string())
    print(f"\nLength: min={df.length_bp.min()} max={df.length_bp.max()} "
          f"mediana={int(df.length_bp.median())}")


if __name__ == "__main__":
    main()
