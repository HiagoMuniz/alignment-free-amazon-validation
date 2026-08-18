#!/usr/bin/env python3
"""
Baixa um pool DIVERSO e NAO-REDUNDANTE de transcritos reais do NCBI para
servir de classe negativa de validacao (caminho B, v2).

Diferenca crucial vs a v1 (8200 falhos): cada negativo e UM transcrito
real distinto, usado UMA vez. Sem fatiamento, sem sobreposicao. E sao
TRANSCRITOS (mRNA/rRNA/genes), casando o tipo molecular dos virus de RNA
do estudo (metatranscriptomica).

Cada sequencia recebe um rotulo de componente no header, para permitir
reponderar a composicao depois sem novo download.

Saida:
  data/negatives_v2/<componente>.fasta  (um por componente)
  data/amazon_negatives_v2.fasta         (combinado)
  data/amazon_negatives_v2_metadata.csv  (accession, componente, comprimento)

Uso:
  python3 06_download_negatives_v2.py --email SEU_EMAIL
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

# componente -> (query, alvo de sequencias)
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
    print(f"\n[{name}] esearch (alvo {target})...", flush=True)
    h = Entrez.esearch(db="nucleotide", term=query, retmax=target)
    r = Entrez.read(h); h.close()
    ids = r["IdList"]
    print(f"[{name}] {len(ids)} ids obtidos (de {r['Count']} disponiveis)", flush=True)
    if not ids:
        return []

    # epost para baixar em lotes
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

    # dedup por accession (caso haja overlap entre queries)
    seen = set(); uniq = []
    for rec, row in zip(all_recs, meta_rows):
        if rec.id in seen:
            continue
        seen.add(rec.id); uniq.append((rec, row))
    recs_u = [u[0] for u in uniq]
    SeqIO.write(recs_u, str(COMBINED), "fasta")
    pd.DataFrame([u[1] for u in uniq]).to_csv(META, index=False)

    print(f"\n[OK] {len(recs_u)} transcritos distintos salvos em {COMBINED}")
    df = pd.DataFrame([u[1] for u in uniq])
    print("\nComposicao final:")
    print(df.groupby("component").agg(n=("accession","size"),
          comp_medio=("length_bp","mean")).round(0).to_string())
    print(f"\nComprimento: min={df.length_bp.min()} max={df.length_bp.max()} "
          f"mediana={int(df.length_bp.median())}")


if __name__ == "__main__":
    main()
