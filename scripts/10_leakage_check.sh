#!/bin/bash
# Leakage check: are the 82 Amazonian viruses near-duplicates of anything in
# the ZOVER training set? Nucleotide MMseqs2 search, Amazonian against training.
# Leakage criterion: a hit covering at least 30% of the query.
# Expected result: no hit reaching 30% query coverage.
set -e
# Requer MMseqs2 no PATH. VSC_ROOT aponta para o repositorio do pipeline
# (Viral-Sequence-Classification), que traz o conjunto de treino do ZOVER.
ENIAC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VSC="${VSC_ROOT:-$ENIAC/../Viral-Sequence-Classification}"
OUT="$ENIAC/results/leakage_check"
mkdir -p "$OUT"

# 1) every hit, with no coverage cutoff, to characterise what is there
mmseqs easy-search \
  "$ENIAC/data/amazon_viruses.fasta" \
  "$VSC/src/data/training/viral.fasta" \
  "$OUT/amazon_vs_zover_all.m8" "$OUT/tmp1" \
  --search-type 3 -e 1e-5 \
  --format-output "query,target,pident,qcov,alnlen,evalue" 2>/dev/null

# 2) requiring at least 30% query coverage, the leakage criterion
mmseqs easy-search \
  "$ENIAC/data/amazon_viruses.fasta" \
  "$VSC/src/data/training/viral.fasta" \
  "$OUT/amazon_vs_zover_cov30.m8" "$OUT/tmp2" \
  --search-type 3 -c 0.3 --cov-mode 0 \
  --format-output "query,target,pident,qcov,alnlen,evalue" 2>/dev/null

rm -rf "$OUT/tmp1" "$OUT/tmp2"
echo "Total hits (no cutoff): $(wc -l < "$OUT/amazon_vs_zover_all.m8")"
echo "Hits with >=30% query coverage: $(wc -l < "$OUT/amazon_vs_zover_cov30.m8")"
echo "Longest alignment among all hits: $(sort -t$'\t' -k5 -rn "$OUT/amazon_vs_zover_all.m8" | head -1 | cut -f5) bp"
