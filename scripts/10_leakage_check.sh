#!/bin/bash
# Verificacao de vazamento: os 82 virus amazonicos sao quase-duplicatas
# de algo no treino do ZOVER? Busca MMseqs2 (nucleotideo) amazon vs treino.
# Criterio de vazamento: hit com >=30% de cobertura do query.
# Resultado esperado: 0 hits >=30% cobertura (sem vazamento).
set -e
# Requer MMseqs2 no PATH. VSC_ROOT aponta para o repositorio do pipeline
# (Viral-Sequence-Classification), que traz o conjunto de treino do ZOVER.
ENIAC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VSC="${VSC_ROOT:-$ENIAC/../Viral-Sequence-Classification}"
OUT="$ENIAC/results/leakage_check"
mkdir -p "$OUT"

# 1) todos os hits (sem corte de cobertura), para caracterizar
mmseqs easy-search \
  "$ENIAC/data/amazon_viruses.fasta" \
  "$VSC/src/data/training/viral.fasta" \
  "$OUT/amazon_vs_zover_all.m8" "$OUT/tmp1" \
  --search-type 3 -e 1e-5 \
  --format-output "query,target,pident,qcov,alnlen,evalue" 2>/dev/null

# 2) exigindo >=30% de cobertura do query (criterio de vazamento)
mmseqs easy-search \
  "$ENIAC/data/amazon_viruses.fasta" \
  "$VSC/src/data/training/viral.fasta" \
  "$OUT/amazon_vs_zover_cov30.m8" "$OUT/tmp2" \
  --search-type 3 -c 0.3 --cov-mode 0 \
  --format-output "query,target,pident,qcov,alnlen,evalue" 2>/dev/null

rm -rf "$OUT/tmp1" "$OUT/tmp2"
echo "Hits totais (sem corte): $(wc -l < "$OUT/amazon_vs_zover_all.m8")"
echo "Hits com >=30% cobertura do query: $(wc -l < "$OUT/amazon_vs_zover_cov30.m8")"
echo "Maior alnlen entre os hits totais: $(sort -t$'\t' -k5 -rn "$OUT/amazon_vs_zover_all.m8" | head -1 | cut -f5) bp"
