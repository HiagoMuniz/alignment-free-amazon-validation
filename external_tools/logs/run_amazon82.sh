#!/bin/bash
# ============================================================
# Ferramentas externas (DVF, VS2, geNomad) nas 82 sequencias
# virais amazonicas (Fuques et al. 2026) para o paper ENIAC.
# Adaptado dos scripts run_*.sh do BRACIS (related_works/).
# ============================================================
set -uo pipefail

RW="/home/carloscalage/mestrado/Mestrado/related_works"
FASTA="$RW/eniac_amazon/amazon82.fasta"
OUT="$RW/eniac_amazon"
NPROC=$(nproc)

source /home/carloscalage/anaconda3/etc/profile.d/conda.sh

N_SEQS=$(grep -c "^>" "$FASTA")
echo "Input: $FASTA ($N_SEQS sequencias)"
echo "Cores: $NPROC"
echo ""

# ---------------- DeepVirFinder ----------------
run_dvf() {
    echo "=== [1/3] DeepVirFinder ==="
    local odir="$OUT/dvf_output"
    rm -rf "$odir"; mkdir -p "$odir"
    conda activate dvf
    local t0=$(date +%s)
    python "$RW/DeepVirFinder/dvf.py" \
        -i "$FASTA" \
        -o "$odir" \
        -l 1 \
        -c "$NPROC" 2>&1 | tail -5
    local t1=$(date +%s)
    echo "DVF wall time: $((t1-t0))s"
    conda deactivate
    local pred=$(ls "$odir"/*dvfpred.txt 2>/dev/null | head -1)
    [ -n "$pred" ] && echo "DVF output: $pred ($(tail -n +2 "$pred" | wc -l) preds)"
    echo ""
}

# ---------------- VirSorter2 ----------------
run_vs2() {
    echo "=== [2/3] VirSorter2 ==="
    local odir="$OUT/vs2_output"
    rm -rf "$odir"; mkdir -p "$odir"
    conda activate vs2
    local t0=$(date +%s)
    virsorter run \
        -w "$odir" \
        -i "$FASTA" \
        --min-length 0 \
        --min-score 0.5 \
        --include-groups dsDNAphage,ssDNA,NCLDV,RNA,lavidaviridae \
        -j "$NPROC" \
        --db-dir "$RW/vs2_db" \
        all 2>&1 | tail -8
    local t1=$(date +%s)
    echo "VS2 wall time: $((t1-t0))s"
    conda deactivate
    [ -f "$odir/final-viral-score.tsv" ] && \
        echo "VS2 detected: $(tail -n +2 "$odir/final-viral-score.tsv" | wc -l) / $N_SEQS"
    echo ""
}

# ---------------- geNomad ----------------
run_genomad() {
    echo "=== [3/3] geNomad ==="
    local odir="$OUT/genomad_output"
    rm -rf "$odir"; mkdir -p "$odir"
    conda activate genomad
    local t0=$(date +%s)
    genomad end-to-end \
        "$FASTA" \
        "$odir" \
        "$RW/genomad_db/genomad_db" \
        --cleanup \
        --enable-score-calibration \
        --threads "$NPROC" 2>&1 | tail -8
    local t1=$(date +%s)
    echo "geNomad wall time: $((t1-t0))s"
    conda deactivate
    local summ=$(find "$odir" -name "*virus_summary.tsv" 2>/dev/null | head -1)
    [ -n "$summ" ] && echo "geNomad detected: $(tail -n +2 "$summ" | wc -l) / $N_SEQS"
    echo ""
}

case "${1:-all}" in
    dvf) run_dvf ;;
    vs2) run_vs2 ;;
    genomad) run_genomad ;;
    all) run_dvf; run_vs2; run_genomad ;;
esac

echo "=== Concluido. Outputs em $OUT/{dvf,vs2,genomad}_output ==="
