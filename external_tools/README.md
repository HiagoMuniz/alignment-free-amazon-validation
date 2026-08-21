# Reference tools: versions, command lines and raw outputs

The three reference tools were run on the same workstation as the proposed
classifiers, a 14-core CPU exposing 20 logical threads, 64 GB of RAM and no
GPU, over the same sequences. Each tool was given all 20 threads.

The 82 viral sequences and the 7,542 host transcripts were submitted in two
separate runs, one FASTA each. `amazon82/` holds the outputs for the viral set
and `negatives/` the outputs for the host set. Together they cover the full set
of 7,624 sequences.

## geNomad 1.11.2, database v1.9

```bash
genomad end-to-end input.fasta outdir genomad_db \
    --cleanup \
    --enable-score-calibration \
    --threads 20
```

A sequence counts as viral when it appears in `<prefix>_virus_summary.tsv`,
which is geNomad's own calibrated decision. No additional cutoff was applied.

## VirSorter2 2.2.4

Database installed with `virsorter setup` for this version.

```bash
virsorter run \
    -w outdir \
    -i input.fasta \
    --min-length 0 \
    --min-score 0.5 \
    --include-groups dsDNAphage,ssDNA,NCLDV,RNA,lavidaviridae \
    -j 20 \
    --db-dir vs2_db \
    all
```

A sequence counts as viral when it appears in `final-viral-score.tsv`. All five
classifier groups were enabled, including RNA, which is the relevant one for
this virome.

## DeepVirFinder, GitHub commit 2635ec8

The tool publishes no versioned release. The bundled pretrained models under
`models/` were used, with no retraining.

```bash
python dvf.py -i input.fasta -o outdir -l 1 -c 20
```

A sequence counts as viral when `score >= 0.5` and `pvalue < 0.05`, the
criterion recommended by the authors.

## Wall-clock time

Measured on the same workstation, all tools on CPU with 20 threads. Every figure
below is a line in `logs/`, timed by the run scripts themselves with `date +%s`
around each tool.

| Tool | 82 viral | 7,542 host | Total, 7,624 |
|---|---|---|---|
| geNomad | 47 s | 135 s | 182 s (3.0 min) |
| DeepVirFinder | 80 s | 178 s | 258 s (4.3 min) |
| VirSorter2 | 198 s | 2,769 s | 2,967 s (49.5 min) |

For comparison, the proposed classifiers process the full set of 7,624
sequences, including embedding generation, in about 13 s per model on the same
CPU, or about 27 s for both. See `../results/inference_timing.json`, produced by
`../scripts/13_inference_timing.py`.

The comparison is conservative with respect to parallelism: the three reference
tools were given all 20 threads, whereas the dominant cost of the proposed
pipeline, the vectorization of each sequence into mean-pooled k-mer embeddings,
runs on a single core.

## Provenance of the logs

`logs/` holds the two run logs verbatim, together with the scripts that produced
them. They are evidence artefacts and have not been edited, which is why they
still carry the absolute paths of the machine that ran them.

| File | Input | Sequences | Finished |
|---|---|---|---|
| `logs/run_amazon82.log` | `amazon82.fasta` | 82 viral | 2026-06-20 |
| `logs/run_negatives.log` | `negv2.fasta` | 7,542 host | 2026-06-21 |

Each log opens with the absolute path of its input FASTA and the sequence count,
so the negative-class run can be told apart from an earlier, discarded run over
a different negative set. The two inputs are byte-identical to
`../data/amazon_viruses.fasta` and `../data/amazon_negatives_v2.fasta`; the
checksums match.

The viral run predates the negative run because the positive class never
changed; only the negative class was rebuilt.

## Environments

`environments/` holds `conda env export --no-builds` for the three tool
environments, so the versions above are recorded rather than asserted. The
`prefix` line was removed from each file, since it only names a local path.

## One sequence missing from the DeepVirFinder output

DeepVirFinder returned 7,541 predictions for the 7,542 host transcripts. The
missing one is `NZ_JAATLA010000015.1`, a 3,906 bp *Wolbachia pipientis*
scaffold of which 1,181 bases are ambiguous (N), which the tool's encoder does
not handle. In the specificity calculation the denominator remains 7,542, so
that sequence is counted as correctly rejected. Counting it as a false positive
instead would change DeepVirFinder's specificity from 79.62 % to 79.61 %.
