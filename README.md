# Alignment-Free Viral Metagenomics: External Validation of K-mer Based Ensemble Classifiers on an Independent Amazonian Dataset

Reproduction package for the ENIAC 2026 paper of the same name.

An alignment-free viral classifier previously proposed for metagenomic data,
based on k-mer embeddings (Word2Vec and fastText) and ensemble classifiers
(ExtraTrees and XGBoost), is trained on the ZOVER repository and evaluated,
without any adaptation to the target domain, on a fully independent field
dataset: the RNA mosquito virome of the Amazon Basin published by
[Fuques et al. 2026](https://peerj.com/articles/20880/) (PeerJ 14:e20880).

The validation set has 7,624 sequences: 82 homology-confirmed viruses from ten
families and 7,542 host transcripts, an imbalance of roughly 1 virus per 92
hosts. Everything needed to reproduce the reported numbers is in this
repository, except the training corpus and the trained artefacts, which are
covered in [Not included here](#not-included-here).

## Headline results

Out-of-distribution performance, mean ± standard deviation over 20 independent
embedding repetitions:

| Model | Recall | Specificity | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| ExtraTrees (fastText) | 84.6 ± 1.1 % | 88.7 ± 1.2 % | 0.948 ± 0.005 | 0.507 ± 0.026 |
| XGBoost (Word2Vec) | 82.3 ± 1.3 % | 87.8 ± 0.9 % | 0.949 ± 0.005 | 0.547 ± 0.025 |

Against three established tools, run over the same sequences:

| Method | Recall | Specificity |
|---|---|---|
| geNomad | 89.0 % | 97.3 % |
| DeepVirFinder | 85.4 % | 79.6 % |
| ExtraTrees (proposed) | 84.6 % | 88.7 % |
| XGBoost (proposed) | 82.3 % | 87.8 % |
| VirSorter2 | 81.7 % | 97.2 % |

The five methods sit in a narrow recall band. Paired McNemar tests between each
of the 20 runs of each proposed model and each external tool found no
significant difference in any of the 6 pairs (0 of 20 runs below α = 0.05,
median p between 0.24 and 1.00). The proposed classifiers trail the annotation
tools in specificity but run on CPU in about 15 seconds for the full set of
7,624 sequences per model, against 3 to 49 minutes for the reference tools, with
no marker-gene database required. Versions, exact command lines, decision
criteria and timings for the three tools are in
[`external_tools/README.md`](external_tools/README.md).

Sensitivity is uneven across families: *Phasmaviridae*, *Rhabdoviridae*,
*Phenuiviridae*, *Xinmoviridae* and *Peribunyaviridae* are recovered almost
completely, while *Mesoniviridae* is never detected and *Togaviridae* only
sporadically. Per-family numbers are in `results/all_metrics_n20.json`.

## Repository layout

```
.
├── data/
│   ├── accessions.csv                     # the 82 GenBank accessions (Table 1 of Fuques et al. 2026)
│   ├── amazon_viruses.fasta               # the 82 viral sequences
│   ├── amazon_viruses_metadata.csv        # accession, species, family, novel status, length
│   ├── amazon_negatives_v2.fasta          # the 7,542 host transcripts
│   └── amazon_negatives_v2_metadata.csv   # NCBI accession and component of each negative
├── scripts/                               # numbered, see "Reproducing the paper"
├── results/                               # every number reported in the paper
│   └── leakage_check/                     # MMseqs2 search of the 82 viruses against the training set
└── external_tools/
    ├── README.md                          # tool versions, command lines, decision criteria, timings
    ├── amazon82/                          # raw geNomad, VirSorter2 and DeepVirFinder output on the 82 viruses
    ├── negatives/                         # the same three tools on the 7,542 host transcripts
    ├── logs/                              # the two run logs, verbatim, plus the scripts that produced them
    └── environments/                      # conda env export for the three tool environments
```

## Setup

```bash
pip install -r requirements.txt
```

Python 3.13 with scikit-learn 1.7.2, XGBoost 3.1.3 and gensim 4.4.0 were used
for the reported runs. Everything runs on CPU; no GPU is needed.

The scripts that retrain the classifiers need the training corpus, which lives
in the pipeline repository. Clone it as a sibling directory:

```bash
git clone https://github.com/carloscalage/Viral-Sequence-Classification.git
```

If you keep it elsewhere, point `VSC_ROOT` at it:

```bash
export VSC_ROOT=/path/to/Viral-Sequence-Classification
```

`scripts/10_leakage_check.sh` additionally requires [MMseqs2](https://github.com/soedinglab/MMseqs2)
on the `PATH`.

## Data

The positive class is the 82 viral sequences of Table 1 of Fuques et al. 2026,
downloaded from GenBank by accession. Genome-segmented families deposit
segments as separate records, so a raw download returns 90 entries; keeping
only the L segment of each virus leaves the 82 used here.

The negative class is 7,542 distinct host transcripts downloaded from NCBI:
mRNA of *Anopheles*, *Culex* and *Aedes*, *Wolbachia* genes and Culicidae rRNA,
each transcript used exactly once, with no fragmentation or overlap. Transcripts
match the molecule type of the RNA viruses of the reference study. Only
sequences outside the 500 to 25,000 bp range were filtered out.

Both FASTA files are committed here so the evaluation set is fixed. They can
also be rebuilt with `scripts/01_download_genbank.py` and
`scripts/06_download_negatives_v2.py`, but note that script 06 queries NCBI by
organism rather than by accession, so a fresh download will not return exactly
the same set as the database grows. The authoritative accession lists are in
`data/amazon_viruses_metadata.csv` and `data/amazon_negatives_v2_metadata.csv`.

## Reproducing the paper

Scripts are anchored to the repository root and can be run from anywhere.

| Script | What it does | Output | Feeds |
|---|---|---|---|
| `01_download_genbank.py` | Fetches the 82 viruses from GenBank | `data/amazon_viruses.fasta` | Section 3.1 |
| `06_download_negatives_v2.py` | Builds the host transcript pool | `data/amazon_negatives_v2.fasta` | Section 3.1 |
| `11_all_metrics_n20.py` | 20 embedding repetitions: recall, specificity, ROC-AUC, PR-AUC and per-family recall for both models | `results/all_metrics_n20.json` | **Tables 1 and 2**, and the proposed-classifier rows of Table 3 |
| `12_mcnemar_proposed_vs_external.py` | Paired McNemar, each of the 20 runs against each external tool | `results/mcnemar_proposed_vs_external.json` | Section 4.3 |
| `07_statistical_tests.py` | Wilson confidence intervals and McNemar among the external tools | `results/statistical_tests.txt` | Section 4.3 |
| `10_leakage_check.sh` | MMseqs2 search of the 82 viruses against the ZOVER training set | `results/leakage_check/` | Section 3.1 |
| `05b_embedding_variance_v2.py` | Embedding stochasticity, XGBoost with Word2Vec | `results/embedding_variance_v2.csv` | Section 5 |
| `05c_embedding_variance_extratrees.py` | Embedding stochasticity, ExtraTrees with fastText | `results/embedding_variance_extratrees.csv` | Section 5 |
| `13_inference_timing.py` | Inference wall-clock time over the 7,624 sequences, per model | `results/inference_timing.json` | Sections 4.1 and 5 |
| `08_roc_curve.py` | Illustrative ROC and precision-recall curve, single embedding | `results/roc_pr_curve_illustrative.png` | not in the paper |
| `02_train_classifiers.py` | Retrains the two classifiers on the 80 % ZOVER pool | `models/*.joblib` | single-run path |
| `03_predict_amazon.py` | Single-model inference over the 82 viruses | `results/amazon_predictions.csv` | single-run path |

The quickest full check needs no retraining and takes seconds:

```bash
python3 scripts/07_statistical_tests.py
```

It reads the committed raw tool outputs and reprints the external-tool recall,
specificity and confidence intervals exactly as reported in the paper.

Rebuilding Tables 1 to 3 from scratch retrains 20 embeddings per model. Each
repetition trains both a Word2Vec and a fastText embedding on the 80 % pool and
fits both classifiers, which costs about 80 s, so each script takes roughly half
an hour and the two together about an hour on CPU:

```bash
python3 scripts/11_all_metrics_n20.py
python3 scripts/12_mcnemar_proposed_vs_external.py
```

Scripts `02` and `03` reproduce the single-model path that produced
`results/amazon_predictions.csv`. They depend on an embedding trained on the
80 % split in the earlier work, which is not distributed here; point `EMB_DIR`
at it if you have it. They are not the source of any number in the paper.

## On embedding stochasticity

This matters before you compare any two runs. `gensim` trains Word2Vec and
fastText non-deterministically when `workers > 1`, even with a fixed seed, so
each training run yields a slightly different vector space. Out-of-distribution
recall varies by a few percentage points across runs while in-distribution
performance stays essentially fixed, which is one of the findings of the paper.

Two consequences:

- Every headline number is a mean over 20 independent embedding repetitions,
  never a single run. A single run reproduces the mean only within roughly one
  standard deviation.
- A classifier must always be paired with the embedding from its own training
  run. Pairing a classifier with an embedding from a different run silently
  destroys recall, without raising any error.

The `embedding_variance_*.csv` files come from a separate set of 20 repetitions
run before the consolidated one in `all_metrics_n20.json`. Their means differ by
under one percentage point from the published tables, which is itself the
variance being reported.

## Not included here

- **Training data (ZOVER)**: the viral and non-viral training FASTAs live in the
  [pipeline repository](https://github.com/carloscalage/Viral-Sequence-Classification)
  and are subject to the ZOVER terms of use.
- **Trained models and embeddings**: about 800 MB. Not needed to reproduce the
  paper, since scripts `11` and `12` retrain from the training corpus.
- **External tool databases**: geNomad, VirSorter2 and DeepVirFinder were run
  with their authors' recommended settings and their own reference databases.
  Their raw outputs are committed under `external_tools/`.

## Citation

```bibtex
@inproceedings{muniz2026alignment,
  title     = {Alignment-Free Viral Metagenomics: External Validation of K-mer
               Based Ensemble Classifiers on an Independent Amazonian Dataset},
  author    = {Muniz, Hiago D. and de Almeida, Jo{\~a}o Paulo P. and
               Silveira J{\'u}nior, Carlos Augusto Calage and
               Corr{\^e}a, Ulisses B.},
  booktitle = {Anais do Encontro Nacional de Intelig{\^e}ncia Artificial e
               Computacional (ENIAC)},
  year      = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE). Sequence data redistributed here comes from GenBank
and NCBI and is in the public domain.
