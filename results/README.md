# What each result file is

| File | Produced by | What it holds |
|---|---|---|
| `all_metrics_n20.json` | `11_all_metrics_n20.py` | **Source of Tables 1 and 2, and of the two proposed-classifier rows of Table 3.** Recall, specificity, ROC-AUC, PR-AUC and per-family recall for both models, as mean and standard deviation over the same 20 embedding repetitions. The three reference-tool rows of Table 3 come from `../external_tools/` instead, counted from the raw tool outputs |
| `mcnemar_proposed_vs_external.json` | `12_mcnemar_proposed_vs_external.py` | Paired McNemar of each of the 20 runs of each model against each external tool, over the 82 viral sequences |
| `statistical_tests.txt` | `07_statistical_tests.py` | Wilson 95 % confidence intervals for recall and specificity of the external tools, and McNemar among them |
| `external_tools_recall_by_family.csv` | assembled from `external_tools/` | Supplementary: sequences detected per viral family by geNomad, VirSorter2 and DeepVirFinder. Not a table in the paper, but it is the evidence behind the argument in Section 5 that the families the proposed models miss, *Mesoniviridae* above all, are recovered by the homology-based tools, which points at training set coverage rather than at the method. Per-family recall of the proposed models is in `all_metrics_n20.json`, over 20 runs |
| `embedding_variance_v2.csv` | `05b_embedding_variance_v2.py` | 20 repetitions for XGBoost with Word2Vec: in-distribution F1, out-of-distribution recall and specificity, one row per embedding |
| `embedding_variance_extratrees.csv` | `05c_embedding_variance_extratrees.py` | The same for ExtraTrees with fastText |
| `leakage_check/` | `10_leakage_check.sh` | MMseqs2 nucleotide search of the 82 Amazonian viruses against the ZOVER training set |
| `amazon_predictions.csv`, `amazon_summary.csv` | `03_predict_amazon.py` | Per-sequence probabilities and predictions of the single-run inference path |
| `inference_timing.json` | `13_inference_timing.py` | Wall-clock inference time over the 7,624 sequences, per model, with model loading timed apart |

## Why some numbers differ slightly between files

The `embedding_variance_*.csv` files come from an earlier set of 20 embedding
repetitions. `all_metrics_n20.json`
comes from a later set in which both models and all metrics were computed over
the very same 20 runs, for internal consistency, and that is the set reported in
the paper.

Out-of-distribution recall differs between the two sets by under one percentage
point, 81.5 % against 82.3 % for XGBoost and 85.5 % against 84.6 % for
ExtraTrees, well inside the reported standard deviations. That spread is the
embedding stochasticity the paper describes, not a discrepancy between methods.

## Leakage check

`leakage_check/summary.txt` records it in full. Of the 82 Amazonian viruses, 27 produce some alignment against the ZOVER training set, 4 reach at
least 30 % query coverage, and the strongest of those is 72.0 % identity over up
to 86 % of the query length. The highest identity of any hit, at minimal
coverage, is 89.1 %. No sequence approaches near-duplicate status, so the
out-of-distribution evaluation is not contaminated by the training set.
`leave_out_homologs.txt` records the sanity check that removes even those
homologs and recomputes recall.
