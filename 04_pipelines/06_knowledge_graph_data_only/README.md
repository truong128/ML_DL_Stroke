# Pipeline 6 — Knowledge graph (processed data only)

**Purpose:** Build a knowledge graph **only** from harmonized BRFSS **train-split CSVs**
(`02_datasets/processed/*_train.csv`). Use this to compare with pipeline 5
(`05_knowledge_graph`), which integrates saved ML/GLM outputs.

## What is used

| Source | Role |
|--------|------|
| `brfss_stroke_{ny,all_states}_train.csv` | Sole input file per cohort |

All predictors = every column except metadata (`survey_year`, `state_fips`, `split`,
`stroke`, weights, `strata`, `psu`, `seqno`).

**Preprocessing:** median imputation on the train split only (computed inside pipeline 6;
not read from any ML pipeline artifact).

## Edge layers (computed in-memory)

1. **Marginal Pearson r** — feature vs Stroke  
2. **Univariate logistic** — `scd ~ feature` on train rows  
3. **Partial r → Stroke** — feature vs Stroke controlling other predictors  
4. **Partial r feature–feature** — pairwise, controlling others + Stroke  

## What is **not** used

- LASSO feature lists (`02_lasso_selected_features.txt`)
- Post-hoc GLM CSVs (`posthoc/glm_*`)
- Permutation importance (`09_permutation_importance.csv`)
- `domain_ontology.json`
- Any file under `05_results/01`–`05` or `posthoc/`

## Run

```bash
pip install -r 04_pipelines/05_knowledge_graph/requirements-kg.txt

python 04_pipelines/06_knowledge_graph_data_only/run_kg_data_pipeline.py --dataset both
python 04_pipelines/06_knowledge_graph_data_only/run_kg_data_pipeline.py --dataset ny --quick
```

## Outputs

`05_results/06_knowledge_graph_data_only/{ny,all_states}/`

Same file naming as pipeline 5 (`01_kg_feature_pathway_scores.csv`, …).

## Compare pipeline 5 vs 6

```bash
python scripts/compare_kg_pipeline5_vs_6.py --dataset both
```

Writes `05_results/posthoc/kg_pipeline5_vs_6_comparison_{cohort}.csv` and a summary printout
(rank correlation, top-10 overlap).
