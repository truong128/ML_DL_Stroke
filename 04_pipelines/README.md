# ML pipelines (six variants)

| Subfolder | Language | Final model | Results path |
|-----------|----------|-------------|--------------|
| `01_python_standard/` | Python | Best single classifier (val ROC-AUC) | `05_results/01_python_ml_standard/` |
| `02_python_voting_soft/` | Python | Soft voting, top 3 on validation | `05_results/02_python_ml_voting_soft/` |
| `03_r_tidymodels_voting_soft/` | R | Soft voting (benchmark) | `05_results/03_r_tidymodels_voting_soft/` |
| `04_r_tidymodels_single/` | R | Best single classifier | `05_results/04_r_tidymodels_single/` |
| `05_knowledge_graph/` | Python | ML-integrated KG pathways | `05_results/05_knowledge_graph/` |
| `06_knowledge_graph_data_only/` | Python | Train-data-only KG | `05_results/06_knowledge_graph_data_only/` |
| `shared_scripts/` | Python | Utilities, post-hoc, orchestration | copied to `project/scripts/` |

R pipelines **share** R code in `03_r_tidymodels_voting_soft/R/`; `04_r_tidymodels_single/` only differs by entry script (`variant = "single"`).

## Run from `project/` (after bootstrap)

```bash
export PROJECT_ROOT="$(pwd)"

# 1 — Python standard
python scripts/run_ml_pipeline.py --dataset ny --n-jobs -1

# 2 — Python voting-soft
python voting_soft_pipeline/run_ml_pipeline.py --dataset ny --n-jobs -1

# 3 — R voting-soft
Rscript tidymodels_pipeline/scripts/run_ml_pipeline.R --dataset ny --n-jobs 4 --cv-jobs 1

# 4 — R single best
Rscript tidymodels_single_pipeline/scripts/run_ml_pipeline.R --dataset ny --n-jobs 4 --cv-jobs 1

# 5 — ML-integrated knowledge graph (from package root)
python 04_pipelines/05_knowledge_graph/run_kg_pipeline.py --dataset both

# 6 — Data-only knowledge graph
python 04_pipelines/06_knowledge_graph_data_only/run_kg_data_pipeline.py --dataset both
```

All pipelines evaluate **16 classifiers** with the same temporal split and leakage rules.
