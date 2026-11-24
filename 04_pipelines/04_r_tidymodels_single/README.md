# tidymodels — best single classifier

Same protocol as `tidymodels_pipeline/` (LASSO, 16 models, year-grouped CV, SMOTE in folds only, isotonic calibration on validation), but the **final** model is the **best single** classifier by validation ROC-AUC — parallel to Python `scripts/run_ml_pipeline.py` → `results/ml/`.

| Pipeline | Script | Outputs |
|----------|--------|---------|
| R single | `tidymodels_single_pipeline/scripts/run_ml_pipeline.R` | `results/tidymodels_single/{ny,all_states}/` |
| R voting_soft | `tidymodels_pipeline/scripts/run_ml_pipeline.R` | `results/tidymodels/` |

## Run

```bash
cd /path/to/15_7_stroke
export PROJECT_ROOT="$(pwd)"
Rscript tidymodels_pipeline/setup.R   # once
Rscript tidymodels_single_pipeline/scripts/run_ml_pipeline.R --dataset ny
Rscript tidymodels_single_pipeline/scripts/run_ml_pipeline.R --dataset both
Rscript tidymodels_single_pipeline/scripts/run_ml_pipeline.R --dataset ny --quick
```

Compare all pipelines after runs finish:

```bash
python scripts/compare_pipelines.py --dataset both \
  --pipelines ml,ml_voting_soft,tidymodels,tidymodels_single
```
