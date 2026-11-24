# Voting-soft ML pipeline (separate outputs)

Same methodology as `scripts/run_ml_pipeline.py`, but the **deployed model** is a  
**soft-voting ensemble** of the top three algorithms by **validation ROC-AUC** (not train CV).

**All 16 base classifiers** are trained and compared (see `03_cv_model_comparison.csv` and  
`03b_val_model_selection.csv`). Only the top 3 on validation enter `voting_soft`.

## Output location (does not overwrite `results/ml/`)

```
results/ml_voting_soft/ny/
results/ml_voting_soft/all_states/
```

## Run

```bash
cd /path/to/project
source ~/.venvs/15_7_stroke/bin/activate

# NY only
python voting_soft_pipeline/run_ml_pipeline.py --dataset ny --n-jobs -1

# Both cohorts
python voting_soft_pipeline/run_ml_pipeline.py --dataset both --n-jobs -1

# Quick test
python voting_soft_pipeline/run_ml_pipeline.py --dataset ny --quick --n-jobs -1
```

## Extra outputs vs standard pipeline

- `04b_voting_soft_members.csv` — three base learners in the ensemble
- `04c_val_final_model_comparison.csv` — best single model vs voting_soft on validation
- `pipeline_config.json` — `"final_model": "voting_soft"`, `"voting_members": [...]`
