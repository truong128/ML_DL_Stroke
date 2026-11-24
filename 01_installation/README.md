# Installation

## Python environment

```bash
cd /path/to/repo
bash bootstrap_project.sh
cd project
bash ../01_installation/setup.sh
source ~/.venvs/15_7_stroke/bin/activate
pip install -r ../01_installation/requirements-ml.txt
```

`setup.sh` creates a venv at `$HOME/.venvs/15_7_stroke` (recommended on exFAT/USB drives where symlinks fail).

Override location:

```bash
VENV=/path/to/my/venv bash ../01_installation/setup.sh
```

## R packages

From package root:

```bash
Rscript 04_pipelines/03_r_tidymodels_voting_soft/setup.R
```

Installs tidymodels, glmnet, ranger, xgboost, adabag, themis, survey, broom, and related engines.

## Files

| File | Role |
|------|------|
| `setup.sh` | Create venv + install `requirements.txt` |
| `requirements.txt` | Base deps (includes ML via `-r requirements-ml.txt`) |
| `requirements-ml.txt` | pandas, scikit-learn, imbalanced-learn, shap, statsmodels, xgboost |
