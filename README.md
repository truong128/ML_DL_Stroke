# Stroke Prediction from BRFSS — Multi-Pipeline Machine Learning Benchmark

Reproducible code for predicting self-reported **stroke** from the CDC Behavioral Risk
Factor Surveillance System (BRFSS), 2015–2024, in two cohorts:

- **National** (`all_states`) — pooled U.S. states
- **New York** (`ny`)

The same protocol is implemented in **Python (scikit-learn)**, **R (tidymodels)** and
**PyTorch**, then complemented by interpretable GLMs and knowledge-graph pathway analysis.

## Study design

| Item | Setting |
|------|---------|
| Temporal split | Train 2015–2019 · Validation 2020–2022 · Test 2023–2024 (locked) |
| Batch effects | Survey-year correction on training data only |
| Feature selection | LASSO with nested, year-grouped cross-validation (stability reported) |
| Class imbalance | SMOTE inside CV folds only (no leakage) |
| Models | 16 classifiers (LR, LDA/QDA, NB, KNN, SVM, trees, RF, Extra Trees, Bagging, AdaBoost, Gradient Boosting, XGBoost, MLP, …) |
| Calibration | Isotonic regression on the validation split |
| Evaluation | ROC-AUC, PR-AUC, Brier, balanced accuracy, F1 with bootstrap 95% CIs on the test split |
| Interpretability | Permutation importance, SHAP, survey-weighted GLM with FDR correction, knowledge graphs |

## Pipelines

| # | Folder | Language | Final model |
|---|--------|----------|-------------|
| 1 | `04_pipelines/01_python_standard/` | Python | Best single classifier (validation ROC-AUC) |
| 2 | `04_pipelines/02_python_voting_soft/` | Python | Soft-voting ensemble of top-3 |
| 3 | `04_pipelines/03_r_tidymodels_voting_soft/` | R | Soft-voting ensemble of top-3 |
| 4 | `04_pipelines/04_r_tidymodels_single/` | R | Best single classifier |
| 5 | `04_pipelines/05_knowledge_graph/` | Python | ML-integrated knowledge graph (GLM + ML + ontology) |
| 6 | `04_pipelines/06_knowledge_graph_data_only/` | Python | Knowledge graph from training data only |
| 7 | `04_pipelines/07_python_deep_learning/` | PyTorch | FT-Transformer / ResNet / RealMLP, top-2 ensemble |

Supporting code:

- `04_pipelines/shared_scripts/` — orchestration (`run_all_pipelines.sh`, …), pipeline comparison, scoring, validation
- `06_posthoc/` — interpretable GLMs (Python & R `survey`), FDR, SHAP export, manuscript figures/tables

### Headline test-set ROC-AUC (95% CI)

| Pipeline | National | New York |
|----------|----------|----------|
| Python single model | 0.750 (0.748–0.753) | 0.734 (0.713–0.756) |
| Python soft voting | 0.749 (0.746–0.752) | 0.737 (0.716–0.757) |
| R tidymodels single | 0.750 (0.747–0.753) | 0.732 (0.709–0.754) |
| R tidymodels soft voting | 0.751 (0.748–0.754) | 0.737 (0.715–0.758) |
| Deep learning (FT-Transformer top-2) | 0.755 (0.752–0.757) | 0.740 (0.719–0.761) |

## Repository layout

```
.
├── 01_installation/       # venv setup + Python requirements
├── 02_datasets/           # (empty) place processed BRFSS CSVs here — see 02_datasets/README.md
├── 04_pipelines/          # seven modelling pipelines + shared scripts
├── 06_posthoc/            # GLMs, FDR, SHAP export, figures & tables
└── 05_results/            # (generated, not tracked) pipeline outputs
```

## Installation

### Python (≥ 3.10)

```bash
bash 01_installation/setup.sh               # creates ~/.venvs/15_7_stroke and installs requirements
source ~/.venvs/15_7_stroke/bin/activate
pip install -r 01_installation/requirements-optional.txt   # knowledge graph, docx tables
pip install torch --index-url https://download.pytorch.org/whl/cu128   # pipeline 7 (optional)
```

### R (≥ 4.3)

```bash
Rscript 04_pipelines/03_r_tidymodels_voting_soft/setup.R
```

## Data

Data are **not** included. Download BRFSS annual files from the
[CDC BRFSS website](https://www.cdc.gov/brfss/annual_data/annual_data.htm), harmonize them,
and place the train/val/test CSVs in `02_datasets/processed/` as described in
[02_datasets/README.md](02_datasets/README.md).

## Usage

The pipelines run from a working `project/` folder built by `bootstrap_project.sh`
(it copies the code and `02_datasets/processed/*.csv` into the expected layout; `project/` is gitignored).

```bash
bash bootstrap_project.sh
cd project
export PROJECT_ROOT="$(pwd)"

# 1 — Python best single model
python scripts/run_ml_pipeline.py --dataset both --n-jobs -1
# 2 — Python soft voting
python voting_soft_pipeline/run_ml_pipeline.py --dataset both --n-jobs -1
# 3 — R soft voting / 4 — R single model
Rscript tidymodels_pipeline/scripts/run_ml_pipeline.R --dataset both --n-jobs 4 --cv-jobs 1
Rscript tidymodels_single_pipeline/scripts/run_ml_pipeline.R --dataset both --n-jobs 4 --cv-jobs 1
# 7 — deep learning (GPU if available, CPU fallback)
python deep_learning_pipeline/run_ml_pipeline.py --dataset both --n-jobs 4

# Post-hoc: pipeline comparison, GLMs, importance, row-level scoring
bash scripts/run_posthoc.sh

# Quick smoke test
python scripts/run_ml_pipeline.py --dataset ny --quick
```

Knowledge graphs and manuscript figures run from the package root and read/write `05_results/`:

```bash
python 04_pipelines/05_knowledge_graph/run_kg_pipeline.py --dataset both
python 04_pipelines/06_knowledge_graph_data_only/run_kg_data_pipeline.py --dataset both
python 06_posthoc/make_manuscript_figures.py
```

Each pipeline folder has its own `README.md` with details and outputs.
`make_manuscript_figures.py` requires the Times New Roman font; place the `.ttf` files in
`06_posthoc/fonts/` (not distributed here for licensing reasons) or install them system-wide.

## Notes

- Knowledge-graph pathway scores describe statistical/structural proximity to stroke and
  do **not** imply causal mechanisms.
- BRFSS stroke status is self-reported.

## Contributors

- **Truong Nguyen**

## License

Code released under the [MIT License](LICENSE).
