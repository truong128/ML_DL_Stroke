# Pipeline 7 FINAL — best tabular deep learning (GPU)

This DL pipeline for this BRFSS stroke project.

## What makes it “best” for *this* dataset

| Choice | Why |
|--------|-----|
| **Categorical embeddings** | Most BRFSS vars are ordinal/binary codes |
| **Encode cats on RAW values**, batch-correct **continuous only** | Avoids destroying category codes |
| **LASSO features by default** | Empirically best test AUC on NY vs all-features |
| **FT-Transformer + ResNet + RealMLP** | Strong tabular NN family; soft-ensemble top-2 |
| **Isotonic calibration + GPU** | Stable probs; CUDA by default |

## NY final results (GPU, RTX 5090)

| Setting | Val AUC | Test AUC | Winner |
|---------|---------|----------|--------|
| **LASSO (default / main)** | **0.775** | **0.740** | FT ensemble |
| ALL 57 features | 0.776 | 0.736 | FT ensemble |

Classical P1–P4 NY band: ~0.73–0.74 — DL matches the top of that band.

## Models

- `ft_transformer` / `ft_transformer_large` (primary)
- `tabular_resnet`
- `realmlp`

## Setup

```bash
source ~/.venvs/15_7_stroke/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

## Run (from `project/`)

```bash
cd /path/to/project
source ~/.venvs/15_7_stroke/bin/activate
export PROJECT_ROOT="$(pwd)"

# FINAL recommended (both NY + national, GPU if available)
python deep_learning_pipeline/run_ml_pipeline.py --n-jobs 4
# same as:
python deep_learning_pipeline/run_ml_pipeline.py --dataset both --n-jobs 4

# Single cohort only
python deep_learning_pipeline/run_ml_pipeline.py --dataset ny --n-jobs 4
python deep_learning_pipeline/run_ml_pipeline.py --dataset all_states --n-jobs 4

# Optional: all features / force CPU / smoke
python deep_learning_pipeline/run_ml_pipeline.py --dataset both --all-features
python deep_learning_pipeline/run_ml_pipeline.py --dataset both --cpu
python deep_learning_pipeline/run_ml_pipeline.py --dataset both --quick
```

GPU is **preferred by default**, not required: if CUDA is missing it falls back to CPU. Use `--cpu` to force CPU.
## Outputs

`05_results/07_python_ml_deep_learning/<dataset>/`

Also archived:
- `final_lasso_features/` — recommended winner
- `final_all_features/` — all-feature ablation
