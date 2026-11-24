#!/usr/bin/env python3
"""
make_manuscript_figures.py  (version 5.2)
=========================================

Builds the manuscript figures and tables for the BRFSS stroke-prediction study
from the saved outputs of:

  01_python_ml_standard        (Python single model)
  02_python_ml_voting_soft     (Python soft-voting top-3)
  03_r_tidymodels_voting_soft  (R tidymodels soft-voting top-3)
  04_r_tidymodels_single       (R tidymodels single model)
  07_python_ml_deep_learning   (PyTorch FT-Transformer family; top-2 ensemble)
  05_knowledge_graph           (ML-integrated knowledge graph)
  06_knowledge_graph_data_only (data-only knowledge graph)
  posthoc/                     (interpretable GLMs, row-level scores)

and the processed BRFSS files in 02_datasets/processed.

Figure numbering
----------------
  Figure 2-9                  National cohort (main manuscript)
  Supplementary Figure S2-S9  New York versions of Figures 2-9 (same titles, "for New York")
  Supplementary Figure S1     reserved for the New York version of Figure 1 (workflow, not generated)
  Supplementary Figure S10    Stroke prevalence and sample size by survey year (both cohorts)
  Supplementary Figure S11    Stroke prevalence by age (65+ vs 18-64)
  Supplementary Figure S12    Stroke prevalence by race and ethnicity
  Supplementary Figure S13    Stroke prevalence by sex
  Supplementary Figure S14/15 ML-integrated knowledge-graph associations with stroke (National / New York)
The full list with titles is written to manuscript_figures/figure_list.md.

Primary reporting models (per cohort)
-------------------------------------
Chosen by validation ROC-AUC as saved by the pipelines. Figures never mix Python and R:
  * lead single-model pipeline: Python single-model vs R tidymodels single-model
  * soft-voting ensemble of the same framework as the lead single-model pipeline
  * top neural network: FT-Transformer vs FT-Transformer-large vs deep learning top-2 ensemble
Pipeline codes (P1-P7) are internal to this script and are never printed.

Design rules
------------
* Values are READ from the pipeline outputs. Nothing is re-estimated, except the
  items approved by the study team: unweighted prevalence, Wilson 95% CIs and
  Pearson chi-square tests (S11-S13), and ROC coordinates, confusion-matrix counts,
  calibration bins, score densities and group means drawn from the saved row-level
  scores (posthoc/scored_*.csv). AUC and Brier values printed on plots are read
  from 05_test_metrics_bootstrap.csv.
* Metrics whose point estimate is blank/NA in a pipeline output are left out.
* When a pipeline did not save an output, only the pipelines that saved it are shown;
  README_provenance.md lists what was skipped.
* Ensemble SHAP panels name only the members whose SHAP values were averaged, as recorded
  in the pipeline's 12_shap_meta.json; members without saved SHAP values are listed in the
  figure note (version 5.2).
* Every text element uses Times New Roman. The script stops if the font is not found.
  It looks in 06_posthoc/fonts/ first, then the usual macOS/Windows font folders.
* No em dashes are written to any output.

Usage
-----
  python make_manuscript_figures.py
  python make_manuscript_figures.py --results-dir ~/Desktop/05_results
  python make_manuscript_figures.py --only fig4,table4

Defaults assume this file lives in 0_4_ML_Stroke_Project/06_posthoc/:
  results : 0_4_ML_Stroke_Project/05_results
  data    : 0_4_ML_Stroke_Project/02_datasets/processed
  output  : 0_4_ML_Stroke_Project/05_results/manuscript_figures  (replaced on each run)

Codebook: CDC BRFSS 2024 LLCP codebook (USCODE24_LLCP_082125.HTML).
  sex        = SEXVAR    1 Male, 2 Female
  age_65plus = _AGE65YR  1 Age 18-64, 2 Age 65+, 3 Don't know/Refused/Missing
  race_group = _RACEG21  1 Non-Hispanic White, 2 Non-White or Hispanic
  hispanic   = _HISPANC  1 Hispanic/Latino, 2 Not Hispanic/Latino

Requirements: python>=3.9, numpy, pandas, matplotlib; python-docx for .docx tables.
"""
from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import re
import shutil
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
logging.getLogger("fontTools").setLevel(logging.WARNING)
log = logging.getLogger("manuscript")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS = PROJECT_ROOT / "05_results"
DEFAULT_DATA = PROJECT_ROOT / "02_datasets" / "processed"
DEFAULT_OUT = PROJECT_ROOT / "05_results" / "manuscript_figures"

FONT_DIRS = [SCRIPT_DIR / "fonts", PROJECT_ROOT / "fonts",
             Path("/System/Library/Fonts/Supplemental"), Path("/Library/Fonts"),
             Path.home() / "Library" / "Fonts", Path("C:/Windows/Fonts"),
             Path("/usr/share/fonts/truetype/msttcorefonts")]
FONT = "Times New Roman"

# ---------------------------------------------------------------------------
# Study constants
# ---------------------------------------------------------------------------
COHORTS = [("all_states", "National"), ("ny", "New York")]
CNAME = dict(COHORTS)
SPLITS = [("train", "Train\n2015-2019"), ("val", "Validation\n2020-2022"),
          ("test", "Test\n2023-2024"), ("all", "All years\n2015-2024")]

PIPELINES = {  # internal keys only; never printed
    "P1": dict(dir="01_python_ml_standard", label="Python single-model", framework="Python",
               kind="single", scored="ml"),
    "P2": dict(dir="02_python_ml_voting_soft", label="Python soft-voting ensemble", framework="Python",
               kind="ensemble", scored="voting_soft"),
    "P3": dict(dir="03_r_tidymodels_voting_soft", label="R tidymodels soft-voting ensemble", framework="R tidymodels",
               kind="ensemble", scored=None),
    "P4": dict(dir="04_r_tidymodels_single", label="R tidymodels single-model", framework="R tidymodels",
               kind="single", scored=None),
    "P7": dict(dir="07_python_ml_deep_learning", label="Deep learning (PyTorch)", framework="Deep learning",
               kind="ensemble", scored=None),
}
SINGLE_CANDIDATES = ["P1", "P4"]
# Reporting framework: the Python pipelines supply the LASSO feature set, permutation importances and
# SHAP values used for interpretability and the knowledge graphs, so they are the reporting pipelines in
# both cohorts. Set to None to pick the framework per cohort by validation ROC-AUC instead.
REPORTING_FRAMEWORK = "Python"
# Deep learning run reported per cohort: a subfolder of 07_python_ml_deep_learning/<cohort>/ (empty = the base run).
# New York uses the LASSO-feature final run (locked test ROC-AUC 0.740). Override with --ny-dl-run.
DL_SUBRUN = {"ny": "final_lasso_features"}
# Files that may be read from the base deep learning run when the chosen subfolder did not save them.
# Every such read is flagged on the figure and in README_provenance.md.
DL_FALLBACK_OK = {"09_permutation_importance.csv"}
ENSEMBLE_OF = {"P1": "P2", "P4": "P3"}
BENCH_ORDER = ["P1", "P2", "P4", "P3", "P7"]
KG_DIRS = {"ml": "05_knowledge_graph", "data": "06_knowledge_graph_data_only"}
KIND_LABEL = {"single": "Lead single model", "ensemble": "Soft-voting ensemble", "nn": "Top neural network"}

DEMOGRAPHICS = {
    "age_65plus": ("Age (_AGE65YR)", {1: "18-64 years", 2: "≥65 years"}),
    "race_group": ("Race/ethnicity (_RACEG21)", {1: "Non-Hispanic White", 2: "Non-White or Hispanic"}),
    "hispanic": ("Hispanic ethnicity (_HISPANC)", {1: "Hispanic/Latino", 2: "Not Hispanic/Latino"}),
    "sex": ("Sex (SEXVAR)", {1: "Male", 2: "Female"}),
}

# Figure registry: key -> (label, title, file stem)
MAIN_TITLES = {
    2: "Feature selection and permutation importance (primary reporting pipelines)",
    3: "Classifier and ensemble comparison (primary reporting pipelines)",
    4: "Cross-prediction-pipeline benchmark on the locked test set",
    5: "Locked test-set discrimination: primary reporting pipeline",
    6: "Calibration and score distributions (test set)",
    7: "Model risk scores by year versus BRFSS-reported stroke",
    8: "Model interpretability using SHAP and GLM associations",
    9: "Data-only knowledge-graph associations with stroke",
}
STEMS = {2: "feature_selection_permutation_importance", 3: "classifier_ensemble_comparison",
         4: "cross_pipeline_benchmark_test", 5: "test_discrimination", 6: "calibration_score_distributions",
         7: "risk_scores_by_year", 8: "interpretability_shap_glm", 9: "kg_data_only"}
EXTRA_SUPP = {
    10: ("Stroke prevalence and sample size by survey year", "prevalence_sample_size_by_year"),
    11: ("Stroke prevalence among adults aged 65 years or older versus 18-64 years", "prevalence_age_65plus"),
    12: ("Stroke prevalence by race and ethnicity", "prevalence_race_ethnicity"),
    13: ("Stroke prevalence by sex", "prevalence_sex"),
    14: ("ML-integrated knowledge-graph associations with stroke, National", "kg_ml_integrated_national"),
    15: ("ML-integrated knowledge-graph associations with stroke, New York", "kg_ml_integrated_ny"),
}


def fig_id(n, cohort):
    """Main figure for National, supplementary mirror for New York."""
    if cohort == "all_states":
        return f"Figure {n}", f"figure{n}_national_{STEMS[n]}", f"{MAIN_TITLES[n]}"
    return f"Supplementary Figure S{n}", f"figureS{n}_ny_{STEMS[n]}", f"{MAIN_TITLES[n]} for New York"


# ---------------------------------------------------------------------------
# Visual system (single = blue, ensemble = orange)
# ---------------------------------------------------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
VIOLET, RED, GRAY = "#4a3aa7", "#e34948", "#8a8984"
LIGHT_BLUE, LIGHT_ORANGE = "#9cc0ea", "#f4b699"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PIPE_COLOR = {"P1": BLUE, "P4": BLUE, "P2": ORANGE, "P3": ORANGE, "P7": ORANGE}
ARCH_COLOR = {"ft_transformer": BLUE, "ft_transformer_large": AQUA, "realmlp": GRAY, "tabular_resnet": GRAY,
              "ensemble": ORANGE}
SPLIT_COLOR = {"train": BLUE, "val": ORANGE, "test": AQUA}
GROUP_COLORS = [BLUE, ORANGE]
FIG_W = 7.2

MODEL_LABELS = {
    "logistic_regression": "Logistic regression", "multinom_reg": "Multinomial regression",
    "svm_rbf": "SVM (RBF)", "svm_linear": "SVM (linear)", "decision_tree": "Decision tree",
    "random_forest": "Random forest", "gradient_boosting": "Gradient boosting",
    "extra_trees": "Extra trees", "adaboost": "AdaBoost", "bag_tree": "Bagged tree",
    "bagging": "Bagging", "knn": "k-nearest neighbors", "lda": "LDA", "qda": "QDA",
    "naive_bayes": "Naive Bayes", "mlp": "MLP", "xgboost": "XGBoost", "lightgbm": "LightGBM",
    "ft_transformer": "FT-Transformer", "ft_transformer_large": "FT-Transformer-large",
    "tabular_resnet": "Tabular ResNet", "realmlp": "RealMLP", "mlp_small": "MLP (small)",
    "mlp_medium": "MLP (medium)", "voting_soft": "Soft-voting ensemble", "stacking": "Stacking",
    "ensemble": "Top-2 ensemble",
}

FEATURE_LABELS = {
    "sex": "Sex", "age_group": "Age group (65+ bands)", "age_65plus": "Age ≥65 years",
    "education": "Education", "income": "Household income", "marital": "Marital status",
    "race_group": "Race/ethnicity (NH White vs other)", "hispanic": "Hispanic ethnicity",
    "rent_home": "Home ownership/rental", "employed": "Employment status", "veteran": "Veteran status",
    "children": "Children in household", "general_health": "General health",
    "physical_health_days": "Poor physical health days", "mental_health_days": "Poor mental health days",
    "poor_health_days": "Activity-limiting poor health days",
    "difficulty_deciding": "Difficulty concentrating/deciding", "difficulty_walking": "Difficulty walking",
    "blind": "Blind/serious vision difficulty", "difficulty_dressing": "Difficulty dressing/bathing",
    "difficulty_alone": "Difficulty doing errands alone", "diabetes": "Diabetes",
    "prediabetes": "Prediabetes", "high_blood_pressure": "High blood pressure",
    "bp_meds": "Blood pressure medication", "cholesterol_check": "Cholesterol check",
    "heart_attack": "Heart attack (MI)", "heart_disease": "Coronary heart disease",
    "asthma_ever": "Asthma (ever)", "asthma_now": "Asthma (current)", "skin_cancer": "Skin cancer",
    "other_cancer": "Other cancer", "copd": "COPD", "arthritis": "Arthritis",
    "arthritis_disability": "Arthritis-related limitation", "depression_disorder": "Depressive disorder",
    "kidney_disease": "Kidney disease", "smoked_100": "Smoked ≥100 cigarettes",
    "smoke_days": "Smoking frequency", "smoker_status": "Smoking status", "non_smoker": "Non-smoker",
    "ecig_now": "Current e-cigarette use", "alcohol_days": "Alcohol days (past 30)",
    "avg_drinks": "Average drinks per occasion", "binge_drinking": "Binge drinking",
    "any_exercise": "Any exercise (past 30 days)", "activity_minutes": "Physical activity minutes",
    "has_coverage": "Health care coverage", "no_coverage_gap": "No coverage gap",
    "has_pcp": "Has personal doctor", "routine_checkup": "Routine checkup",
    "cost_barrier": "Cost barrier to care", "flu_shot": "Influenza vaccination",
    "pneumonia_vax": "Pneumococcal vaccination", "bmi": "Body mass index",
    "bmi_category": "BMI category", "healthy_bmi": "Healthy BMI", "stroke": "Stroke", "Stroke": "Stroke",
}

GLYPH_WARNINGS: list[str] = []


def setup_font():
    """Register Times New Roman and make it the only font matplotlib uses."""
    patt = re.compile(r"^(times new roman.*|times_new_roman.*|times|timesbd|timesi|timesbi)$", re.I)
    for d in FONT_DIRS:
        if d.is_dir():
            for f in d.iterdir():
                if f.suffix.lower() in (".ttf", ".otf") and patt.match(f.stem):
                    try:
                        font_manager.fontManager.addfont(str(f))
                    except Exception:  # noqa: BLE001
                        pass
    try:
        path = font_manager.findfont(font_manager.FontProperties(family=FONT), fallback_to_default=False)
    except ValueError:
        path = None
    if not path or font_manager.FontProperties(fname=path).get_name() != FONT:
        raise SystemExit(
            "Times New Roman was not found. Copy the font files into 06_posthoc/fonts/, e.g. on macOS:\n"
            '  mkdir -p 06_posthoc/fonts && cp "/System/Library/Fonts/Supplemental/Times New Roman"*.ttf 06_posthoc/fonts/')
    plt.rcParams.update({
        "font.family": FONT, "font.serif": [FONT], "font.sans-serif": [FONT],
        "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic",
        "mathtext.bf": f"{FONT}:bold", "mathtext.sf": FONT, "mathtext.default": "regular",
        "axes.unicode_minus": False,
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
        "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })

    def _hook(message, category, filename, lineno, file=None, line=None):
        msg = str(message)
        if "Glyph" in msg or "findfont" in msg:
            GLYPH_WARNINGS.append(msg)
        else:
            log.debug("%s", msg)
    warnings.showwarning = _hook
    warnings.simplefilter("always", UserWarning)
    log.info("font: %s", path)
    return path


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def feat(name: str) -> str:
    name = str(name).strip()
    m = re.fullmatch(r"Q\('(.+)'\)", name)
    if m:
        name = m.group(1)
    return FEATURE_LABELS.get(name, name.replace("_", " ").capitalize())


def mlabel(name: str) -> str:
    return MODEL_LABELS.get(str(name), str(name).replace("_", " ").capitalize())


def fmt(x, d=3) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "NA"
    return f"{x:.{d}f}" if np.isfinite(x) else "NA"


def fmt_ci(p, lo, hi, d=3) -> str:
    return f"{p:.{d}f} ({lo:.{d}f}–{hi:.{d}f})"


def fmt_p(p) -> str:
    if p is None or not np.isfinite(p):
        return "NA"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def wrap(text, width=60):
    return "\n".join(textwrap.fill(t, width) for t in str(text).split("\n"))


def join_and(items) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def parse_list(v):
    if isinstance(v, list):
        return v
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    try:
        out = ast.literal_eval(str(v))
        return list(out) if isinstance(out, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return [s.strip() for s in str(v).split(",") if s.strip()]


def parse_shap_members(note: str) -> list[str]:
    """Model names from a 12_shap_meta.json note such as
    'SHAP mean across voting members: adaboost(n=300; PermutationExplainer n=300); random_forest(n=1500)'."""
    if ":" not in str(note):
        return []
    body = re.sub(r"\([^)]*\)", "", str(note).split(":", 1)[1])
    return [s.strip() for s in body.split(";") if s.strip()]


def letters(fig, axes_and_letters):
    """Bold panel letters at each panel's top-left, aligned across rows and columns
    (A top-left, B top-right, C bottom-left, D bottom-right for a 2 x 2 layout)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    items = [(ax, L, ax.get_tightbbox(r).transformed(inv), ax.get_position()) for ax, L in axes_and_letters]
    top, left = {}, {}
    for key, store, pick, agg in [(lambda p: round(p.y1, 2), top, lambda b: b.y1, max),
                                  (lambda p: round(p.x0, 2), left, lambda b: b.x0, min)]:
        groups = {}
        for ax, _, bb, pos in items:
            groups.setdefault(key(pos), []).append((ax, bb))
        for members in groups.values():
            v = agg(pick(bb) for _, bb in members)
            for ax, _ in members:
                store[id(ax)] = v
    for ax, L, _, _ in items:
        fig.text(left[id(ax)], top[id(ax)], L, fontsize=12, fontweight="bold", ha="left", va="bottom", color=INK)


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
class Results:
    def __init__(self, results_dir: Path, data_dir: Path, dl_subrun: dict | None = None):
        self.root = results_dir
        self.data = data_dir
        self.missing: list[str] = []
        self.used: set[str] = set()
        self._cache: dict = {}
        # cohort -> subfolder of 07_python_ml_deep_learning/<cohort>/ holding the reported deep learning run
        self.dl_subrun = {k: v for k, v in (dl_subrun if dl_subrun is not None else DL_SUBRUN).items() if v}
        self.dl_fallbacks: set[tuple[str, str]] = set()  # (cohort, file) read from the base run instead
        # (cohort, pipeline label, SHAP members, members without SHAP) for README_provenance.md
        self.shap_partial: list[tuple[str, str, list[str], list[str]]] = []

    def path(self, pid_or_dir, cohort, fname) -> Path:
        d = PIPELINES[pid_or_dir]["dir"] if pid_or_dir in PIPELINES else pid_or_dir
        if pid_or_dir == "P7" and cohort in self.dl_subrun:
            p = self.root / d / cohort / self.dl_subrun[cohort] / fname
            if not p.exists() and fname in DL_FALLBACK_OK:
                alt = self.root / d / cohort / fname
                if alt.exists():
                    self.dl_fallbacks.add((cohort, fname))
                    return alt
            return p
        return self.root / d / cohort / fname if cohort else self.root / d / fname

    def dl_fallback(self, cohort, fname) -> bool:
        self.path("P7", cohort, fname)
        return (cohort, fname) in self.dl_fallbacks

    def csv(self, pid, cohort, fname, **kw):
        p = self.path(pid, cohort, fname)
        key = (str(p), tuple(sorted(kw.items())))
        if key in self._cache:
            return self._cache[key]
        if not p.exists():
            rel = str(p.relative_to(self.root)) if str(p).startswith(str(self.root)) else str(p)
            if rel not in self.missing:
                self.missing.append(rel)
            log.warning("missing: %s", p)
            self._cache[key] = None
            return None
        df = pd.read_csv(p, **kw)
        self.used.add(str(p))
        self._cache[key] = df
        return df

    def config(self, pid, cohort) -> dict:
        p = self.path(pid, cohort, "pipeline_config.json")
        if not p.exists():
            self.missing.append(str(p))
            return {}
        self.used.add(str(p))
        return json.loads(p.read_text())

    def boot(self, pid, cohort):
        df = self.csv(pid, cohort, "05_test_metrics_bootstrap.csv")
        if df is None:
            return None
        df = df.copy()
        for c in ["point", "ci_low", "ci_high"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.set_index("metric")

    def scored(self, pid, cohort, split):
        tag = PIPELINES[pid]["scored"]
        if tag is None:
            return None
        return self.csv("posthoc", None, f"scored_{cohort}_{split}_{tag}.csv")

    def final_model_name(self, pid, cohort) -> str:
        cfg = self.config(pid, cohort)
        if pid in ("P2", "P3"):
            mem = self.csv(pid, cohort, "04b_voting_soft_members.csv")
            members = list(mem["model"]) if mem is not None else parse_list(cfg.get("voting_members"))
            return "Soft voting: " + ", ".join(mlabel(m) for m in members)
        if pid == "P7":
            mem = (self.csv(pid, cohort, "04b_ensemble_members.csv")
                   if self.path(pid, cohort, "04b_ensemble_members.csv").exists() else None)
            if mem is not None:
                members = list(mem["model"])
            else:  # e.g. "dl_ensemble_top2(ft_transformer,ft_transformer_large)" in pipeline_config.json
                m = re.search(r"\(([^)]*)\)", str(cfg.get("final_model", "")))
                members = [s.strip() for s in m.group(1).split(",")] if m else []
            return "Top-2 ensemble: " + " + ".join(mlabel(m) for m in members)
        return mlabel(cfg.get("final_model") or cfg.get("best_model", "?"))

    def ensemble_members(self, pid, cohort) -> list[str]:
        mem = self.csv(pid, cohort, "04b_voting_soft_members.csv")
        return list(mem["model"]) if mem is not None else []

    def shap_members(self, pid, cohort) -> list[str] | None:
        """Ensemble members actually averaged into 12_shap_mean_abs.csv, read from 12_shap_meta.json.
        None if the meta file is missing or does not list members."""
        p = self.path(pid, cohort, "12_shap_meta.json")
        if not p.exists():
            return None
        self.used.add(str(p))
        members = parse_shap_members(json.loads(p.read_text()).get("note", ""))
        return members or None

    def final_val_auc(self, pid, cohort) -> float:
        cfg = self.config(pid, cohort)
        if pid == "P1":
            vs = self.csv(pid, cohort, "03b_val_model_selection.csv")
            row = vs.loc[vs["model"] == cfg.get("best_model"), "val_roc_auc"]
            return float(row.iloc[0]) if len(row) else np.nan
        if pid in ("P2", "P3"):
            return float(cfg.get("val_roc_auc_voting_soft", np.nan))
        if pid == "P4":
            return float(cfg.get("val_roc_auc_single", np.nan))
        if pid == "P7":
            return float(cfg.get("val_roc_auc", np.nan))
        return np.nan

    def nn_val(self, cohort) -> dict:
        vs = self.csv("P7", cohort, "03b_val_model_selection.csv")
        out = {} if vs is None else {m: float(v) for m, v in zip(vs.model, vs.val_roc_auc)
                                     if m in ("ft_transformer", "ft_transformer_large")}
        out["ensemble"] = self.final_val_auc("P7", cohort)
        return out

    def primaries(self, cohort) -> dict:
        best = lambda cands, f: max(cands, key=lambda k: f(k) if np.isfinite(f(k)) else -np.inf)  # noqa: E731
        cands = [p for p in SINGLE_CANDIDATES
                 if REPORTING_FRAMEWORK is None or PIPELINES[p]["framework"] == REPORTING_FRAMEWORK]
        single = best(cands or SINGLE_CANDIDATES, lambda p: self.final_val_auc(p, cohort))
        nnv = self.nn_val(cohort)
        nn = best(list(nnv), lambda k: nnv[k])
        return {"single": single, "ensemble": ENSEMBLE_OF[single], "nn": nn}

    def prim_label(self, cohort, prim, kind) -> str:
        if kind == "nn":
            return "Deep learning (PyTorch): " + (self.final_model_name("P7", cohort) if prim["nn"] == "ensemble"
                                                 else mlabel(prim["nn"]))
        pid = prim[kind]
        return f"{PIPELINES[pid]['label']} ({self.final_model_name(pid, cohort)})"

    def single_model(self, pid, cohort) -> str:
        cfg = self.config(pid, cohort)
        return cfg.get("final_model") or cfg.get("best_model")

    def selected_features(self, pid, cohort):
        return parse_list(self.config(pid, cohort).get("selected_features"))

    def demographics(self, cohort) -> pd.DataFrame:
        key = ("demo", cohort)
        if key not in self._cache:
            name = "brfss_stroke_new_york_2015_2024.csv" if cohort == "ny" else "brfss_stroke_all_states_2015_2024.csv"
            p = self.data / name
            log.info("reading %s", p)
            self._cache[key] = pd.read_csv(p, usecols=["survey_year", "split", "stroke"] + list(DEMOGRAPHICS))
            self.used.add(str(p))
        return self._cache[key]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
class Writer:
    def __init__(self, out: Path):
        self.out = out
        self.fig_dir = out / "figures"
        self.tab_dir = out / "tables"
        self.tables = []
        self.figs: list[tuple[str, str, str, str]] = []  # (label, title, stem, note)
        self.notes: list[str] = []

    def prepare(self, clean: bool):
        if clean and self.out.exists():
            try:
                shutil.rmtree(self.out)
            except OSError as e:
                log.warning("could not clear %s (%s); files will be overwritten", self.out, e)
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.tab_dir.mkdir(parents=True, exist_ok=True)

    def fig(self, fig, label, title, stem, note=""):
        for ext in ("png", "pdf"):
            fig.savefig(self.fig_dir / f"{stem}.{ext}")
        plt.close(fig)
        self.figs.append((label, title, stem, note))
        log.info("%s: %s", label, stem)

    def table(self, key, title, parts, footnote=""):
        frames = []
        for sub, df in parts:
            d = df.copy()
            if sub:
                d.insert(0, "Panel", sub)
            frames.append(d)
        pd.concat(frames, ignore_index=True).to_csv(self.tab_dir / f"{key}.csv", index=False)
        self.tables.append((key, title, parts, footnote))
        log.info("table: %s", key)

    def write_docx(self):
        try:
            from docx import Document
            from docx.enum.section import WD_ORIENT
            from docx.oxml.ns import qn
            from docx.shared import Pt, Inches
        except ImportError:
            log.warning("python-docx not installed; .docx tables skipped (pip install python-docx)")
            self.notes.append("python-docx not installed: tables written as CSV only.")
            return
        doc = Document()
        sec = doc.sections[0]
        sec.orientation = WD_ORIENT.LANDSCAPE
        sec.page_width, sec.page_height = sec.page_height, sec.page_width
        for m in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
            setattr(sec, m, Inches(0.7))

        def set_fonts(el):
            rpr = el.get_or_add_rPr()
            rfonts = rpr.find(qn("w:rFonts"))
            if rfonts is None:
                rfonts = rpr.makeelement(qn("w:rFonts"), {})
                rpr.append(rfonts)
            for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                rfonts.set(qn(a), FONT)

        def tnr(run, size=12, bold=False, italic=False):
            run.font.name = FONT
            run.font.size = Pt(size)
            run.bold, run.italic = bold, italic
            set_fonts(run._element)

        for sname in ("Normal", "Table Grid"):
            try:
                st = doc.styles[sname]
                st.font.name = FONT
                st.font.size = Pt(12)
                set_fonts(st.element)
            except KeyError:
                pass

        for i, (key, title, parts, foot) in enumerate(self.tables):
            if i:
                doc.add_page_break()
            tnr(doc.add_paragraph().add_run(title), bold=True)
            for sub, df in parts:
                if sub:
                    tnr(doc.add_paragraph().add_run(sub), italic=True)
                t = doc.add_table(rows=1, cols=len(df.columns))
                t.style = "Table Grid"
                for j, c in enumerate(df.columns):
                    tnr(t.rows[0].cells[j].paragraphs[0].add_run(str(c)), bold=True)
                for _, r in df.iterrows():
                    cells = t.add_row().cells
                    for j, v in enumerate(r):
                        tnr(cells[j].paragraphs[0].add_run("" if pd.isna(v) else str(v)))
                doc.add_paragraph()
            if foot:
                tnr(doc.add_paragraph().add_run(foot), size=12, italic=True)
        doc.save(self.tab_dir / "manuscript_tables.docx")
        log.info("tables docx written")


# ---------------------------------------------------------------------------
# Statistics approved for S11-S13 and plot geometry
# ---------------------------------------------------------------------------
Z = 1.959963984540054


def wilson(k, n):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    den = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / den
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return p, centre - half, centre + half


def chisq_2x2(a, b, c, d):
    n = a + b + c + d
    den = (a + b) * (c + d) * (a + c) * (b + d)
    if den == 0:
        return np.nan, np.nan
    chi2 = n * (a * d - b * c) ** 2 / den
    return chi2, math.erfc(math.sqrt(chi2 / 2))


def roc_points(y, s):
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    distinct = np.r_[np.where(np.diff(s))[0], y.size - 1]
    tps = np.cumsum(y)[distinct]
    fps = (1 + distinct) - tps
    return np.r_[0, fps / max(fps[-1], 1)], np.r_[0, tps / max(tps[-1], 1)]


def calibration_bins(y, s, n_bins=10):
    edges = np.unique(np.quantile(s, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 2)
    pred, obs = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum():
            pred.append(s[m].mean()); obs.append(y[m].mean())
    return np.array(pred), np.array(obs)


def smooth_density(s, lo, hi, n=512):
    hist, edges = np.histogram(s, bins=n, range=(lo, hi), density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]
    bw = 1.06 * np.std(s) * s.size ** (-1 / 5)
    sig = max(bw / width, 1.0)
    k = np.arange(-int(4 * sig), int(4 * sig) + 1)
    kern = np.exp(-0.5 * (k / sig) ** 2)
    return centers, np.convolve(hist, kern / kern.sum(), mode="same")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def table1(R, W):
    p = R.data / "pooling_summary_by_year.csv"
    if not p.exists():
        R.missing.append(str(p)); return
    R.used.add(str(p))
    ps = pd.read_csv(p)
    nat = ps[ps.dataset == "all_states"].set_index("survey_year")
    ny = ps[ps.dataset == "new_york"].set_index("survey_year")
    split_name = {"train": "Train", "val": "Validation", "test": "Test"}
    rows = []
    for y in sorted(nat.index):
        rows.append({
            "Split": split_name.get(nat.at[y, "split"], nat.at[y, "split"]), "Survey year": int(y),
            "National n": f"{int(nat.at[y, 'n']):,}",
            "National stroke prevalence, %": f"{100 * nat.at[y, 'stroke_rate']:.1f}",
            "National states, n": int(nat.at[y, "n_states"]),
            "New York n": f"{int(ny.at[y, 'n']):,}" if y in ny.index else "NA",
            "New York stroke prevalence, %": f"{100 * ny.at[y, 'stroke_rate']:.1f}" if y in ny.index else "NA",
        })
    W.table("table1_sample_size_prevalence_by_year",
            "Table 1. Analytic sample size and unweighted stroke prevalence by survey year",
            [("", pd.DataFrame(rows))],
            "Values from 02_datasets/processed/pooling_summary_by_year.csv (unweighted). "
            "Train 2015-2019, validation 2020-2022, locked test 2023-2024.")


def _stability(R, pid, cohort, top=15):
    st = R.csv(pid, cohort, "02c_lasso_feature_stability.csv")
    if st is None:
        return None
    st = st.reset_index(drop=True).assign(_o=lambda d: np.arange(len(d)))
    return st.sort_values(["stability", "_o"], ascending=[False, True]).head(top)


def table2(R, W, prims):
    parts, notes = [], []
    letter = iter("ABCD")
    for cohort, cname in COHORTS:
        pid = prims[cohort]["single"]
        st = _stability(R, pid, cohort)
        if st is None:
            continue
        sel = set(R.selected_features(pid, cohort))
        folds = R.csv(pid, cohort, "02a_lasso_nested_cv_outer_folds.csv")
        k = len(folds) if folds is not None else None
        df = pd.DataFrame({
            "Rank": range(1, len(st) + 1),
            "Feature": [feat(f) for f in st.feature],
            f"Times selected (of {k} outer folds)" if k else "Times selected": st.times_selected.astype(int).values,
            "Selection frequency": [f"{v:.2f}" for v in st.stability],
            "Included in final model": ["Yes" if f in sel else "No" for f in st.feature],
        })
        L = next(letter)
        parts.append((f"{L}. {cname}: {PIPELINES[pid]['framework']} LASSO ({PIPELINES[pid]['label']} pipeline)", df))
        s = R.csv(pid, cohort, "02d_lasso_nested_cv_summary.csv")
        if s is not None:
            s = s.iloc[0]
            notes.append(f"{L}: nested outer-CV ROC-AUC {s.outer_roc_auc_mean:.3f} (SD {s.outer_roc_auc_std:.3f}), "
                         f"{int(s.n_stable_features)} stable features")
    W.table("table2_lasso_feature_stability",
            "Table 2. LASSO feature stability (training data, primary reporting pipeline)", parts,
            "Top 15 features by outer-fold selection frequency (ties kept in pipeline order), for the LASSO run of the "
            "lead single-model pipeline in each cohort. " + "; ".join(notes) + ". The Python LASSO is shared by the "
            "Python single-model, Python soft-voting and deep learning pipelines; each R tidymodels pipeline ran its own "
            "LASSO. Source: 02c_lasso_feature_stability.csv and 02d_lasso_nested_cv_summary.csv.")


def table3(R, W, prims):
    parts = []
    letter = iter("ABCD")
    for cohort, cname in COHORTS:
        s, e = prims[cohort]["single"], prims[cohort]["ensemble"]
        vs = R.csv(s, cohort, "03b_val_model_selection.csv")
        cv = R.csv(s, cohort, "03_cv_model_comparison.csv")
        if vs is not None and cv is not None:
            m = vs.merge(cv[["model", "roc_auc_mean", "roc_auc_std"]], on="model", how="left")
            m = m.sort_values("val_roc_auc", ascending=False).head(10)
            chosen = R.single_model(s, cohort)
            members = set(R.ensemble_members(e, cohort))
            parts.append((f"{next(letter)}. {cname}: {PIPELINES[s]['label']} pipeline ({len(vs)} classifiers)",
                          pd.DataFrame({
                              "Rank": range(1, len(m) + 1),
                              "Model": [mlabel(x) for x in m.model],
                              "Train CV ROC-AUC, mean (SD)": [f"{a:.3f} ({b:.3f})" for a, b in zip(m.roc_auc_mean, m.roc_auc_std)],
                              "Validation ROC-AUC": [f"{v:.3f}" for v in m.val_roc_auc],
                              "Selected single model": ["Yes" if x == chosen else "No" for x in m.model],
                              "Soft-voting member": ["Yes" if x in members else "No" for x in m.model],
                          })))
        nn = R.csv("P7", cohort, "03b_val_model_selection.csv")
        if nn is not None:
            nn = nn.sort_values("val_roc_auc", ascending=False)
            df = pd.DataFrame({
                "Rank": [str(i) for i in range(1, len(nn) + 1)],
                "Model": [mlabel(x) for x in nn.model],
                "Architecture": [str(x).replace(" +cat_emb", " + categorical embeddings") for x in nn.summary],
                "Best epoch": [str(int(x)) for x in nn.best_epoch],
                "Validation ROC-AUC": [f"{v:.3f}" for v in nn.val_roc_auc],
            })
            ens = pd.DataFrame([{"Rank": "NA", "Model": R.final_model_name("P7", cohort), "Architecture": "weighted soft average",
                                 "Best epoch": "NA", "Validation ROC-AUC": fmt(R.final_val_auc("P7", cohort))}])
            parts.append((f"{next(letter)}. {cname}: deep learning (PyTorch) pipeline", pd.concat([df, ens], ignore_index=True)))
    W.table("table3_top_classifiers_validation",
            "Table 3. Top ten classifiers by validation ROC-AUC (primary reporting pipeline)", parts,
            "Validation years 2020-2022. Train CV, grouped cross-validation by survey year on the training split. "
            "Soft-voting members are those of the same framework's soft-voting pipeline. Deep learning rows from "
            "07_python_ml_deep_learning/03b_val_model_selection.csv; the ensemble validation ROC-AUC is the value recorded "
            "in pipeline_config.json (after isotonic calibration on validation).")


def _prim_tag(prims, pid):
    tags = []
    for cohort, cname in COHORTS:
        p = prims[cohort]
        if pid in (p["single"], p["ensemble"]) or (pid == "P7" and p["nn"] == "ensemble"):
            tags.append(cname)
    return ", ".join(tags) if tags else "No"


def table4(R, W, prims):
    rows = []
    for pid in BENCH_ORDER:
        row = {"Pipeline": PIPELINES[pid]["label"], "Primary reporting model": _prim_tag(prims, pid)}
        for cohort, cname in COHORTS:
            b = R.boot(pid, cohort)
            row[f"{cname} final model"] = R.final_model_name(pid, cohort)
            if b is not None and "roc_auc" in b.index and np.isfinite(b.at["roc_auc", "point"]):
                r = b.loc["roc_auc"]
                row[f"{cname} ROC-AUC (95% CI)"] = fmt_ci(r.point, r.ci_low, r.ci_high)
            else:
                row[f"{cname} ROC-AUC (95% CI)"] = "NA"
        rows.append(row)
    W.table("table4_pipeline_comparison_test_roc_auc",
            "Table 4. Comparison of prediction pipelines using ROC-AUC on locked test data (2023-2024)",
            [("", pd.DataFrame(rows))],
            "Bootstrap 95% CIs from each pipeline's 05_test_metrics_bootstrap.csv. Single FT-Transformer and "
            "FT-Transformer-large locked test metrics were not saved by the deep learning pipeline (validation values in Table 3).")


def table5(R, W, prims):
    metric_rows = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("accuracy", "Accuracy"),
                   ("sensitivity", "Sensitivity"), ("specificity", "Specificity"),
                   ("f1", "F1 score"), ("brier", "Brier score")]
    cols = []
    for cohort, cname in COHORTS:
        p = prims[cohort]
        for kind in ("single", "ensemble", "nn"):
            pid = p[kind] if kind != "nn" else ("P7" if p["nn"] == "ensemble" else None)
            cols.append((cohort, cname, kind, pid))
    head = lambda cname, kind: f"{cname}: {KIND_LABEL[kind].lower()}"  # noqa: E731
    table = {"Metric": ["Pipeline and final model", "Test n", "Classification threshold (Youden J, validation)"]}
    for c, cname, kind, pid in cols:
        if pid is None:
            table[head(cname, kind)] = [f"Deep learning: {mlabel(prims[c]['nn'])}", "NA", "NA"]
            continue
        cfg = R.config(pid, c)
        table[head(cname, kind)] = [f"{PIPELINES[pid]['label']}: {R.final_model_name(pid, c)}",
                                    f"{int(cfg.get('n_test', 0)):,}", fmt(cfg.get("classification_threshold"), 4)]
    omitted = []
    for key, lab in metric_rows:
        vals = []
        for c, cname, kind, pid in cols:
            b = R.boot(pid, c) if pid else None
            if b is None or key not in b.index or not np.isfinite(b.at[key, "point"]):
                vals = None; break
            r = b.loc[key]
            vals.append(fmt_ci(r.point, r.ci_low, r.ci_high, 4 if key == "brier" else 3))
        if vals is None:
            omitted.append(lab); continue
        table["Metric"].append(f"{lab} (95% CI)")
        for (c, cname, kind, pid), v in zip(cols, vals):
            table[head(cname, kind)].append(v)
    foot = ("Locked test set 2023-2024; bootstrap 95% CIs from 05_test_metrics_bootstrap.csv. Primary reporting models: "
            "the lead single-model pipeline (Python vs R tidymodels, by validation ROC-AUC), the soft-voting ensemble of "
            "the same framework, and the top deep learning model.")
    if omitted:
        foot += f" Rows omitted because at least one pipeline did not report a point estimate: {', '.join(omitted)}."
    W.table("table5_primary_pipeline_test_performance",
            "Table 5. Locked test-set performance: primary reporting pipeline per cohort",
            [("", pd.DataFrame(table))], foot)


R_SINGLE, R_ENSEMBLE = "P4", "P3"


def table_r_lasso(R, W):
    """R tidymodels LASSO feature stability, both cohorts (benchmark counterpart of Table 2)."""
    parts, notes = [], []
    for L, (cohort, cname) in zip("AB", COHORTS):
        st = _stability(R, R_SINGLE, cohort)
        if st is None:
            continue
        sel = set(R.selected_features(R_SINGLE, cohort))
        folds = R.csv(R_SINGLE, cohort, "02a_lasso_nested_cv_outer_folds.csv")
        k = len(folds) if folds is not None else None
        parts.append((f"{L}. {cname}: R tidymodels LASSO (R tidymodels single-model pipeline)", pd.DataFrame({
            "Rank": range(1, len(st) + 1),
            "Feature": [feat(f) for f in st.feature],
            f"Times selected (of {k} outer folds)" if k else "Times selected": st.times_selected.astype(int).values,
            "Selection frequency": [f"{v:.2f}" for v in st.stability],
            "Included in final model": ["Yes" if f in sel else "No" for f in st.feature],
        })))
        sm = R.csv(R_SINGLE, cohort, "02d_lasso_nested_cv_summary.csv")
        if sm is not None:
            sm = sm.iloc[0]
            notes.append(f"{L}: nested outer-CV ROC-AUC {sm.outer_roc_auc_mean:.3f} (SD {sm.outer_roc_auc_std:.3f}), "
                         f"{int(sm.n_stable_features)} stable features")
    W.table("tableR1_r_lasso_feature_stability",
            "Supplementary Table R1. LASSO feature stability, R tidymodels pipelines (training data)", parts,
            "Benchmark counterpart of Table 2. Top 15 features by outer-fold selection frequency (ties kept in pipeline "
            "order) from the R tidymodels single-model pipeline in each cohort. " + "; ".join(notes) +
            ". The R tidymodels soft-voting pipeline ran its own LASSO; the Python LASSO used for the reported models is "
            "in Table 2.")


def table_r_classifiers(R, W):
    """Top ten R tidymodels classifiers by validation ROC-AUC (benchmark counterpart of Table 3A)."""
    parts = []
    for L, (cohort, cname) in zip("AB", COHORTS):
        vs = R.csv(R_SINGLE, cohort, "03b_val_model_selection.csv")
        cv = R.csv(R_SINGLE, cohort, "03_cv_model_comparison.csv")
        if vs is None or cv is None:
            continue
        m = vs.merge(cv[["model", "roc_auc_mean", "roc_auc_std"]], on="model", how="left")
        m = m.sort_values("val_roc_auc", ascending=False).head(10)
        chosen = R.single_model(R_SINGLE, cohort)
        members = set(R.ensemble_members(R_ENSEMBLE, cohort))
        parts.append((f"{L}. {cname}: R tidymodels single-model pipeline ({len(vs)} classifiers)", pd.DataFrame({
            "Rank": range(1, len(m) + 1),
            "Model": [mlabel(x) for x in m.model],
            "Train CV ROC-AUC, mean (SD)": [f"{a:.3f} ({b:.3f})" for a, b in zip(m.roc_auc_mean, m.roc_auc_std)],
            "Validation ROC-AUC": [f"{v:.3f}" for v in m.val_roc_auc],
            "Selected single model": ["Yes" if x == chosen else "No" for x in m.model],
            "Soft-voting member": ["Yes" if x in members else "No" for x in m.model],
        })))
    W.table("tableR2_r_top_classifiers_validation",
            "Supplementary Table R2. Top ten classifiers by validation ROC-AUC, R tidymodels pipelines", parts,
            "Benchmark counterpart of Table 3A. Validation years 2020-2022; Train CV is grouped cross-validation by "
            "survey year on the training split. Soft-voting members are those of the R tidymodels soft-voting pipeline, "
            "which ran its own feature selection and model fitting.")


def table_r_test(R, W):
    """Locked test-set performance of the R tidymodels pipelines (benchmark counterpart of Table 5)."""
    metric_rows = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("accuracy", "Accuracy"),
                   ("sensitivity", "Sensitivity"), ("specificity", "Specificity"),
                   ("f1", "F1 score"), ("brier", "Brier score")]
    cols = [(c, cn, pid) for c, cn in COHORTS for pid in (R_SINGLE, R_ENSEMBLE)]
    head = lambda cn, pid: f"{cn}: {'single model' if pid == R_SINGLE else 'soft-voting ensemble'}"  # noqa: E731
    table = {"Metric": ["Final model", "Test n", "Classification threshold (Youden J, validation)",
                        "Validation ROC-AUC"]}
    for c, cn, pid in cols:
        cfg = R.config(pid, c)
        table[head(cn, pid)] = [R.final_model_name(pid, c), f"{int(cfg.get('n_test', 0)):,}",
                                fmt(cfg.get("classification_threshold"), 4), fmt(R.final_val_auc(pid, c))]
    omitted = []
    for key, lab in metric_rows:
        vals = []
        for c, cn, pid in cols:
            b = R.boot(pid, c)
            if b is None or key not in b.index or not np.isfinite(b.at[key, "point"]):
                vals = None; break
            r = b.loc[key]
            vals.append(fmt_ci(r.point, r.ci_low, r.ci_high, 4 if key == "brier" else 3))
        if vals is None:
            omitted.append(lab); continue
        table["Metric"].append(f"{lab} (95% CI)")
        for (c, cn, pid), v in zip(cols, vals):
            table[head(cn, pid)].append(v)
    foot = ("Benchmark counterpart of Table 5. Locked test set 2023-2024; bootstrap 95% CIs from "
            "05_test_metrics_bootstrap.csv. The R tidymodels pipelines were run independently of the Python pipelines, "
            "with their own LASSO feature selection, model fitting, calibration and threshold tuning.")
    if omitted:
        foot += f" Rows omitted because a point estimate was not reported: {', '.join(omitted)}."
    W.table("tableR3_r_test_performance",
            "Supplementary Table R3. Locked test-set performance, R tidymodels pipelines", [("", pd.DataFrame(table))],
            foot)


def _glm_table(R, cohort):
    if cohort == "ny":
        df = R.csv("posthoc", "glm_ny", "glm_or_table_train_svy_weighted.r.csv")
        label = "design-weighted survey logistic regression (svyglm)"
    else:
        df = R.csv("posthoc", "glm_all_states", "glm_or_table_train_unweighted.r.csv")
        label = "unweighted logistic regression"
    if df is None:
        return None, label
    df = df[~df["term"].astype(str).str.contains("Intercept")].copy()
    df = df.rename(columns={"estimate": "or", "conf.low": "lo", "conf.high": "hi", "p.value": "p"})
    for c in ["or", "lo", "hi", "p", "q_value_fdr_bh"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["sig"] = df["significant_fdr_05"].astype(str).str.upper().eq("TRUE")
    return df, label


def table6(R, W):
    parts = []
    for L, (cohort, cname) in zip("AB", COHORTS):
        df, label = _glm_table(R, cohort)
        if df is None:
            continue
        s = df[df.q_value_fdr_bh < 0.05].sort_values("q_value_fdr_bh")
        parts.append((f"{L}. {cname}: {label}", pd.DataFrame({
            "Predictor": [feat(t) for t in s.term],
            "aOR (95% CI)": [f"{o:.2f} ({lo:.2f}–{hi:.2f})" for o, lo, hi in zip(s["or"], s.lo, s.hi)],
            "P value": [fmt_p(p) for p in s.p],
            "q (BH)": [fmt_p(q) for q in s.q_value_fdr_bh],
        })))
    W.table("table6_glm_fdr_predictors",
            "Table 6. GLM predictors with FDR q < 0.05 (LASSO feature set, training data)", parts,
            "aOR, adjusted odds ratio per one-unit increase in the BRFSS-coded predictor. q, Benjamini-Hochberg FDR "
            "across LASSO-selected predictors. Source: posthoc/glm_*/glm_or_table_train_*.r.csv.")


# ---------------------------------------------------------------------------
# Figure building blocks
# ---------------------------------------------------------------------------
def _hbar(ax, labels, values, err=None, color=BLUE, xlabel=""):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, xerr=err, color=color, height=0.7, edgecolor="white", linewidth=0.5,
            error_kw=dict(ecolor=INK2, lw=0.7, capsize=1.5))
    ax.set_yticks(y, labels, fontsize=7.5)
    ax.set_xlabel(xlabel)
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)


def _bars(ax, items, ylabel, baseline=0.5, ymax=None, ref=None, ref_label=None, width=0.72, vfs=7):
    """items: (x, label, value, lo, hi, color)."""
    for x, _, v, lo, hi, col in items:
        if not np.isfinite(v):
            continue
        ax.bar(x, v - baseline, bottom=baseline, width=width, color=col, edgecolor="white", linewidth=0.8)
        if np.isfinite(lo) and np.isfinite(hi):
            ax.errorbar(x, v, yerr=[[v - lo], [hi - v]], color=INK, lw=0.8, capsize=2.5)
        top = hi if np.isfinite(hi) else v
        ax.text(x, top + (ymax - baseline) * 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=vfs, color=INK2)
    ax.set_xticks([i[0] for i in items], [i[1] for i in items], fontsize=7)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_ylim(baseline, ymax)
    if ref is not None:
        ax.axhline(ref, color=INK2, ls="--", lw=0.8)
        ax.text(ax.get_xlim()[1], ref, f" {ref_label}", va="center", ha="left", fontsize=7, color=INK2)


def _framework_labels(ax, groups, offset=-30, fs=7.5):
    for name, xs in groups:
        ax.annotate(name, xy=(np.mean(xs), 0), xycoords=("data", "axes fraction"), xytext=(0, offset),
                    textcoords="offset points", ha="center", va="top", fontsize=fs, fontweight="bold", color=INK)


def _perm_title(R, cohort, kind, pid):
    if pid == "P7":
        if R.dl_fallback(cohort, "09_permutation_importance.csv"):
            # importances come from the base run, so name that run's best network, not the reported run's
            p = R.root / PIPELINES["P7"]["dir"] / cohort / "pipeline_config.json"
            best = json.loads(p.read_text()).get("best_single_model", "") if p.exists() else ""
            if p.exists():
                R.used.add(str(p))
            return f"Permutation importance: deep learning, base run, best single network ({mlabel(best)})"
        best = R.config("P7", cohort).get("best_single_model", "")
        return f"Permutation importance: deep learning, best single network ({mlabel(best)})"
    return f"Permutation importance: {PIPELINES[pid]['label']} ({R.final_model_name(pid, cohort)})"


def fig2(R, W, cohort, prim):
    label, stem, title = fig_id(2, cohort)
    s, e = prim["single"], prim["ensemble"]
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 2.65 * 2 + 0.3), layout="constrained")
    ax = axes[0, 0]
    st = _stability(R, s, cohort, top=12)
    _hbar(ax, [feat(f) for f in st.feature], st.stability, color=BLUE, xlabel="Selection frequency")
    ax.set_xlim(0, 1.05)
    ax.set_title(wrap(f"LASSO selection frequency: {PIPELINES[s]['framework']} LASSO (training data)", 44), loc="left", fontsize=8)
    for ax, (kind, pid) in zip([axes[0, 1], axes[1, 0], axes[1, 1]], [("single", s), ("ensemble", e), ("nn", "P7")]):
        pi = R.csv(pid, cohort, "09_permutation_importance.csv").sort_values("importance_mean", ascending=False).head(12)
        _hbar(ax, [feat(f) for f in pi.feature], pi.importance_mean, pi.importance_std,
              color=ORANGE if kind == "ensemble" else BLUE, xlabel="Mean decrease in ROC-AUC (validation)")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.set_title(wrap(_perm_title(R, cohort, kind, pid), 44), loc="left", fontsize=8)
    letters(fig, zip(axes.flatten(), "ABCD"))
    note = (f"A: LASSO run of the {PIPELINES[s]['label']} pipeline. B-D: permutation importance on validation data for "
            "the lead single model, the soft-voting ensemble of the same framework and the deep learning pipeline's best "
            "single architecture.")
    if PIPELINES[s]["framework"] != "Python":
        note += " The R soft-voting pipeline ran its own LASSO (Table 2); the deep learning pipeline used the Python LASSO."
    if R.dl_fallback(cohort, "09_permutation_importance.csv"):
        note += (f" D: the {R.dl_subrun[cohort]} deep learning run did not save permutation importances; panel D shows "
                 "the base deep learning run for this cohort.")
    W.fig(fig, label, title, stem, note)


def _dotplot(ax, R, s, e, cohort):
    vs = R.csv(s, cohort, "03b_val_model_selection.csv")
    cv = R.csv(s, cohort, "03_cv_model_comparison.csv")
    m = vs.merge(cv[["model", "roc_auc_mean"]], on="model", how="left").sort_values("val_roc_auc")
    chosen = R.single_model(s, cohort)
    members = set(R.ensemble_members(e, cohort))
    y = np.arange(len(m))
    for yi, (_, r) in zip(y, m.iterrows()):
        ax.plot([r.roc_auc_mean, r.val_roc_auc], [yi, yi], color=GRID, lw=1.5, zorder=1)
    face = [BLUE if x == chosen else (ORANGE if x in members else LIGHT_BLUE) for x in m.model]
    edge = [ORANGE if (x == chosen and x in members) else "white" for x in m.model]
    lw = [1.6 if (x == chosen and x in members) else 0.6 for x in m.model]
    ax.scatter(m.val_roc_auc, y, s=30, color=face, edgecolor=edge, lw=lw, zorder=3)
    ax.scatter(m.roc_auc_mean, y, s=26, facecolor="white", edgecolor=INK2, lw=0.9, zorder=2)
    # label the selected single model's validation ROC-AUC next to its point
    sel = m.reset_index(drop=True)
    for yi, (_, r) in enumerate(sel.iterrows()):
        if r.model == chosen:
            ax.text(max(r.val_roc_auc, r.roc_auc_mean) + 0.016, yi, f"{r.val_roc_auc:.3f}", va="center", ha="left",
                    fontsize=7, color=INK)
    lo_x = min(m.val_roc_auc.min(), m.roc_auc_mean.min())
    hi_x = max(m.val_roc_auc.max(), m.roc_auc_mean.max())
    ax.set_xlim(lo_x - 0.02, hi_x + 0.06)
    ax.set_yticks(y, [mlabel(x) for x in m.model], fontsize=7.5)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("ROC-AUC")
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title(wrap(f"{PIPELINES[s]['label']} pipeline: {len(m)} classifiers, training CV vs validation", 46),
                 loc="left", fontsize=8)


def fig3(R, W, cohort, prim):
    label, stem, title = fig_id(3, cohort)
    s, e = prim["single"], prim["ensemble"]
    fw = PIPELINES[s]["framework"]
    fig = plt.figure(figsize=(FIG_W, 6.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0], hspace=0.36, wspace=0.95)

    axA = fig.add_subplot(gs[0, 0])
    _dotplot(axA, R, s, e, cohort)

    axB = fig.add_subplot(gs[0, 1])
    nn = R.csv("P7", cohort, "03b_val_model_selection.csv")[["model", "val_roc_auc"]]
    nn = pd.concat([nn, pd.DataFrame([{"model": "ensemble", "val_roc_auc": R.final_val_auc("P7", cohort)}])])
    nn = nn.sort_values("val_roc_auc").reset_index(drop=True)
    y = np.arange(len(nn))
    lo = nn.val_roc_auc.min() - 0.01
    axB.hlines(y, lo, nn.val_roc_auc, color=GRID, lw=1.5)
    axB.scatter(nn.val_roc_auc, y, s=36, zorder=3, edgecolor="white", lw=0.6,
                color=[ARCH_COLOR.get(a, GRAY) for a in nn.model])
    for yi, v in zip(y, nn.val_roc_auc):
        axB.text(v + 0.0015, yi, f"{v:.3f}", va="center", fontsize=6.8, color=INK2)
    axB.set_yticks(y, [mlabel(a) for a in nn.model], fontsize=7.5)
    axB.tick_params(axis="y", length=0)
    axB.set_xlim(lo, nn.val_roc_auc.max() + 0.012)
    axB.set_xlabel("Validation ROC-AUC")
    axB.set_title("Deep learning (PyTorch) architectures", loc="left", fontsize=8)

    axC = fig.add_subplot(gs[1, 0])
    nnv = R.nn_val(cohort)
    items = [(0, "Single", R.final_val_auc(s, cohort), np.nan, np.nan, BLUE),
             (0.75, "Soft-\nvoting", R.final_val_auc(e, cohort), np.nan, np.nan, ORANGE),
             (1.85, "FT-T", nnv.get("ft_transformer", np.nan), np.nan, np.nan, ARCH_COLOR["ft_transformer"]),
             (2.6, "FT-T-\nlarge", nnv.get("ft_transformer_large", np.nan), np.nan, np.nan, ARCH_COLOR["ft_transformer_large"]),
             (3.35, "Top-2\nens.", nnv.get("ensemble", np.nan), np.nan, np.nan, ORANGE)]
    top = max(i[2] for i in items if np.isfinite(i[2]))
    _bars(axC, items, "Validation ROC-AUC", baseline=0.5, ymax=top + 0.06, width=0.6, vfs=6)
    axC.set_yticks([0.5, 0.6, 0.7, 0.8])
    for t in axC.get_xticklabels():
        t.set_fontsize(6.5)
    _framework_labels(axC, [(fw, [0, 0.75]), ("Deep learning", [1.85, 3.35])], offset=-24, fs=7)
    axC.set_title(wrap(f"Validation: single model ({R.final_model_name(s, cohort)}) vs ensembles", 46), loc="left", fontsize=8)

    axD = fig.add_subplot(gs[1, 1])
    cv = R.csv(s, cohort, "03_cv_model_comparison.csv").set_index("model")
    chosen = R.single_model(s, cohort)
    sm, ssd = float(cv.at[chosen, "roc_auc_mean"]), float(cv.at[chosen, "roc_auc_std"])
    ec = R.csv(e, cohort, "04_ensemble_comparison.csv")
    col = [c for c in ec.columns if c != "ensemble"][0]
    ev = ec.loc[ec.ensemble.str.contains("traincv"), col]
    ev = float(ev.iloc[0]) if len(ev) else np.nan
    items = [(0, "Single\nmodel", sm, sm - ssd, sm + ssd, BLUE), (0.9, "Soft-voting\nensemble", ev, np.nan, np.nan, ORANGE)]
    _bars(axD, items, "Training CV ROC-AUC", baseline=0.5, ymax=max(sm + ssd, ev) + 0.06, width=0.55, vfs=6.5)
    axD.set_xlim(-0.6, 1.5)
    axD.set_yticks([0.5, 0.6, 0.7, 0.8])
    axD.set_title(wrap(f"Training CV: single model vs soft-voting ensemble ({fw})", 46), loc="left", fontsize=8)

    handles = [Line2D([], [], marker="o", ls="", color=BLUE, label="Selected single model"),
               Line2D([], [], marker="o", ls="", color=ORANGE, label="Soft-voting member / ensemble"),
               Line2D([], [], marker="o", ls="", color=LIGHT_BLUE, label="Other classifier (validation)"),
               Line2D([], [], marker="o", ls="", markerfacecolor="white", color=INK2, label="Training CV mean"),
               Patch(color=AQUA, label="FT-Transformer-large"), Patch(color=GRAY, label="Other architectures")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), fontsize=7.5)
    letters(fig, zip([axA, axB, axC, axD], "ABCD"))
    note = ("FT-T, FT-Transformer; ens., ensemble. D: single model shown as training CV mean (SD); ensemble value from "
            "04_ensemble_comparison.csv. Deep learning ensemble validation value from pipeline_config.json.")
    if fw != "Python":
        note += " A: classifier values from the R single-model pipeline; soft-voting members from the R soft-voting pipeline."
    W.fig(fig, label, title, stem, note)


def fig4(R, W, cohort, prim):
    label, stem, title = fig_id(4, cohort)
    s, e = prim["single"], prim["ensemble"]
    fw = PIPELINES[s]["framework"]
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 3.5))
    X = {s: 0, e: 1, "P7": 2.3}
    ticks = {s: "Single\nmodel", e: "Soft-voting\nensemble", "P7": "Top-2\nensemble"}
    for ax, (metric, ylab) in zip(axes, [("roc_auc", "Test ROC-AUC (95% CI)"), ("pr_auc", "Test PR-AUC (95% CI)")]):
        items = []
        for p in (s, e, "P7"):
            b = R.boot(p, cohort)
            if b is None or metric not in b.index:
                continue
            v = b.loc[metric]
            items.append((X[p], ticks[p], v.point, v.ci_low, v.ci_high, PIPE_COLOR[p]))
        top = max(i[4] for i in items if np.isfinite(i[4]))
        if metric == "roc_auc":
            _bars(ax, items, ylab, baseline=0.5, ymax=top + 0.05, width=0.6, vfs=6.5)
        else:
            comp = R.csv("posthoc", None, "pipeline_comparison_both.csv")
            prev = np.nan
            if comp is not None:
                sr = comp.loc[comp.dataset == cohort, "test_prevalence_recomputed"].dropna()
                prev = float(sr.iloc[0]) if len(sr) else np.nan
            _bars(ax, items, ylab, baseline=0.0, ymax=top * 1.18, width=0.6, vfs=6.5,
                  ref=prev if np.isfinite(prev) else None,
                  ref_label=f"prevalence\n {prev:.3f}" if np.isfinite(prev) else None)
        ax.set_xlim(-0.6, 2.9)
        _framework_labels(ax, [(fw, [0, 1]), ("Deep learning", [2.3])], offset=-26, fs=7)
    handles = [Patch(color=BLUE, label="Single model"), Patch(color=ORANGE, label="Ensemble")]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(w_pad=3)
    letters(fig, zip(axes, "AB"))
    W.fig(fig, label, title, stem,
          f"{fw} single-model and soft-voting pipelines (lead framework by validation ROC-AUC) and the deep learning top-2 "
          "ensemble. All five pipelines are compared in Table 4. Single FT-Transformer and FT-Transformer-large test "
          "metrics were not saved.")


def scored_pair(R, cohort, prim):
    s, e = prim["single"], prim["ensemble"]
    if R.scored(s, cohort, "test") is not None:
        return [p for p in (s, e) if R.scored(p, cohort, "test") is not None], True
    return ["P1", "P2"], False


def _fallback_note(ok):
    return ""


def _discrimination_row(R, cohort, pid, axA, axB):
    sc = R.scored(pid, cohort, "test")
    b = R.boot(pid, cohort)
    y = sc.stroke.to_numpy().astype(int)
    yp = sc.stroke_pred.to_numpy().astype(int)
    cm = np.array([[np.sum((y == 0) & (yp == 0)), np.sum((y == 0) & (yp == 1))],
                   [np.sum((y == 1) & (yp == 0)), np.sum((y == 1) & (yp == 1))]])
    axA.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15)
    names = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            axA.text(j, i, f"{names[i][j]}\n{cm[i, j]:,}", ha="center", va="center", fontsize=8.5,
                     color="white" if cm[i, j] > cm.max() * 0.55 else INK)
    axA.set_xticks([0, 1], ["No stroke", "Stroke"]); axA.set_yticks([0, 1], ["No stroke", "Stroke"])
    axA.set_xlabel("Predicted"); axA.set_ylabel("Observed (BRFSS)")
    for sp in axA.spines.values():
        sp.set_visible(False)
    thr = float(sc.classification_threshold.iloc[0])
    axA.set_title(wrap(f"{PIPELINES[pid]['label']}: {R.final_model_name(pid, cohort)}", 46)
                  + f"\nn = {len(sc):,}; threshold = {thr:.4f}", loc="left", fontsize=8)
    r = b.loc["roc_auc"]
    fpr, tpr = roc_points(y, sc.stroke_prob.to_numpy())
    axB.plot(fpr, tpr, color=PIPE_COLOR[pid], lw=2, label=f"AUC = {r.point:.3f} ({r.ci_low:.3f}–{r.ci_high:.3f})")
    axB.plot([0, 1], [0, 1], ls="--", color=GRAY, lw=0.9, label="Chance")
    axB.set_xlim(0, 1); axB.set_ylim(0, 1.01); axB.set_aspect("equal")
    axB.set_xlabel("False positive rate (1 - specificity)"); axB.set_ylabel("True positive rate (sensitivity)")
    axB.grid(True); axB.set_axisbelow(True)
    axB.legend(loc="lower right", fontsize=7.5)
    axB.set_title(f"ROC curve: {PIPELINES[pid]['label']}", loc="left", fontsize=8)


def fig5(R, W, cohort, prim):
    label, stem, title = fig_id(5, cohort)
    pids, ok = scored_pair(R, cohort, prim)
    fig, axes = plt.subplots(len(pids), 2, figsize=(FIG_W, 3.6 * len(pids)), squeeze=False)
    for r, pid in enumerate(pids):
        _discrimination_row(R, cohort, pid, axes[r, 0], axes[r, 1])
    fig.tight_layout(h_pad=2.5, w_pad=3)
    letters(fig, zip(axes.flatten(), "ABCD"))
    W.fig(fig, label, title, stem, "Rows: single-model pipeline, then soft-voting ensemble of the same framework." + _fallback_note(ok))


def fig6(R, W, cohort, prim):
    label, stem, title = fig_id(6, cohort)
    pids, ok = scored_pair(R, cohort, prim)
    fig, axes = plt.subplots(len(pids), 2, figsize=(FIG_W, 3.4 * len(pids)), squeeze=False)
    for r, pid in enumerate(pids):
        sc = R.scored(pid, cohort, "test")
        s, y = sc.stroke_prob.to_numpy(), sc.stroke.to_numpy()
        ax = axes[r, 0]
        pr, ob = calibration_bins(y, s)
        br = R.boot(pid, cohort).loc["brier"]
        ax.plot(pr, ob, marker="o", ms=4, lw=1.6, color=PIPE_COLOR[pid], label="Observed vs predicted (deciles)")
        lim = max(pr.max(), ob.max()) * 1.08
        ax.plot([0, lim], [0, lim], ls=":", color=GRAY, lw=1, label="Perfect calibration")
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("Mean predicted P(stroke)"); ax.set_ylabel("Observed stroke fraction")
        ax.grid(True); ax.set_axisbelow(True)
        ax.legend(loc="upper left", fontsize=7)
        ax.text(0.97, 0.04, f"Brier {fmt_ci(br.point, br.ci_low, br.ci_high, 4)}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5, color=INK2)
        ax.set_title(wrap(f"Calibration: {PIPELINES[pid]['label']} ({R.final_model_name(pid, cohort)})", 48),
                     loc="left", fontsize=8)
        ax = axes[r, 1]
        hi = float(np.quantile(s, 0.995))
        for val, colr, lab in [(0, BLUE, "No stroke"), (1, ORANGE, "Stroke")]:
            x, d = smooth_density(s[y == val], 0, hi)
            ax.plot(x, d, color=colr, lw=1.8, label=f"{lab} (n = {int((y == val).sum()):,})")
            ax.fill_between(x, d, color=colr, alpha=0.12, lw=0)
        thr = float(sc.classification_threshold.iloc[0])
        ax.axvline(thr, color=INK2, ls="--", lw=0.8)
        ax.text(thr, ax.get_ylim()[1] * 0.97, f" threshold {thr:.3f}", fontsize=7, color=INK2, va="top")
        ax.set_xlim(0, hi)
        ax.set_xlabel("Predicted P(stroke)"); ax.set_ylabel("Density")
        ax.legend(loc="upper right", fontsize=7)
        ax.set_title(f"Score distribution: {PIPELINES[pid]['label']}", loc="left", fontsize=8)
    fig.tight_layout(h_pad=2.5, w_pad=2.5)
    letters(fig, zip(axes.flatten(), "ABCD"))
    W.fig(fig, label, title, stem, "Rows: single-model pipeline, then soft-voting ensemble of the same framework. "
          "Row-level scores were not saved by the deep learning pipeline." + _fallback_note(ok))


def fig7(R, W, cohort, prim):
    label, stem, title = fig_id(7, cohort)
    pids, ok = scored_pair(R, cohort, prim)
    fig, axes = plt.subplots(len(pids), 2, figsize=(FIG_W, 3.3 * len(pids) + 0.3), squeeze=False,
                             gridspec_kw=dict(width_ratios=[2, 3]))
    for r, pid in enumerate(pids):
        for c, split in enumerate(["test", "val"]):
            ax = axes[r, c]
            sc = R.scored(pid, cohort, split)
            if sc is None:
                ax.axis("off"); continue
            years = sorted(sc.survey_year.unique())
            data, pos, cols = [], [], []
            for i, yr in enumerate(years):
                for k, (val, colr) in enumerate([(0, BLUE), (1, ORANGE)]):
                    data.append(sc.loc[(sc.survey_year == yr) & (sc.stroke == val), "stroke_prob"].to_numpy())
                    pos.append(i + (k - 0.5) * 0.36); cols.append(colr)
            bp = ax.boxplot(data, positions=pos, widths=0.3, showfliers=False, patch_artist=True,
                            medianprops=dict(color=INK, lw=1), whiskerprops=dict(color=INK2, lw=0.8),
                            capprops=dict(color=INK2, lw=0.8), boxprops=dict(lw=0))
            for patch, colr in zip(bp["boxes"], cols):
                patch.set_facecolor(colr); patch.set_alpha(0.85)
            for v, p in zip(data, pos):
                if v.size:
                    ax.hlines(v.mean(), p - 0.17, p + 0.17, colors=INK, linestyles="--", lw=1)
            ax.set_xticks(range(len(years)), [str(int(yy)) for yy in years])
            ax.set_xlabel("Survey year"); ax.set_ylabel("Predicted P(stroke)")
            ax.yaxis.grid(True); ax.set_axisbelow(True)
            period = "locked test years" if split == "test" else "validation years"
            ax.set_title(wrap(f"{PIPELINES[pid]['label']}: {period}", 40), loc="left", fontsize=8)
    handles = [Patch(color=BLUE, label="BRFSS: no stroke"), Patch(color=ORANGE, label="BRFSS: stroke"),
               Line2D([], [], color=INK, ls="--", label="Group mean"), Line2D([], [], color=INK, label="Median")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout(h_pad=2.2, w_pad=3)
    letters(fig, zip(axes.flatten(), "ABCD"))
    W.fig(fig, label, title, stem, "Rows: single-model pipeline, then soft-voting ensemble of the same framework. Stroke "
          "status is the evaluation label, not a model input." + _fallback_note(ok))


def _forest(ax, df, label):
    d = df.assign(abslog=np.log(df["or"]).abs()).sort_values("abslog", ascending=False).head(10).iloc[::-1]
    y = np.arange(len(d))
    for yi, (_, r) in zip(y, d.iterrows()):
        col = BLUE if r["or"] >= 1 else ORANGE
        ax.plot([r.lo, r.hi], [yi, yi], color=col, lw=1.4)
        ax.scatter(r["or"], yi, s=36, marker="s" if r["or"] >= 1 else "o", zorder=3,
                   facecolor=col if r.sig else "white", edgecolor=col, lw=1.2)
        ax.text(1.02, yi, f"{r['or']:.2f} ({r.lo:.2f}–{r.hi:.2f})", transform=ax.get_yaxis_transform(),
                va="center", fontsize=7, color=INK2)
    ax.axvline(1, color=INK2, lw=0.8)
    ax.set_xscale("log")
    lo_, hi_ = min(d.lo.min(), 0.9), max(d.hi.max(), 1.1)
    ticks = [t for t in (0.1, 0.2, 0.25, 0.5, 1, 2, 4, 8) if lo_ * 0.9 <= t <= hi_ * 1.1]
    ax.set_xlim(lo_ * 0.85, hi_ * 1.15)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(y, [feat(t) for t in d.term], fontsize=7.5); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Adjusted odds ratio (log scale)")
    ax.xaxis.grid(True, which="major"); ax.set_axisbelow(True)
    ax.set_title(f"Top 10 GLM associations by |log aOR|, training data: {label}", loc="left", fontsize=8)


def _shap_title_and_note(R, pid, cohort, letter):
    """Panel title and figure-note text for one SHAP panel. For an ensemble, the title names only the members
    whose SHAP values were averaged (12_shap_meta.json), and the note names any member left out."""
    if PIPELINES[pid]["kind"] != "ensemble":
        return f"SHAP: {PIPELINES[pid]['label']} ({R.final_model_name(pid, cohort)})", ""
    members = R.ensemble_members(pid, cohort)
    shap_mem = R.shap_members(pid, cohort)
    if not shap_mem:
        return (f"SHAP: {PIPELINES[pid]['label']} ({R.final_model_name(pid, cohort)})",
                f"{letter}: the pipeline did not record which ensemble members its SHAP values cover.")
    title = f"SHAP: {PIPELINES[pid]['label']}, mean of {join_and(mlabel(m) for m in shap_mem)} SHAP values"
    left_out = [m for m in members if m not in shap_mem]
    note = (f"{letter}: ensemble SHAP is the mean of the saved member SHAP values for "
            f"{join_and(mlabel(m) for m in shap_mem)}")
    if left_out:
        s = "s" if len(left_out) > 1 else ""
        note += (f"; {join_and(mlabel(m) for m in left_out)} (ensemble member{s}) had no saved SHAP values and "
                 f"{'are' if s else 'is'} not included")
        R.shap_partial.append((CNAME[cohort], PIPELINES[pid]["label"], shap_mem, left_out))
    return title, note + "."


def fig8(R, W, cohort, prim):
    label, stem, title = fig_id(8, cohort)
    lead = [prim["single"], prim["ensemble"]]
    ok = all(R.path(p, cohort, "12_shap_mean_abs.csv").exists() for p in lead)
    pids = lead if ok else ["P1", "P2"]
    fig = plt.figure(figsize=(FIG_W, 6.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.2, 1], hspace=0.55, wspace=1.6)
    tagged, notes = [], []
    for i, pid in enumerate(pids):
        ax = fig.add_subplot(gs[0, i * 2:(i + 1) * 2])
        sh = R.csv(pid, cohort, "12_shap_mean_abs.csv")
        if sh is None:
            ax.axis("off"); continue
        sh = sh.sort_values("shap_mean_abs", ascending=False).head(15)
        _hbar(ax, [feat(f) for f in sh.feature], sh.shap_mean_abs, color=PIPE_COLOR[pid],
              xlabel="Mean |SHAP value| (validation)")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ptitle, pnote = _shap_title_and_note(R, pid, cohort, "AB"[len(tagged)])
        ax.set_title(wrap(ptitle, 40), loc="left", fontsize=8)
        if pnote:
            notes.append(pnote)
        tagged.append(ax)
    if len(tagged) == 2:
        notes.append("Mean |SHAP| values in A and B come from different models and explainers; compare the two panels "
                     "by feature rank, not by magnitude.")
    ax = fig.add_subplot(gs[1, 1:4])
    df, glab = _glm_table(R, cohort)
    if df is not None:
        _forest(ax, df, glab)
        tagged.append(ax)
        handles = [Line2D([], [], marker="s", color=BLUE, ls="-", label="aOR > 1"),
                   Line2D([], [], marker="o", color=ORANGE, ls="-", label="aOR < 1"),
                   Line2D([], [], marker="s", color=INK2, ls="", markerfacecolor=INK2, label="FDR q < 0.05 (filled)"),
                   Line2D([], [], marker="s", color=INK2, ls="", markerfacecolor="white", label="q ≥ 0.05 (open)")]
        fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    letters(fig, zip(tagged, "ABC"))
    note = " ".join(notes + [f"GLM: {glab}."])
    W.fig(fig, label, title, stem, note)


def _kg(ax, R, kgkey, cohort, n_nodes=18):
    d = KG_DIRS[kgkey]
    scores = R.csv(d, cohort, "01_kg_feature_pathway_scores.csv")
    paths = R.csv(d, cohort, "03_kg_shortest_pathways_to_stroke.csv")
    edges = R.csv(d, cohort, "02_kg_edges.csv")
    if scores is None or paths is None or edges is None:
        ax.axis("off"); return 0
    pm = paths.set_index("feature")
    top = scores[scores.feature.isin(pm.index)].sort_values("rank").head(n_nodes).reset_index(drop=True)
    dist = pm.loc[top.feature, "path_length"].to_numpy(dtype=float)
    hops = pm.loc[top.feature, "hops"].to_numpy(dtype=int)
    order = np.argsort(dist, kind="mergesort")
    rank = np.empty(len(dist), int); rank[order] = np.arange(len(dist))
    rr = 0.42 + 0.58 * rank / max(len(dist) - 1, 1)
    ang = np.pi / 2 + rank * np.deg2rad(137.508)
    xy = {f: (r * np.cos(a), r * np.sin(a)) for f, r, a in zip(top.feature, rr, ang)}
    med = np.median(dist)
    ff = edges[edges.source.isin(xy) & edges.target.isin(xy)]
    for _, e in ff.iterrows():
        (x1, y1), (x2, y2) = xy[e.source], xy[e.target]
        ax.plot([x1, x2], [y1, y2], color=VIOLET, ls=(0, (3, 2)), lw=0.7, alpha=0.5, zorder=1)
    for f, dv, h in zip(top.feature, dist, hops):
        x, y = xy[f]
        ax.plot([0, x], [0, y], color=BLUE if dv <= med else ORANGE, lw=1.8 if dv <= med else 1.1,
                ls="-" if h == 1 else (0, (1, 1.2)), zorder=2)
    ps_ = top.pathway_score.to_numpy(dtype=float)
    sizes = 40 + 180 * (ps_ - ps_.min()) / (np.ptp(ps_) or 1)
    for f, sz in zip(top.feature, sizes):
        x, y = xy[f]
        ax.scatter(x, y, s=sz, color="white", edgecolor=INK2, lw=1, zorder=3)
        a = math.atan2(y, x)
        ca, sa = math.cos(a), math.sin(a)
        ax.text(x + 0.09 * ca, y + 0.09 * sa, feat(f), fontsize=7, zorder=4,
                ha="left" if ca > 0.3 else ("right" if ca < -0.3 else "center"),
                va="bottom" if sa > 0.3 else ("top" if sa < -0.3 else "center"),
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.75))
    ax.scatter(0, 0, s=600, color=RED, zorder=5, edgecolor="white", lw=1.5)
    ax.text(0, 0, "Stroke", color="white", fontsize=8, fontweight="bold", ha="center", va="center", zorder=6)
    ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.25, 1.2); ax.set_aspect("equal"); ax.axis("off")
    return len(top)


def fig_kg(R, W, kgkey, cohort, label, title, stem):
    fig, ax = plt.subplots(figsize=(5.0, 4.9))
    n = _kg(ax, R, kgkey, cohort)
    handles = [Line2D([], [], color=BLUE, lw=1.8, label="Shorter path to stroke (closest half)"),
               Line2D([], [], color=ORANGE, lw=1.1, label="Longer path to stroke (farthest half)"),
               Line2D([], [], color=INK2, lw=1.1, ls=(0, (1, 1.2)), label="Multi-hop shortest path"),
               Line2D([], [], color=VIOLET, lw=0.9, ls=(0, (3, 2)), label="Feature-feature association"),
               Line2D([], [], marker="o", ls="", color=INK2, markerfacecolor="white", label="Larger node = higher pathway score")]
    ax.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.03), fontsize=7, frameon=False)
    W.fig(fig, label, title, stem,
          f"Top {n} predictors by composite pathway score. Radial position = rank of weighted shortest-path distance to "
          "stroke (03_kg_shortest_pathways_to_stroke.csv); innermost = closest.")


def fig9(R, W, cohort, prim):
    label, stem, title = fig_id(9, cohort)
    fig_kg(R, W, "data", cohort, label, title, stem)


# ---------------------------------------------------------------------------
# Supplementary S10-S17
# ---------------------------------------------------------------------------
def supp_id(n):
    t, s = EXTRA_SUPP[n]
    return f"Supplementary Figure S{n}", t, f"figureS{n}_{s}"


def figS10(R, W):
    label, title, stem = supp_id(10)
    ps = pd.read_csv(R.data / "pooling_summary_by_year.csv")
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 6.4))
    for r, (col, ylab) in enumerate([("stroke_rate", "Stroke prevalence (%)"), ("n", "Analytic sample (n)")]):
        for c, (cohort, cname) in enumerate(COHORTS):
            ax = axes[r, c]
            d = ps[ps.dataset == ("new_york" if cohort == "ny" else "all_states")]
            vals = d[col] * (100 if col == "stroke_rate" else 1)
            ax.bar(d.survey_year, vals, color=[SPLIT_COLOR[s] for s in d.split], width=0.72, edgecolor="white", lw=0.8)
            for x, v in zip(d.survey_year, vals):
                txt = f"{v:.1f}" if col == "stroke_rate" else (f"{v / 1000:.0f}k" if v >= 10000 else f"{v:,.0f}")
                ax.text(x, v, txt, ha="center", va="bottom", fontsize=6.5, color=INK2)
            ax.set_xticks(d.survey_year, [str(y) for y in d.survey_year], rotation=45)
            ax.set_ylabel(ylab); ax.yaxis.grid(True); ax.set_axisbelow(True)
            if col == "n":
                ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
            ax.set_title(cname, loc="left")
    handles = [Patch(color=SPLIT_COLOR[s], label=l) for s, l in
               [("train", "Train (2015-2019)"), ("val", "Validation (2020-2022)"), ("test", "Locked test (2023-2024)")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(h_pad=2.0)
    letters(fig, zip(axes.flatten(), "ABCD"))
    W.fig(fig, label, title, stem, "Unweighted, from pooling_summary_by_year.csv.")


def prevalence_stats(R) -> pd.DataFrame:
    rows = []
    for cohort, cname in COHORTS:
        df = R.demographics(cohort)
        for var, (vlab, codes) in DEMOGRAPHICS.items():
            for split, _ in SPLITS:
                d = df if split == "all" else df[df.split == split]
                excluded = int((~d[var].isin(list(codes))).sum())
                cnt = {}
                for code, glab in codes.items():
                    g = d[d[var] == code]
                    k, n = int(g.stroke.sum()), len(g)
                    p, lo, hi = wilson(k, n)
                    cnt[code] = (k, n)
                    rows.append(dict(cohort=cname, variable=var, variable_label=vlab, split=split, code=code,
                                     group=glab, n=n, stroke_cases=k, prevalence_pct=100 * p,
                                     wilson_low_pct=100 * lo, wilson_high_pct=100 * hi,
                                     excluded_missing_or_unknown=excluded))
                (k1, n1), (k2, n2) = cnt[1], cnt[2]
                chi2, pv = chisq_2x2(k1, n1 - k1, k2, n2 - k2)
                for rr in rows[-2:]:
                    rr.update(chi_square=chi2, p_value=pv)
    return pd.DataFrame(rows)


def _prev_panel(ax, st, title, var):
    codes = DEMOGRAPHICS[var][1]
    w = 0.36
    top = st.wilson_high_pct.max()
    for i, (split, _) in enumerate(SPLITS):
        s = st[st.split == split]
        for k, code in enumerate(codes):
            r = s[s.code == code].iloc[0]
            x = i + (k - 0.5) * w
            ax.bar(x, r.prevalence_pct, width=w * 0.95, color=GROUP_COLORS[k], edgecolor="white", lw=0.8)
            ax.errorbar(x, r.prevalence_pct, yerr=[[r.prevalence_pct - r.wilson_low_pct],
                                                   [r.wilson_high_pct - r.prevalence_pct]], color=INK, lw=0.8, capsize=2)
            ax.text(x, r.wilson_high_pct + top * 0.01, f"{r.prevalence_pct:.1f}", ha="center", va="bottom",
                    fontsize=6.3, color=INK2)
        pv = s.p_value.iloc[0]
        yb = s.wilson_high_pct.max() + top * 0.09
        ax.plot([i - w / 2, i - w / 2, i + w / 2, i + w / 2], [yb - top * 0.02, yb, yb, yb - top * 0.02], color=INK2, lw=0.7)
        ax.text(i, yb + top * 0.01, "P < 0.001" if pv < 0.001 else f"P = {fmt_p(pv)}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(len(SPLITS)), [s[1] for s in SPLITS], fontsize=7)
    ax.set_ylim(0, top * 1.28)
    ax.set_ylabel("Stroke prevalence, % (95% Wilson CI)")
    ax.yaxis.grid(True); ax.set_axisbelow(True)
    ax.set_title(title, loc="left")


def figS11_S13(R, W, stats):
    for n, variables in [(11, ["age_65plus"]), (12, ["race_group", "hispanic"]), (13, ["sex"])]:
        label, title, stem = supp_id(n)
        fig, axes = plt.subplots(len(variables), 2, figsize=(FIG_W, 3.9 * len(variables)), squeeze=False)
        for r, var in enumerate(variables):
            for c, (cohort, cname) in enumerate(COHORTS):
                st = stats[(stats.cohort == cname) & (stats.variable == var)]
                _prev_panel(axes[r, c], st, f"{cname}: {DEMOGRAPHICS[var][0]}", var)
            handles = [Patch(color=GROUP_COLORS[k], label=lab) for k, lab in enumerate(DEMOGRAPHICS[var][1].values())]
            axes[r, 1].legend(handles=handles, loc="upper right", fontsize=7)
        fig.tight_layout(h_pad=2.0)
        letters(fig, zip(axes.flatten(), "ABCD"))
        W.fig(fig, label, title, stem,
              "Unweighted prevalence; P values from Pearson chi-square tests (df = 1) comparing the two groups within "
              "each split. Missing/unknown codes excluded (counts in tables/tableS_prevalence_by_group.csv).")


def figS14_S15(R, W):
    for n, cohort in [(14, "all_states"), (15, "ny")]:
        label, title, stem = supp_id(n)
        fig_kg(R, W, "ml", cohort, label, title, stem)


# ---------------------------------------------------------------------------
# Provenance and figure list
# ---------------------------------------------------------------------------
def write_readme(R, W, prims):
    def sortkey(f):
        lab = f[0]
        return (0 if lab.startswith("Figure") else 1, int(re.sub(r"\D", "", lab)))
    figs = sorted(W.figs, key=sortkey)
    lst = ["# Figure list", "",
           "Main figures show the National cohort; Supplementary Figures S2-S9 are the New York versions of Figures 2-9. "
           "Supplementary Figure S1 is reserved for the New York version of Figure 1 (workflow, not generated). "
           "Figures show only the lead framework (Python or R tidymodels, whichever single-model pipeline has the higher "
           "validation ROC-AUC) plus the deep learning pipeline.", ""]
    for lab, title, stem, note in figs:
        lst.append(f"- **{lab}. {title}.** `figures/{stem}.png`" + (f"  \n  Note: {note}" if note else ""))
    lst += ["", "# Table list", ""] + [f"- {t[1]} (`tables/{t[0]}.csv`)" for t in W.tables]
    (W.out / "figure_list.md").write_text("\n".join(lst), encoding="utf-8")

    lines = ["# Manuscript figures and tables: provenance", "",
             f"Results directory: `{R.root}`  ", f"Processed data: `{R.data}`  ",
             f"Generated by: `{Path(__file__).name}` (version 5.2)", "",
             "## Deep learning run used per cohort", ""] + [
             f"- {cname}: `{PIPELINES['P7']['dir']}/{cohort}/{R.dl_subrun[cohort]}/`" if cohort in R.dl_subrun
             else f"- {cname}: `{PIPELINES['P7']['dir']}/{cohort}/` (base run)" for cohort, cname in COHORTS] + [
             *([""] + [f"- {CNAME[c]}: `{f}` was not saved by the {R.dl_subrun[c]} run; read from the base run instead."
                       for c, f in sorted(R.dl_fallbacks)] if R.dl_fallbacks else []),
             "",
             "## Primary reporting models (by validation ROC-AUC)", ""]
    for cohort, cname in COHORTS:
        p = prims[cohort]
        sv = ", ".join(f"{PIPELINES[k]['label']} {R.final_val_auc(k, cohort):.4f}" for k in SINGLE_CANDIDATES)
        nn = ", ".join(f"{mlabel(k)} {v:.4f}" for k, v in R.nn_val(cohort).items())
        lines += [f"- **{cname}** (lead framework: {PIPELINES[p['single']]['framework']})",
                  f"  - Lead single model: {R.prim_label(cohort, p, 'single')} (single-model validation ROC-AUC: {sv})",
                  f"  - Soft-voting ensemble of the same framework: {R.prim_label(cohort, p, 'ensemble')} "
                  f"(validation ROC-AUC {R.final_val_auc(p['ensemble'], cohort):.4f})",
                  f"  - Top neural network: {R.prim_label(cohort, p, 'nn')} (validation ROC-AUC: {nn}; the ensemble value "
                  "is recorded after isotonic calibration on validation)"]
    lines += ["", "## Outputs a pipeline did not save", "",
              "- Row-level predicted probabilities (`posthoc/scored_*.csv`) exist only for the Python single-model and "
              "Python soft-voting pipelines.",
              "- Locked test-set metrics for the single FT-Transformer and FT-Transformer-large were not saved by the deep "
              "learning pipeline; only its top-2 ensemble appears in test-set outputs.",
              "- SHAP values exist only for the Python single-model and soft-voting pipelines."]
    for cname, plabel, shap_mem, left_out in R.shap_partial:
        lines.append(f"- {cname}, {plabel}: SHAP (`12_shap_mean_abs.csv`) averages "
                     f"{join_and(mlabel(m) for m in shap_mem)} only (per `12_shap_meta.json`); "
                     f"{join_and(mlabel(m) for m in left_out)} has no saved SHAP values. Not recomputed.")
    lines += ["- Sensitivity and specificity point estimates are blank/NA in the Python and R bootstrap files; those rows "
              "are omitted from Table 5 (not recomputed).",
              "- Excluded exploratory runs: 03_*/adaboost_probe, 03_*/smoke_adaboost, 04_*/adaboost_probe, "
              "07_*/ny/final_all_features, and any 07_*/<cohort>/ run not listed under Deep learning run used per cohort.",
              "", "## Values computed by this script (approved)", "",
              "- Unweighted stroke prevalence, Wilson 95% CIs and Pearson chi-square tests (S11-S13), in "
              "`tables/tableS_prevalence_by_group.csv`.",
              "- ROC coordinates, confusion-matrix counts (from saved `stroke_pred`), decile calibration bins, score "
              "densities and group means. Printed AUC/Brier values are read from `05_test_metrics_bootstrap.csv`.", ""]
    if R.missing:
        lines += ["## Expected files not found", ""] + [f"- `{m}`" for m in R.missing] + [""]
    if GLYPH_WARNINGS:
        lines += ["## Font warnings", ""] + [f"- {g}" for g in sorted(set(GLYPH_WARNINGS))] + [""]
    if W.notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in W.notes] + [""]
    lines += ["## Input files read", ""] + [f"- `{u}`" for u in sorted(R.used)]
    (W.out / "README_provenance.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--only", default="", help="comma list, e.g. fig2,fig4,table1,supp (default: everything)")
    ap.add_argument("--no-clean", action="store_true", help="do not delete the previous output folder")
    ap.add_argument("--ny-dl-run", default=DL_SUBRUN.get("ny", ""),
                    help="subfolder of 07_python_ml_deep_learning/ny/ to report (default: final_lasso_features; "
                         "pass \"\" for the base ny run)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    setup_font()

    R = Results(args.results_dir.expanduser().resolve(), args.data_dir.expanduser().resolve(),
                dl_subrun={**DL_SUBRUN, "ny": args.ny_dl_run})
    log.info("deep learning runs: %s", {c: R.dl_subrun.get(c, "(base)") for c, _ in COHORTS})
    W = Writer(args.out_dir.expanduser().resolve())
    only = {s.strip().lower() for s in args.only.split(",") if s.strip()}
    want = lambda k: not only or k.lower() in only  # noqa: E731
    W.prepare(clean=not args.no_clean and not only)

    prims = {cohort: R.primaries(cohort) for cohort, _ in COHORTS}
    log.info("primary models: %s", prims)

    steps = [("table1", lambda: table1(R, W)), ("table2", lambda: table2(R, W, prims)),
             ("table3", lambda: table3(R, W, prims)), ("table4", lambda: table4(R, W, prims)),
             ("table5", lambda: table5(R, W, prims)), ("table6", lambda: table6(R, W)),
             ("tabler", lambda: table_r_lasso(R, W)), ("tabler", lambda: table_r_classifiers(R, W)),
             ("tabler", lambda: table_r_test(R, W))]
    for n, fn in [(2, fig2), (3, fig3), (4, fig4), (5, fig5), (6, fig6), (7, fig7), (8, fig8), (9, fig9)]:
        for cohort, _ in COHORTS:
            steps.append((f"fig{n}", lambda fn=fn, cohort=cohort: fn(R, W, cohort, prims[cohort])))
    steps += [("supp", lambda: figS10(R, W)), ("supp", lambda: figS14_S15(R, W))]
    for key, fn in steps:
        if want(key):
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                log.exception("%s failed: %s", key, e)
                W.notes.append(f"{key} failed: {e}")
    if want("supp"):
        stats = prevalence_stats(R)
        stats.to_csv(W.tab_dir / "tableS_prevalence_by_group.csv", index=False)
        figS11_S13(R, W, stats)
    if W.tables:
        W.write_docx()
    write_readme(R, W, prims)
    if GLYPH_WARNINGS:
        log.warning("font warnings: %s", sorted(set(GLYPH_WARNINGS)))
    log.info("done -> %s", W.out)


if __name__ == "__main__":
    main()
