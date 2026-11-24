#!/usr/bin/env python3
"""
BRFSS stroke ML pipeline 7 — FINAL tabular deep learning (GPU-first).

Designed as the strongest practical DL pipeline for this project's BRFSS data:
  - Mixed categorical embeddings + continuous features (most BRFSS vars are ordinal)
  - Models: FT-Transformer (primary), Tabular ResNet, RealMLP-style net
  - Optional soft ensemble of top-2 by validation ROC-AUC
  - Isotonic calibration on validation
  - Uses ALL features by default (LASSO optional via --lasso)
  - Auto CUDA when available (RTX / CUDA builds)

Same evaluation contract as pipelines 1–2:
  temporal split → batch correction → val threshold (Youden) → test bootstrap.

Outputs: results/ml_deep_learning/<dataset>/  (= 05_results/07_python_ml_deep_learning/)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
import joblib

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Pipeline 7 requires PyTorch (CUDA recommended).\n"
        "  pip install -r 01_installation/requirements-dl.txt"
    ) from exc


def _load_base_module():
    here = Path(__file__).resolve()
    candidates = [
        Path.cwd() / "scripts" / "run_ml_pipeline.py",
        *(base / "scripts" / "run_ml_pipeline.py" for base in (here, *here.parents)),
        here.parents[1] / "shared_scripts" / "run_ml_pipeline.py",
        here.parents[1] / "01_python_standard" / "run_ml_pipeline.py",
    ]
    for path in candidates:
        path = path.resolve()
        if path.is_file() and path != here:
            spec = importlib.util.spec_from_file_location("_stroke_ml_base", path)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit(
        "Could not locate scripts/run_ml_pipeline.py (run from project/ after bootstrap)."
    )


base = _load_base_module()
RESULTS_SUBDIR = "ml_deep_learning"

try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass

# Continuous-forced columns (high cardinality / truly numeric BRFSS measures)
FORCE_CONTINUOUS = {
    "bmi",
    "mental_health_days",
    "physical_health_days",
    "poor_health_days",
    "alcohol_days",
    "avg_drinks",
    "binge_drinking",
    "children",
    "kidney_disease",
}

# Final architecture catalog
# Milder LR + stronger dropout to avoid early val collapse on BRFSS
ARCH_CATALOG: dict[str, dict[str, Any]] = {
    "ft_transformer": {
        "model_type": "ft_transformer",
        "d_token": 96,
        "n_blocks": 4,
        "n_heads": 8,
        "d_ffn": 192,
        "dropout": 0.15,
        "lr": 2e-4,
    },
    "ft_transformer_large": {
        "model_type": "ft_transformer",
        "d_token": 128,
        "n_blocks": 5,
        "n_heads": 8,
        "d_ffn": 256,
        "dropout": 0.12,
        "lr": 1.5e-4,
    },
    "tabular_resnet": {
        "model_type": "resnet",
        "d_hidden": 256,
        "n_blocks": 5,
        "dropout": 0.20,
        "lr": 5e-4,
    },
    "realmlp": {
        "model_type": "realmlp",
        "d_hidden": 256,
        "n_blocks": 4,
        "dropout": 0.20,
        "lr": 7e-4,
    },
}


# ---------------------------------------------------------------------------
# Feature schema + encoding (categorical embeddings)
# ---------------------------------------------------------------------------


def infer_feature_schema(
    feature_names: list[str],
    X: np.ndarray,
    *,
    max_cat_cardinality: int = 16,
) -> dict[str, Any]:
    """Split features into categorical vs continuous using train cardinalities."""
    cat_idx: list[int] = []
    cont_idx: list[int] = []
    cat_names: list[str] = []
    cont_names: list[str] = []
    for j, name in enumerate(feature_names):
        n_unique = int(np.unique(X[:, j][~np.isnan(X[:, j])]).size) if X.shape[0] else 0
        if name in FORCE_CONTINUOUS or n_unique > max_cat_cardinality:
            cont_idx.append(j)
            cont_names.append(name)
        else:
            cat_idx.append(j)
            cat_names.append(name)
    return {
        "feature_names": feature_names,
        "cat_idx": cat_idx,
        "cont_idx": cont_idx,
        "cat_names": cat_names,
        "cont_names": cont_names,
    }


class MixedFeatureEncoder:
    """
    Train-fit encoder:
      - categorical → integer codes [0=unseen/missing, 1..K]
      - continuous → median-impute + StandardScaler
    """

    def __init__(self, schema: dict[str, Any]):
        self.schema = schema
        self.cat_maps_: list[dict[float, int]] = []
        self.cat_cardinalities_: list[int] = []  # including 0 = unseen
        self.cont_medians_: np.ndarray | None = None
        self.scaler_: StandardScaler | None = None

    @property
    def n_cat(self) -> int:
        return len(self.schema["cat_idx"])

    @property
    def n_cont(self) -> int:
        return len(self.schema["cont_idx"])

    def fit(self, X: np.ndarray) -> "MixedFeatureEncoder":
        self.cat_maps_ = []
        self.cat_cardinalities_ = []
        for j in self.schema["cat_idx"]:
            col = X[:, j]
            vals = col[~np.isnan(col)]
            uniq = np.unique(vals)
            # reserve 0 for unseen/NaN
            mapping = {float(v): i + 1 for i, v in enumerate(uniq)}
            self.cat_maps_.append(mapping)
            self.cat_cardinalities_.append(len(mapping) + 1)

        if self.n_cont:
            cont = X[:, self.schema["cont_idx"]].astype(np.float64)
            self.cont_medians_ = np.nanmedian(cont, axis=0)
            cont_imp = cont.copy()
            inds = np.where(np.isnan(cont_imp))
            cont_imp[inds] = np.take(self.cont_medians_, inds[1])
            self.scaler_ = StandardScaler()
            self.scaler_.fit(cont_imp)
        else:
            self.cont_medians_ = np.array([])
            self.scaler_ = None
        return self

    def transform(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = X.shape[0]
        if self.n_cat:
            x_cat = np.zeros((n, self.n_cat), dtype=np.int64)
            for i, j in enumerate(self.schema["cat_idx"]):
                col = X[:, j]
                mp = self.cat_maps_[i]
                codes = np.zeros(n, dtype=np.int64)
                for r in range(n):
                    v = col[r]
                    if np.isnan(v):
                        codes[r] = 0
                    else:
                        codes[r] = mp.get(float(v), 0)
                x_cat[:, i] = codes
        else:
            x_cat = np.zeros((n, 0), dtype=np.int64)

        if self.n_cont:
            cont = X[:, self.schema["cont_idx"]].astype(np.float64)
            inds = np.where(np.isnan(cont))
            cont[inds] = np.take(self.cont_medians_, inds[1])
            assert self.scaler_ is not None
            x_cont = self.scaler_.transform(cont).astype(np.float32)
        else:
            x_cont = np.zeros((n, 0), dtype=np.float32)
        return x_cat, x_cont


class TabularTorchDataset(Dataset):
    def __init__(self, x_cat: np.ndarray, x_cont: np.ndarray, y: np.ndarray | None = None):
        self.x_cat = torch.from_numpy(x_cat.astype(np.int64))
        self.x_cont = torch.from_numpy(x_cont.astype(np.float32))
        self.y = None if y is None else torch.from_numpy(np.asarray(y, dtype=np.float32))

    def __len__(self) -> int:
        return int(self.x_cat.shape[0] if self.x_cat.ndim else 0) or int(self.x_cont.shape[0])

    def __getitem__(self, idx: int):
        if self.y is None:
            return self.x_cat[idx], self.x_cont[idx]
        return self.x_cat[idx], self.x_cont[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Models with categorical embeddings
# ---------------------------------------------------------------------------


class _FeatureTokenizerMixed(nn.Module):
    """One token per feature: Embedding for cats, Linear(1→d) for continuous."""

    def __init__(self, cat_cards: list[int], n_cont: int, d_token: int):
        super().__init__()
        self.cat_embs = nn.ModuleList(
            [nn.Embedding(card, d_token) for card in cat_cards]
        )
        self.cont_weight = nn.Parameter(torch.empty(n_cont, d_token)) if n_cont else None
        self.cont_bias = nn.Parameter(torch.empty(n_cont, d_token)) if n_cont else None
        if self.cont_weight is not None:
            nn.init.kaiming_uniform_(self.cont_weight, a=math.sqrt(5))
            nn.init.zeros_(self.cont_bias)
        for emb in self.cat_embs:
            nn.init.normal_(emb.weight, std=0.02)
            # unseen/pad index 0 near zero
            with torch.no_grad():
                emb.weight[0].fill_(0.0)

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        tokens = []
        for i, emb in enumerate(self.cat_embs):
            tokens.append(emb(x_cat[:, i]).unsqueeze(1))
        if self.cont_weight is not None and x_cont.shape[1] > 0:
            # [B, C, D] = bias + weight * x
            cont_tok = self.cont_bias.unsqueeze(0) + self.cont_weight.unsqueeze(0) * x_cont.unsqueeze(-1)
            tokens.append(cont_tok)
        if not tokens:
            raise RuntimeError("No features to tokenize")
        return torch.cat(tokens, dim=1)  # [B, F, D]


class FTTransformerMixed(nn.Module):
    def __init__(
        self,
        cat_cards: list[int],
        n_cont: int,
        d_token: int = 96,
        n_blocks: int = 4,
        n_heads: int = 8,
        d_ffn: int = 192,
        dropout: float = 0.12,
    ):
        super().__init__()
        if d_token % n_heads != 0:
            raise ValueError(f"d_token={d_token} not divisible by n_heads={n_heads}")
        self.tokenizer = _FeatureTokenizerMixed(cat_cards, n_cont, d_token)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="enable_nested_tensor is True", category=UserWarning)
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_blocks)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.ReLU(inplace=True), nn.Linear(d_token, 1))

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        tok = self.tokenizer(x_cat, x_cont)
        cls = self.cls.expand(tok.size(0), -1, -1)
        h = self.encoder(torch.cat([cls, tok], dim=1))
        return self.head(h[:, 0, :]).squeeze(-1)


class _CatContEmbed(nn.Module):
    """Flattened categorical embeddings + continuous → vector."""

    def __init__(self, cat_cards: list[int], n_cont: int, d_emb: int = 16):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(c, d_emb) for c in cat_cards])
        for emb in self.embs:
            nn.init.normal_(emb.weight, std=0.02)
            with torch.no_grad():
                emb.weight[0].fill_(0.0)
        self.n_cont = n_cont
        self.out_dim = d_emb * len(cat_cards) + n_cont

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        parts = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        if self.n_cont:
            parts.append(x_cont)
        return torch.cat(parts, dim=1) if parts else x_cont


class _ResBlock(nn.Module):
    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.norm = nn.BatchNorm1d(d)
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc1(self.norm(x)), inplace=True)
        h = self.drop(h)
        h = self.drop(self.fc2(h))
        return x + h


class TabularResNetMixed(nn.Module):
    def __init__(
        self,
        cat_cards: list[int],
        n_cont: int,
        d_hidden: int = 256,
        n_blocks: int = 5,
        dropout: float = 0.15,
        d_emb: int = 16,
    ):
        super().__init__()
        self.stem = _CatContEmbed(cat_cards, n_cont, d_emb=d_emb)
        self.first = nn.Linear(self.stem.out_dim, d_hidden)
        self.blocks = nn.ModuleList([_ResBlock(d_hidden, dropout) for _ in range(n_blocks)])
        self.head_norm = nn.BatchNorm1d(d_hidden)
        self.head = nn.Linear(d_hidden, 1)

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.first(self.stem(x_cat, x_cont)), inplace=True)
        for b in self.blocks:
            x = b(x)
        return self.head(F.relu(self.head_norm(x), inplace=True)).squeeze(-1)


class RealMLPMixed(nn.Module):
    """Modern residual MLP with SELU-ish style blocks (compact RealMLP spirit)."""

    def __init__(
        self,
        cat_cards: list[int],
        n_cont: int,
        d_hidden: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.15,
        d_emb: int = 16,
    ):
        super().__init__()
        self.stem = _CatContEmbed(cat_cards, n_cont, d_emb=d_emb)
        self.first = nn.Linear(self.stem.out_dim, d_hidden)
        layers = []
        for _ in range(n_blocks):
            layers.extend(
                [
                    nn.BatchNorm1d(d_hidden),
                    nn.Linear(d_hidden, d_hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(d_hidden, d_hidden),
                    nn.Dropout(dropout),
                ]
            )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.BatchNorm1d(d_hidden),
                    nn.Linear(d_hidden, d_hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(d_hidden, d_hidden),
                    nn.Dropout(dropout),
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = nn.Sequential(nn.BatchNorm1d(d_hidden), nn.ReLU(inplace=True), nn.Linear(d_hidden, 1))

    def forward(self, x_cat: torch.Tensor, x_cont: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.first(self.stem(x_cat, x_cont)), inplace=True)
        for blk in self.blocks:
            x = x + blk(x)
        return self.head(x).squeeze(-1)


def build_model(
    model_type: str,
    cat_cards: list[int],
    n_cont: int,
    cfg: dict[str, Any],
) -> nn.Module:
    if model_type == "ft_transformer":
        return FTTransformerMixed(
            cat_cards,
            n_cont,
            d_token=int(cfg.get("d_token", 96)),
            n_blocks=int(cfg.get("n_blocks", 4)),
            n_heads=int(cfg.get("n_heads", 8)),
            d_ffn=int(cfg.get("d_ffn", 192)),
            dropout=float(cfg.get("dropout", 0.12)),
        )
    if model_type == "resnet":
        return TabularResNetMixed(
            cat_cards,
            n_cont,
            d_hidden=int(cfg.get("d_hidden", 256)),
            n_blocks=int(cfg.get("n_blocks", 5)),
            dropout=float(cfg.get("dropout", 0.15)),
        )
    if model_type == "realmlp":
        return RealMLPMixed(
            cat_cards,
            n_cont,
            d_hidden=int(cfg.get("d_hidden", 256)),
            n_blocks=int(cfg.get("n_blocks", 4)),
            dropout=float(cfg.get("dropout", 0.15)),
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def _arch_summary(name: str, cfg: dict[str, Any]) -> str:
    mt = cfg["model_type"]
    if mt == "ft_transformer":
        return f"ft_transformer d={cfg.get('d_token')} blocks={cfg.get('n_blocks')} heads={cfg.get('n_heads')} +cat_emb"
    if mt == "resnet":
        return f"resnet d={cfg.get('d_hidden')} blocks={cfg.get('n_blocks')} +cat_emb"
    if mt == "realmlp":
        return f"realmlp d={cfg.get('d_hidden')} blocks={cfg.get('n_blocks')} +cat_emb"
    return name


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class TorchMixedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        model_type: str = "ft_transformer",
        model_cfg: dict[str, Any] | None = None,
        cat_cards: list[int] | None = None,
        n_cont: int = 0,
        max_epochs: int = 150,
        patience: int = 20,
        batch_size: int = 1024,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        device: str = "cuda",
        seed: int = 42,
        verbose: bool = True,
        mixup_alpha: float = 0.0,
    ):
        self.model_type = model_type
        self.model_cfg = model_cfg or {}
        self.cat_cards = cat_cards or []
        self.n_cont = n_cont
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.seed = seed
        self.verbose = verbose
        self.mixup_alpha = mixup_alpha
        self.classes_ = np.array([0, 1])
        self.model_: nn.Module | None = None
        self.best_epoch_: int = 0
        self.history_: list[dict] = []
        self.calibrator_: IsotonicRegression | None = None

    def _set_seed(self) -> None:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def fit(
        self,
        x_cat: np.ndarray,
        x_cont: np.ndarray,
        y: np.ndarray,
        x_cat_val: np.ndarray | None = None,
        x_cont_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        calibrate: bool = True,
    ):
        self._set_seed()
        device = torch.device(self.device)
        ys = np.asarray(y, dtype=np.float32)
        n_pos = max(float(ys.sum()), 1.0)
        n_neg = max(float(len(ys) - ys.sum()), 1.0)
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)

        self.model_ = build_model(
            self.model_type, list(self.cat_cards), int(self.n_cont), dict(self.model_cfg)
        ).to(device)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(self.max_epochs, 1), eta_min=self.lr * 0.05
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        ds = TabularTorchDataset(x_cat, x_cont, ys)
        loader = DataLoader(
            ds,
            batch_size=min(self.batch_size, len(ds)),
            shuffle=True,
            drop_last=False,
            pin_memory=(device.type == "cuda"),
            num_workers=0,
        )

        has_val = y_val is not None and x_cat_val is not None and x_cont_val is not None
        best_state = None
        best_auc = -1.0
        bad = 0
        self.history_ = []

        for epoch in range(1, self.max_epochs + 1):
            self.model_.train()
            total_loss = 0.0
            n_seen = 0
            for batch in loader:
                xb_cat, xb_cont, yb = batch
                xb_cat = xb_cat.to(device, non_blocking=True)
                xb_cont = xb_cont.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                if self.mixup_alpha > 0 and xb_cat.size(0) > 1:
                    lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                    idx = torch.randperm(xb_cat.size(0), device=device)
                    # mix continuous + labels; cats take from one sample (approx)
                    xb_cont = lam * xb_cont + (1 - lam) * xb_cont[idx]
                    yb = lam * yb + (1 - lam) * yb[idx]
                    if lam < 0.5:
                        xb_cat = xb_cat[idx]

                opt.zero_grad(set_to_none=True)
                logits = self.model_(xb_cat, xb_cont)
                loss = loss_fn(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                opt.step()
                total_loss += float(loss.item()) * len(yb)
                n_seen += len(yb)
            sched.step()
            train_loss = total_loss / max(n_seen, 1)

            val_auc = float("nan")
            if has_val:
                probs = self._predict_proba_raw(x_cat_val, x_cont_val)
                try:
                    val_auc = float(roc_auc_score(y_val, probs[:, 1]))
                except ValueError:
                    val_auc = float("nan")
                score = val_auc if np.isfinite(val_auc) else -1.0
                if score > best_auc + 1e-4:
                    best_auc = score
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                    self.best_epoch_ = epoch
                    bad = 0
                else:
                    bad += 1
            else:
                best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                self.best_epoch_ = epoch

            self.history_.append({"epoch": epoch, "train_loss": train_loss, "val_roc_auc": val_auc})
            if self.verbose and (epoch == 1 or epoch % 5 == 0 or bad == 0):
                msg = f"  epoch {epoch:03d} loss={train_loss:.4f}"
                if has_val:
                    msg += f" val_auc={val_auc:.4f}"
                print(msg, flush=True)

            if has_val and bad >= self.patience:
                if self.verbose:
                    print(
                        f"  early stop at epoch {epoch} (best={self.best_epoch_}, auc={best_auc:.4f})",
                        flush=True,
                    )
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.model_.eval()

        self.calibrator_ = None
        if calibrate and has_val:
            raw = self._predict_proba_raw(x_cat_val, x_cont_val)[:, 1]
            self.calibrator_ = IsotonicRegression(out_of_bounds="clip")
            self.calibrator_.fit(raw, y_val)
        return self

    def _predict_proba_raw(self, x_cat: np.ndarray, x_cont: np.ndarray) -> np.ndarray:
        assert self.model_ is not None
        device = torch.device(self.device)
        self.model_.eval()
        ds = TabularTorchDataset(x_cat, x_cont, None)
        loader = DataLoader(
            ds,
            batch_size=min(self.batch_size * 2, max(len(ds), 1)),
            shuffle=False,
            pin_memory=(device.type == "cuda"),
        )
        outs = []
        with torch.no_grad():
            for xb_cat, xb_cont in loader:
                xb_cat = xb_cat.to(device, non_blocking=True)
                xb_cont = xb_cont.to(device, non_blocking=True)
                logits = self.model_(xb_cat, xb_cont)
                outs.append(torch.sigmoid(logits).cpu().numpy())
        p1 = np.concatenate(outs) if outs else np.array([])
        return np.column_stack([1.0 - p1, p1])

    def predict_proba(self, x_cat: np.ndarray, x_cont: np.ndarray) -> np.ndarray:
        raw = self._predict_proba_raw(x_cat, x_cont)
        if self.calibrator_ is not None:
            p1 = self.calibrator_.predict(raw[:, 1])
            p1 = np.clip(p1, 1e-6, 1 - 1e-6)
            return np.column_stack([1.0 - p1, p1])
        return raw

    # sklearn API stubs (permutation importance needs X 2D — use packed matrix helper)
    def predict(self, X_packed: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use predict_proba(x_cat, x_cont)")


class EnsembleProba:
    """Average probabilities from multiple TorchMixedClassifier models."""

    def __init__(self, members: list[TorchMixedClassifier], weights: list[float] | None = None):
        self.members = members
        self.weights = weights or [1.0] * len(members)

    def predict_proba(self, x_cat: np.ndarray, x_cont: np.ndarray) -> np.ndarray:
        probs = []
        wsum = 0.0
        for m, w in zip(self.members, self.weights):
            probs.append(w * m.predict_proba(x_cat, x_cont))
            wsum += w
        return sum(probs) / max(wsum, 1e-8)


def _resolve_device(use_gpu: bool) -> str:
    if use_gpu and torch.cuda.is_available():
        return "cuda"
    if use_gpu and not torch.cuda.is_available():
        print("WARNING: --use-gpu requested but CUDA not available; using CPU.", flush=True)
    return "cpu"


def _make_clf(
    name: str,
    *,
    cat_cards: list[int],
    n_cont: int,
    device: str,
    quick: bool,
    seed: int = 42,
    verbose: bool = True,
) -> TorchMixedClassifier:
    cfg = dict(ARCH_CATALOG[name])
    model_type = cfg.pop("model_type")
    lr = float(cfg.pop("lr", 3e-4))
    if quick:
        max_epochs, patience = 40, 10
        batch_size = 512 if device == "cuda" else 256
        mixup = 0.0
    else:
        max_epochs, patience = 150, 20
        batch_size = 2048 if device == "cuda" else 512
        mixup = 0.1 if model_type != "ft_transformer" else 0.0
    if model_type == "ft_transformer" and device == "cpu":
        batch_size = min(batch_size, 256)
    # Shrink batches when GPU VRAM is nearly full
    if device == "cuda" and torch.cuda.is_available():
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        if free_gb < 4:
            batch_size = min(batch_size, 256 if model_type == "ft_transformer" else 512)
        elif free_gb < 8:
            batch_size = min(batch_size, 512 if model_type == "ft_transformer" else 1024)
    return TorchMixedClassifier(
        model_type=model_type,
        model_cfg=cfg,
        cat_cards=cat_cards,
        n_cont=n_cont,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        lr=lr,
        weight_decay=1e-4,
        device=device,
        seed=seed,
        verbose=verbose,
        mixup_alpha=mixup,
    )


def run_pipeline(
    dataset: str,
    quick: bool = False,
    n_jobs: int = -1,
    use_gpu: bool = True,
    use_lasso: bool = True,
    ensemble_top_k: int = 2,
) -> None:
    base.init_parallel(n_jobs)
    device = _resolve_device(use_gpu)
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

    prefix = "brfss_stroke_new_york" if dataset == "ny" else "brfss_stroke_all_states"
    out_dir = base.pipeline_results_dir(RESULTS_SUBDIR, dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    if quick:
        arch_names = ["ft_transformer", "tabular_resnet", "realmlp"]
    else:
        arch_names = list(ARCH_CATALOG)

    print(f"=== ML pipeline 7 FINAL (deep_learning): {dataset} ===", flush=True)
    print(f"Outputs: {out_dir}", flush=True)
    print(f"Device: {device} (torch {torch.__version__})", flush=True)
    print(f"Features: {'LASSO-selected (recommended)' if use_lasso else 'ALL features'}", flush=True)
    print(f"Architectures: {', '.join(arch_names)}", flush=True)
    for n in arch_names:
        print(f"  - {n}: {_arch_summary(n, ARCH_CATALOG[n])}", flush=True)

    train, val, test = base.load_splits(prefix)
    feature_cols = base.get_feature_columns(train)
    X_fit_raw, y_fit, _ = base.prepare_xy(train, feature_cols, include_batch=False)
    X_val_raw, y_val, _ = base.prepare_xy(val, feature_cols, include_batch=False)
    X_te, y_te, _ = base.prepare_xy(test, feature_cols, include_batch=False)
    batches_fit = train[base.BATCH_COL].values
    batches_val = val[base.BATCH_COL].values
    batches_te = test[base.BATCH_COL].values

    # PCA diagnostics on full matrix (unchanged visuals)
    imp_final, gm_final, ym_final = base.fit_batch_correction(X_fit_raw, batches_fit)
    X_fit_bc_all = base.apply_batch_correction(
        X_fit_raw, batches_fit, imp_final, gm_final, ym_final
    )
    base.plot_batch_pca(
        X_fit_raw, y_fit, batches_fit, out_dir / "01_batch_pca_before.png",
        "PCA before batch correction (train, by survey_year)",
    )
    base.plot_batch_pca(
        X_fit_bc_all, y_fit, batches_fit, out_dir / "01_batch_pca_after.png",
        "PCA after year-mean correction (train-fitted)",
    )

    n_boot = 200 if quick else 1000
    selected = feature_cols
    lasso_summary = None
    if use_lasso:
        outer_lasso = 3 if quick else 5
        inner_lasso = 2 if quick else 3
        C_grid = [0.01, 0.05, 0.1, 0.5] if quick else [0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
        print("Nested CV LASSO feature selection (on batch-corrected matrix)...", flush=True)
        lasso_summary = base.lasso_nested_cv(
            X_fit_bc_all, y_fit, batches_fit, feature_cols, out_dir,
            C_grid=C_grid, outer_splits=outer_lasso, inner_splits=inner_lasso,
            max_features=30, stability_threshold=0.5,
        )
        selected = lasso_summary["stable_features"]
        print(
            f"LASSO: {len(selected)} features, outer AUC="
            f"{lasso_summary['outer_roc_auc_mean']:.4f}",
            flush=True,
        )
    else:
        (out_dir / "02_lasso_selected_features.txt").write_text(
            "\n".join(selected)
            + "\n\n# ALL features used; categoricals encoded BEFORE batch correction\n"
        )
        pd.DataFrame({"feature": selected}).to_csv(out_dir / "02_lasso_coefficients.csv", index=False)
        print(f"Using ALL {len(selected)} features (no LASSO).", flush=True)

    sel_idx = [feature_cols.index(f) for f in selected]
    X_fit_sel_raw = X_fit_raw[:, sel_idx]
    X_val_sel_raw = X_val_raw[:, sel_idx]
    X_test_sel_raw = X_te[:, sel_idx]

    # CRITICAL: infer schema + encode categoricals on RAW values (not batch-corrected),
    # then batch-correct ONLY continuous columns. Otherwise year-mean correction
    # turns binary/ordinal codes into many floats and breaks embeddings.
    schema = infer_feature_schema(selected, X_fit_sel_raw)
    enc = MixedFeatureEncoder(schema).fit(X_fit_sel_raw)
    xtr_cat, xtr_cont_raw = enc.transform(X_fit_sel_raw)
    xva_cat, xva_cont_raw = enc.transform(X_val_sel_raw)
    xte_cat, xte_cont_raw = enc.transform(X_test_sel_raw)

    if enc.n_cont:
        cont_idx = schema["cont_idx"]
        # Batch-correct continuous columns only (train-fit)
        X_fit_cont = X_fit_sel_raw[:, cont_idx]
        X_val_cont = X_val_sel_raw[:, cont_idx]
        X_te_cont = X_test_sel_raw[:, cont_idx]
        imp_c, gm_c, ym_c = base.fit_batch_correction(X_fit_cont, batches_fit)
        X_fit_cont_bc = base.apply_batch_correction(X_fit_cont, batches_fit, imp_c, gm_c, ym_c)
        X_val_cont_bc = base.apply_batch_correction(X_val_cont, batches_val, imp_c, gm_c, ym_c)
        X_te_cont_bc = base.apply_batch_correction(X_te_cont, batches_te, imp_c, gm_c, ym_c)
        # Re-scale continuous after batch correction
        scaler = StandardScaler()
        xtr_cont = scaler.fit_transform(X_fit_cont_bc).astype(np.float32)
        xva_cont = scaler.transform(X_val_cont_bc).astype(np.float32)
        xte_cont = scaler.transform(X_te_cont_bc).astype(np.float32)
        enc.scaler_ = scaler
        enc.cont_batch_correction_ = {"imputer": imp_c, "global_mean": gm_c, "year_means": ym_c}
    else:
        xtr_cont = xtr_cont_raw
        xva_cont = xva_cont_raw
        xte_cont = xte_cont_raw
        enc.cont_batch_correction_ = None

    print(
        f"Schema: {enc.n_cat} categorical embeddings, {enc.n_cont} continuous "
        f"(cards sample={enc.cat_cardinalities_[:8]})",
        flush=True,
    )
    (out_dir / "02f_feature_schema.json").write_text(
        json.dumps(
            {
                "n_cat": enc.n_cat,
                "n_cont": enc.n_cont,
                "cat_names": schema["cat_names"],
                "cont_names": schema["cont_names"],
                "cat_cardinalities": enc.cat_cardinalities_,
                "batch_correction": "continuous_only",
                "categorical_encoding": "raw_before_batch_correction",
            },
            indent=2,
        )
        + "\n"
    )

    # --- Validation selection ---
    val_rows = []
    fitted: dict[str, TorchMixedClassifier] = {}
    print("Training / validation scoring...", flush=True)
    for name in arch_names:
        clf = _make_clf(
            name,
            cat_cards=enc.cat_cardinalities_,
            n_cont=enc.n_cont,
            device=device,
            quick=quick,
            seed=42,
            verbose=True,
        )
        print(f"[val fit] {name}", flush=True)
        clf.fit(xtr_cat, xtr_cont, y_fit, xva_cat, xva_cont, y_val, calibrate=True)
        fitted[name] = clf
        val_auc = float(roc_auc_score(y_val, clf.predict_proba(xva_cat, xva_cont)[:, 1]))
        val_rows.append(
            {
                "model": name,
                "summary": _arch_summary(name, ARCH_CATALOG[name]),
                "val_roc_auc": val_auc,
                "best_epoch": clf.best_epoch_,
            }
        )
        pd.DataFrame(clf.history_).to_csv(out_dir / f"03d_train_history_{name}.csv", index=False)
        print(f"[val done] {name} AUC={val_auc:.4f}", flush=True)

    val_df = pd.DataFrame(val_rows).sort_values("val_roc_auc", ascending=False)
    val_df.to_csv(out_dir / "03b_val_model_selection.csv", index=False)
    val_df.to_csv(out_dir / "03_cv_model_comparison.csv", index=False)  # val ranking as primary compare
    (out_dir / "03c_classifiers_evaluated.txt").write_text(
        f"n_architectures={len(arch_names)}\n" + "\n".join(arch_names) + "\n"
    )

    best_name = str(val_df.iloc[0]["model"])
    top_k = max(1, min(ensemble_top_k, len(val_df)))
    top_names = val_df.head(top_k)["model"].tolist()
    top_aucs = [float(val_df.loc[val_df["model"] == n, "val_roc_auc"].iloc[0]) for n in top_names]
    # weight by softmax of val auc
    w = np.exp(np.array(top_aucs) * 10)
    w = (w / w.sum()).tolist()

    if top_k == 1:
        final_est: Any = fitted[best_name]
        final_label = best_name
        y_prob_va = final_est.predict_proba(xva_cat, xva_cont)[:, 1]
        y_prob_te = final_est.predict_proba(xte_cat, xte_cont)[:, 1]
    else:
        final_est = EnsembleProba([fitted[n] for n in top_names], weights=w)
        final_label = "dl_ensemble_top" + str(top_k) + "(" + ",".join(top_names) + ")"
        y_prob_va = final_est.predict_proba(xva_cat, xva_cont)[:, 1]
        y_prob_te = final_est.predict_proba(xte_cat, xte_cont)[:, 1]
        # re-calibrate ensemble on val
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(y_prob_va, y_val)
        y_prob_va = np.clip(iso.predict(y_prob_va), 1e-6, 1 - 1e-6)
        y_prob_te = np.clip(iso.predict(y_prob_te), 1e-6, 1 - 1e-6)
        final_est.ensemble_calibrator_ = iso  # type: ignore[attr-defined]

    print(
        f"Best single: {best_name} ({val_df.iloc[0]['val_roc_auc']:.4f}) | "
        f"Final: {final_label}",
        flush=True,
    )
    pd.DataFrame(
        {
            "rank": range(1, len(top_names) + 1),
            "model": top_names,
            "val_roc_auc": top_aucs,
            "ensemble_weight": w if top_k > 1 else [1.0],
        }
    ).to_csv(out_dir / "04b_ensemble_members.csv", index=False)

    best_threshold, threshold_info = base.choose_classification_threshold(y_val, y_prob_va)
    y_pred_te = (y_prob_te >= best_threshold).astype(int)
    cm = confusion_matrix(y_te, y_pred_te)
    print(
        f"Threshold (val, Youden J): {best_threshold:.3f} | "
        f"test pred+={y_pred_te.mean():.3f} (prev={y_te.mean():.3f}) | "
        f"sens={cm[1, 1] / max(cm[1].sum(), 1):.3f} "
        f"spec={cm[0, 0] / max(cm[0].sum(), 1):.3f}",
        flush=True,
    )

    boot_df = base.bootstrap_metrics(y_te, y_prob_te, y_pred_te, n_boot=n_boot)
    for m, fn in [
        ("roc_auc", lambda: roc_auc_score(y_te, y_prob_te)),
        ("accuracy", lambda: accuracy_score(y_te, y_pred_te)),
        ("f1", lambda: f1_score(y_te, y_pred_te, zero_division=0)),
        ("brier", lambda: brier_score_loss(y_te, y_prob_te)),
        ("pr_auc", lambda: average_precision_score(y_te, y_prob_te)),
    ]:
        boot_df.loc[boot_df["metric"] == m, "point"] = fn()
    tn, fp, fn_, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    boot_df.loc[boot_df["metric"] == "sensitivity", "point"] = tp / (tp + fn_) if (tp + fn_) else 0.0
    boot_df.loc[boot_df["metric"] == "specificity", "point"] = tn / (tn + fp) if (tn + fp) else 0.0
    boot_df.to_csv(out_dir / "05_test_metrics_bootstrap.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(cm, annot=True, fmt="d", ax=ax[0], cmap="Blues")
    ax[0].set_title(f"Test CM — {final_label} (thr={best_threshold:.3f})")
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("Actual")
    fpr, tpr, _ = roc_curve(y_te, y_prob_te)
    ax[1].plot(fpr, tpr, label=f"AUC={roc_auc_score(y_te, y_prob_te):.3f}")
    ax[1].plot([0, 1], [0, 1], "k--")
    ax[1].set_title("Test ROC")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(out_dir / "06_confusion_matrix_test.png", dpi=150)
    plt.close(fig)

    prob_true, prob_pred = calibration_curve(y_te, y_prob_te, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(prob_pred, prob_true, "o-", label=final_label)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration (test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "07_calibration_test.png", dpi=150)
    plt.close(fig)

    years = test[base.BATCH_COL].values
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_df = pd.DataFrame({"year": years, "score": y_prob_te, "stroke": y_te})
    sns.violinplot(data=plot_df, x="year", y="score", hue="stroke", split=True, ax=ax, inner="quart")
    ax.set_title("Test predicted probability by survey_year")
    fig.tight_layout()
    fig.savefig(out_dir / "08_per_year_score_violin.png", dpi=150)
    plt.close(fig)

    # Permutation importance on best single model (shuffle one raw feature at a time)
    print("Permutation importance (validation, best single)...", flush=True)

    def _encode_raw(X_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c, _ = enc.transform(X_raw)
        if enc.n_cont:
            cont = X_raw[:, schema["cont_idx"]].astype(np.float64)
            bc = enc.cont_batch_correction_
            cont_bc = base.apply_batch_correction(
                cont, batches_val, bc["imputer"], bc["global_mean"], bc["year_means"]
            )
            k = enc.scaler_.transform(cont_bc).astype(np.float32)
        else:
            k = np.zeros((X_raw.shape[0], 0), dtype=np.float32)
        return c, k

    best_single = fitted[best_name]
    base_auc = float(roc_auc_score(y_val, best_single.predict_proba(xva_cat, xva_cont)[:, 1]))
    rng = np.random.default_rng(42)
    n_repeats = 5 if quick else 10
    imp_means = []
    imp_stds = []
    for j, fname in enumerate(selected):
        drops = []
        for _ in range(n_repeats):
            Xp = X_val_sel_raw.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            c, k = _encode_raw(Xp)
            auc = float(roc_auc_score(y_val, best_single.predict_proba(c, k)[:, 1]))
            drops.append(base_auc - auc)
        imp_means.append(float(np.mean(drops)))
        imp_stds.append(float(np.std(drops)))
    imp_df = pd.DataFrame(
        {"feature": selected, "importance_mean": imp_means, "importance_std": imp_stds}
    ).sort_values("importance_mean", ascending=False)
    imp_df.to_csv(out_dir / "09_permutation_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.22 * min(len(selected), 30))))
    top = imp_df.head(25).iloc[::-1]
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    ax.set_title(f"Permutation importance — {best_name}")
    fig.tight_layout()
    fig.savefig(out_dir / "09_permutation_importance.png", dpi=150)
    plt.close(fig)

    joblib.dump(
        {
            "final_label": final_label,
            "best_single": best_name,
            "top_names": top_names,
            "ensemble_weights": w,
            "estimators": {n: fitted[n] for n in top_names},
            "encoder": enc,
            "schema": schema,
            "selected_features": selected,
            "feature_cols_all": feature_cols,
            "batch_correction_full_pca": {
                "imputer": imp_final,
                "global_mean": gm_final,
                "year_means": ym_final,
            },
            "batch_correction_continuous": getattr(enc, "cont_batch_correction_", None),
            "threshold": best_threshold,
            "ensemble_calibrator": getattr(final_est, "ensemble_calibrator_", None),
            "version": "final_v2_catemb_raw_then_bc",
        },
        out_dir / "11_best_model.joblib",
    )
    # save best single torch weights
    torch.save(
        {
            "architecture": best_name,
            "arch_cfg": ARCH_CATALOG[best_name],
            "state_dict": fitted[best_name].model_.state_dict()
            if fitted[best_name].model_ is not None
            else None,
            "cat_cards": enc.cat_cardinalities_,
            "n_cont": enc.n_cont,
            "schema": schema,
        },
        out_dir / "11_best_model.pt",
    )

    test_auc = float(roc_auc_score(y_te, y_prob_te))
    val_auc_final = float(roc_auc_score(y_val, y_prob_va))
    config = {
        "pipeline": "07_python_deep_learning",
        "version": "final_v2_catemb_raw_then_bc",
        "dataset": dataset,
        "framework": "pytorch",
        "dl_family": "ft_transformer + tabular_resnet + realmlp (categorical embeddings)",
        "torch_version": torch.__version__,
        "device": device,
        "cuda_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "final_model": final_label,
        "best_single_model": best_name,
        "best_single_summary": _arch_summary(best_name, ARCH_CATALOG[best_name]),
        "architectures_evaluated": arch_names,
        "n_architectures": len(arch_names),
        "use_lasso": use_lasso,
        "n_features": len(selected),
        "n_categorical": enc.n_cat,
        "n_continuous": enc.n_cont,
        "selected_features": selected,
        "lasso_nested_cv": lasso_summary,
        "val_roc_auc": val_auc_final,
        "val_roc_auc_best_single": float(val_df.iloc[0]["val_roc_auc"]),
        "test_roc_auc": test_auc,
        "classification_threshold": best_threshold,
        "threshold_info": threshold_info,
        "n_train": int(len(y_fit)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_te)),
        "quick": quick,
        "ensemble_top_k": top_k,
        "best_epoch_best_single": int(fitted[best_name].best_epoch_),
        "isotonic_calibration": True,
    }
    (out_dir / "pipeline_config.json").write_text(json.dumps(config, indent=2) + "\n")

    report = [
        f"# Workflow report — Pipeline 7 FINAL ({dataset})",
        "",
        f"- Version: `final_v2_catemb_raw_then_bc`",
        f"- Framework: PyTorch {torch.__version__} on `{device}`"
        + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""),
        f"- Features: {len(selected)} ({'LASSO' if use_lasso else 'ALL'}) — "
        f"{enc.n_cat} categorical embeddings + {enc.n_cont} continuous",
        f"- Models: {', '.join(arch_names)}",
        f"- Best single: **{best_name}**",
        f"- Final predictor: **{final_label}**",
        f"- Val ROC-AUC: {val_auc_final:.4f}",
        f"- Test ROC-AUC: {test_auc:.4f}",
        f"- Threshold (Youden J): {best_threshold:.4f}",
        f"- Calibration: isotonic on validation",
        "",
        "This final DL pipeline uses categorical embeddings (critical for BRFSS ordinal codes),",
        "GPU training when available, FT-Transformer + ResNet + RealMLP, and a soft ensemble.",
        "",
    ]
    (out_dir / "WORKFLOW_REPORT.md").write_text("\n".join(report))
    print(f"Done. Outputs: {out_dir}", flush=True)
    print(f"Test ROC-AUC: {test_auc:.4f}", flush=True)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Pipeline 7 FINAL — best tabular DL (cat embeddings, GPU, ensemble)"
    )
    p.add_argument(
        "--dataset",
        choices=["ny", "all_states", "both"],
        default="both",
        help="Which cohort(s) to run (default: both = ny then all_states)",
    )
    p.add_argument("--quick", action="store_true")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument(
        "--use-gpu",
        action="store_true",
        default=True,
        help="Prefer CUDA when available (default). Falls back to CPU if no GPU.",
    )
    p.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    p.add_argument("--lasso", action="store_true", default=True, help="Use LASSO features (default; best on BRFSS)")
    p.add_argument("--all-features", action="store_true", help="Use ALL features (no LASSO)")
    p.add_argument("--ensemble-top-k", type=int, default=2, help="Soft-ensemble top-K by val AUC")
    args = p.parse_args(argv)

    use_gpu = not args.cpu
    datasets = ["ny", "all_states"] if args.dataset == "both" else [args.dataset]
    print(
        f"Pipeline 7 will run datasets: {', '.join(datasets)} | "
        f"device={'cuda (if available)' if use_gpu else 'cpu'}",
        flush=True,
    )
    for ds in datasets:
        run_pipeline(
            ds,
            quick=args.quick,
            n_jobs=args.n_jobs,
            use_gpu=use_gpu,
            use_lasso=(not args.all_features),
            ensemble_top_k=args.ensemble_top_k,
        )


if __name__ == "__main__":
    main()
