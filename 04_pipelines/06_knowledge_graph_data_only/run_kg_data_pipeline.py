#!/usr/bin/env python3
"""
Pipeline 6 — Knowledge graph from processed train data only.

Builds a heterogeneous graph using ONLY rows in 02_datasets/processed/*_train.csv.
Does NOT read LASSO lists, GLM CSVs, permutation importance, ontology JSON, or
any artifact from Python/R ML pipelines (05_results/01–05, posthoc/).

Edge layers (all computed fresh from train split):
  - marginal Pearson r(feature, Stroke)
  - univariate logistic(scd ~ feature)
  - partial r(feature, Stroke | other predictors)
  - partial r(feature, feature | others + Stroke)

Usage:
  python 04_pipelines/06_knowledge_graph_data_only/run_kg_data_pipeline.py --dataset both
  python 04_pipelines/06_knowledge_graph_data_only/run_kg_data_pipeline.py --dataset ny --quick
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PIPE_DIR = Path(__file__).resolve().parent
PKG = PIPE_DIR.parents[1]
PIPE5_DIR = PKG / "04_pipelines" / "05_knowledge_graph"
sys.path.insert(0, str(PIPE5_DIR))
sys.path.insert(0, str(PIPE_DIR))

from kg_core import (  # noqa: E402
    OUTCOME_NODE,
    build_graph,
    edges_to_dataframe,
    score_nodes,
    top_pathways,
)
from kg_data_core import (  # noqa: E402
    marginal_logistic_edges,
    partial_correlation_matrix,
    partial_correlation_to_outcome,
    point_biserial_edges,
    predictor_columns,
)
from kg_visualize import (  # noqa: E402
    export_graphml,
    plot_interactive_html,
    plot_publication_kg,
    plot_static_kg,
)

PROCESSED = PKG / "02_datasets" / "processed"
OUT_ROOT = PKG / "05_results" / "06_knowledge_graph_data_only"


def _dataset_prefix(dataset: str) -> str:
    return "brfss_stroke_new_york" if dataset == "ny" else "brfss_stroke_all_states"


def run_kg_data_only(dataset: str, *, quick: bool = False) -> None:
    prefix = _dataset_prefix(dataset)
    train_path = PROCESSED / f"{prefix}_train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing processed train split: {train_path}")

    train = pd.read_csv(train_path)
    features = predictor_columns(train)
    if not features:
        raise ValueError(f"No predictor columns found in {train_path}")

    min_r = 0.18 if quick else 0.10
    min_marginal = 0.08 if quick else 0.05
    min_log_or = 0.08 if quick else 0.05

    edge_records: list[dict] = []
    edge_records.extend(point_biserial_edges(train, features, min_abs_r=min_marginal))
    edge_records.extend(marginal_logistic_edges(train, features, min_abs_log_or=min_log_or))
    edge_records.extend(partial_correlation_to_outcome(train, features, min_abs_r=min_r))
    edge_records.extend(partial_correlation_matrix(train, features, min_abs_r=min_r))

    g = build_graph(edge_records, features)
    scores_df = score_nodes(g, features)
    pathways_df = top_pathways(g, features)
    edges_df = edges_to_dataframe(g)
    if edges_df.empty:
        edges_df = pd.DataFrame(columns=["source", "target", "weight", "sources", "relation"])

    out_dir = OUT_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_df.to_csv(out_dir / "01_kg_feature_pathway_scores.csv", index=False)
    edges_df.to_csv(out_dir / "02_kg_edges.csv", index=False)
    pathways_df.to_csv(out_dir / "03_kg_shortest_pathways_to_stroke.csv", index=False)
    export_graphml(g, out_dir / "04_kg_graph.graphml")

    score_map = dict(zip(scores_df["feature"], scores_df["pathway_score"]))
    cohort_label = "New York" if dataset == "ny" else "National"
    plot_publication_kg(
        g,
        score_map,
        out_dir / "05_kg_network",
        cohort=dataset,
        title=None,
    )
    plot_static_kg(
        g,
        score_map,
        out_dir / "05_kg_network_spring.png",
        title=f"Data-only KG pathways to stroke ({cohort_label}, train split)",
    )
    plot_interactive_html(
        g,
        score_map,
        out_dir / "06_kg_network_interactive.html",
        title=f"Data-only Stroke knowledge graph — {cohort_label}",
    )

    meta = {
        "pipeline": "06_knowledge_graph_data_only",
        "dataset": dataset,
        "n_predictors": len(features),
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "outcome_node": OUTCOME_NODE,
        "train_source": str(train_path),
        "edge_layers": [
            "point_biserial (marginal r with Stroke)",
            "marginal_logistic (univariate logistic on train)",
            "partial_corr_scd (partial r with Stroke)",
            "partial_corr (feature–feature)",
        ],
        "preprocessing": "Median imputation on train split only (computed in pipeline 6; not from ML pipelines).",
        "excluded_inputs": (
            "No LASSO feature list, post-hoc GLM CSV, permutation CSV, "
            "domain ontology, or any 05_results/01–05 / posthoc artifact."
        ),
        "interpretation": (
            "Pathway scores summarize data-derived associations on the training split only. "
            "Compare with pipeline 05_knowledge_graph to assess convergence with ML-integrated KG."
        ),
        "top_10_features": scores_df.head(10)["feature"].tolist(),
    }
    (out_dir / "pipeline_config.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"=== Pipeline 6 (data-only KG): {dataset} ===")
    print(f"  predictors={len(features)} nodes={meta['n_nodes']} edges={meta['n_edges']}")
    print(f"  train={train_path}")
    print(f"  wrote {out_dir}/")
    print("  Top pathways to stroke:")
    for _, row in scores_df.head(8).iterrows():
        print(f"    {int(row['rank']):2d}. {row['feature']:<22} score={row['pathway_score']:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Pipeline 6: knowledge graph from processed train CSV only"
    )
    ap.add_argument("--dataset", choices=["ny", "all_states", "both"], default="both")
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Higher correlation / OR thresholds (fewer edges, faster)",
    )
    args = ap.parse_args()
    datasets = ["ny", "all_states"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        run_kg_data_only(ds, quick=args.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
