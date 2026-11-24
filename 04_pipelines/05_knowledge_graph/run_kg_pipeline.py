#!/usr/bin/env python3
"""
Knowledge-graph pathway analysis for BRFSS stroke (pipeline 5).

Builds a heterogeneous graph from:
  - survey-weighted / unweighted GLM adjusted associations (post-hoc)
  - permutation importance (lead Python pipeline)
  - partial correlations on training data
  - domain ontology priors (epidemiologic feature groupings)

Ranks features by pathway contribution toward stroke. This is an **associational**
pathway model — not proof of causation.

Usage (from project/ after bootstrap):
  python knowledge_graph_pipeline/run_kg_pipeline.py --dataset ny
  python knowledge_graph_pipeline/run_kg_pipeline.py --dataset both
  python knowledge_graph_pipeline/run_kg_pipeline.py --dataset all_states --quick
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PIPE_DIR = Path(__file__).resolve().parent
PKG = PIPE_DIR.parents[1]  # Collaboration_Package root
sys.path.insert(0, str(PIPE_DIR))

from kg_core import (  # noqa: E402
    OUTCOME_NODE,
    build_graph,
    domain_prior_edges,
    edges_to_dataframe,
    load_glm_edges,
    load_ontology,
    load_permutation_edges,
    partial_correlation_matrix,
    score_nodes,
    top_pathways,
)
from kg_visualize import (  # noqa: E402
    export_graphml,
    plot_interactive_html,
    plot_publication_kg,
    plot_static_kg,
)

PROJECT = PKG / "project"
if not (PROJECT / "datasets").exists() and (PKG / "02_datasets").exists():
    # Allow running without bootstrap: read datasets from package layout.
    PROCESSED_FALLBACK = PKG / "02_datasets" / "processed"
else:
    PROCESSED_FALLBACK = None

if (PROJECT / "datasets").exists():
    PROCESSED = PROJECT / "datasets" / "processed"
    RESULTS_ML = PROJECT / "results" / "ml"
    POSTHOC = PROJECT / "results" / "posthoc"
else:
    PROCESSED = PROCESSED_FALLBACK or (PKG / "02_datasets" / "processed")
    RESULTS_ML = PKG / "05_results" / "01_python_ml_standard"
    POSTHOC = PKG / "05_results" / "posthoc"
OUT_ROOT = PKG / "05_results" / "05_knowledge_graph"
ONTOLOGY = PIPE_DIR / "domain_ontology.json"


def _dataset_prefix(dataset: str) -> str:
    return "brfss_stroke_new_york" if dataset == "ny" else "brfss_stroke_all_states"


def _load_features(dataset: str) -> list[str]:
    feat_path = RESULTS_ML / dataset / "02_lasso_selected_features.txt"
    if not feat_path.exists():
        cfg = RESULTS_ML / dataset / "pipeline_config.json"
        if cfg.exists():
            return json.loads(cfg.read_text(encoding="utf-8"))["selected_features"]
        raise FileNotFoundError(
            f"No LASSO features for {dataset}. Run pipeline 1 first: "
            f"python scripts/run_ml_pipeline.py --dataset {dataset}"
        )
    return [
        ln.strip()
        for ln in feat_path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]


GLM_FILE = {
    "all_states": "glm_or_table_train_unweighted.r.csv",
    "ny": "glm_or_table_train_svy_weighted.r.csv",
}
def _glm_path(dataset: str) -> Path:
    p = POSTHOC / f"glm_{dataset}" / GLM_FILE[dataset]
    if not p.exists():
        raise FileNotFoundError(
            f"Missing GLM table for {dataset}: {p}. "
            "Run the posthoc GLM script (fit_interpretable_glm.R) first."
        )
    return p


def run_kg(dataset: str, *, quick: bool = False) -> None:
    prefix = _dataset_prefix(dataset)
    train_path = PROCESSED / f"{prefix}_train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing training data: {train_path}")

    features = _load_features(dataset)
    feat_set = set(features)
    train = pd.read_csv(train_path)
    ontology = load_ontology(ONTOLOGY)

    out_dir = OUT_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_records: list[dict] = []
    edge_records.extend(load_glm_edges(_glm_path(dataset), feat_set))
    edge_records.extend(
        load_permutation_edges(
            RESULTS_ML / dataset / "09_permutation_importance.csv", feat_set
        )
    )
    min_r = 0.18 if quick else 0.12
    edge_records.extend(
        partial_correlation_matrix(train, features, min_abs_r=min_r)
    )
    edge_records.extend(domain_prior_edges(ontology, feat_set))

    g = build_graph(edge_records, features)
    scores_df = score_nodes(g, features)
    pathways_df = top_pathways(g, features)
    edges_df = edges_to_dataframe(g)

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
        title=f"Knowledge graph pathways to stroke ({cohort_label}, train-derived)",
    )
    plot_interactive_html(
        g,
        score_map,
        out_dir / "06_kg_network_interactive.html",
        title=f"Stroke knowledge graph — {cohort_label}",
    )

    meta = {
        "pipeline": "05_knowledge_graph",
        "dataset": dataset,
        "n_features": len(features),
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "outcome_node": OUTCOME_NODE,
        "glm_source": str(_glm_path(dataset)),
        "permutation_source": str(RESULTS_ML / dataset / "09_permutation_importance.csv"),
        "train_source": str(train_path),
        "ontology": str(ONTOLOGY),
        "interpretation": (
            "Pathway scores integrate GLM associations, ML permutation importance, "
            "partial correlations, and domain priors. High scores indicate features that "
            "sit on dense associational pathways toward stroke — not established causation."
        ),
        "top_10_features": scores_df.head(10)["feature"].tolist(),
    }
    (out_dir / "pipeline_config.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print(f"=== Knowledge graph: {dataset} ===")
    print(f"  nodes={meta['n_nodes']} edges={meta['n_edges']}")
    print(f"  wrote {out_dir}/")
    print("  Top pathways to stroke:")
    for _, row in scores_df.head(8).iterrows():
        print(f"    {int(row['rank']):2d}. {row['feature']:<22} score={row['pathway_score']:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="BRFSS stroke knowledge-graph pathway pipeline")
    ap.add_argument("--dataset", choices=["ny", "all_states", "both"], default="both")
    ap.add_argument("--quick", action="store_true", help="Higher partial-corr threshold (faster)")
    args = ap.parse_args()

    datasets = ["ny", "all_states"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        run_kg(ds, quick=args.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
