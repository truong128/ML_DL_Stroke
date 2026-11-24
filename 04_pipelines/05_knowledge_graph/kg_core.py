"""Knowledge-graph construction and pathway scoring for BRFSS stroke features."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

OUTCOME_NODE = "Stroke"
EDGE_SOURCES = ("glm", "permutation", "partial_corr", "domain_prior", "category")


def load_ontology(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def feature_category_map(ontology: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for cat_id, info in ontology["categories"].items():
        for feat in info.get("features", []):
            mapping[feat] = cat_id
    return mapping


def _clean_glm_term(term: str) -> str:
    s = str(term)
    if s.startswith("Q('") and s.endswith("')"):
        return s[3:-2]
    return (
        s.replace("Q('", "")
        .replace("')", "")
        .replace("C(", "")
        .replace(")", "")
        .strip()
    )


def load_glm_edges(glm_path: Path, features: set[str]) -> list[dict[str, Any]]:
    if not glm_path.exists():
        return []
    df = pd.read_csv(glm_path)
    df = df[df["term"].astype(str).str.contains("Intercept", case=False) == False].copy()
    df["feature"] = df["term"].map(_clean_glm_term)
    or_col = "or" if "or" in df.columns else "estimate"
    p_col = "p_value" if "p_value" in df.columns else "p.value"
    edges: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        feat = row["feature"]
        if feat not in features:
            continue
        or_val = float(row[or_col])
        p = float(row[p_col]) if pd.notna(row.get(p_col)) else 1.0
        q = float(row.get("q_value_fdr_bh", np.nan))
        sig = bool(row.get("significant_fdr_05", p < 0.05))
        weight = min(-np.log10(max(p, 1e-300)), 50.0) * abs(np.log(or_val if or_val > 0 else 1.0))
        if sig and pd.notna(q):
            weight *= 1.0 + min(-np.log10(max(q, 1e-300)), 20.0) / 20.0
        edges.append(
            {
                "source": feat,
                "target": OUTCOME_NODE,
                "weight": float(weight),
                "source_type": "glm",
                "relation": "adjusted_association",
                "or": or_val,
                "p_value": p,
                "q_value": q if pd.notna(q) else None,
                "significant_fdr_05": sig,
            }
        )
    return edges


def load_permutation_edges(perm_path: Path, features: set[str]) -> list[dict[str, Any]]:
    if not perm_path.exists():
        return []
    df = pd.read_csv(perm_path)
    max_imp = float(df["importance_mean"].max()) or 1.0
    edges: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        feat = row["feature"]
        if feat not in features:
            continue
        imp = float(row["importance_mean"])
        edges.append(
            {
                "source": feat,
                "target": OUTCOME_NODE,
                "weight": imp / max_imp,
                "source_type": "permutation",
                "relation": "predictive_importance",
                "importance_mean": imp,
                "importance_std": float(row.get("importance_std", np.nan)),
            }
        )
    return edges


def partial_correlation_matrix(
    df: pd.DataFrame, features: list[str], *, min_abs_r: float = 0.12
) -> list[dict[str, Any]]:
    """Pairwise partial correlations controlling for all other features + outcome."""
    cols = features + ["stroke"]
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    sub = sub.dropna()
    if len(sub) < 50:
        return []

    edges: list[dict[str, Any]] = []
    n = len(cols)
    data = sub.values
    precision = np.linalg.pinv(np.cov(data, rowvar=False))
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            a, b = i, j
            if precision[a, a] <= 0 or precision[b, b] <= 0:
                continue
            r = -precision[a, b] / np.sqrt(precision[a, a] * precision[b, b])
            if abs(r) < min_abs_r:
                continue
            w = abs(float(r))
            edges.append(
                {
                    "source": features[a],
                    "target": features[b],
                    "weight": w,
                    "source_type": "partial_corr",
                    "relation": "conditional_association",
                    "partial_r": float(r),
                }
            )
    return edges


def domain_prior_edges(
    ontology: dict[str, Any], features: set[str]
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for link in ontology.get("directed_priors", []):
        src, tgt = link["source"], link["target"]
        if src in features and tgt in features:
            edges.append(
                {
                    "source": src,
                    "target": tgt,
                    "weight": float(link["weight"]),
                    "source_type": "domain_prior",
                    "relation": link.get("relation", "prior_pathway"),
                }
            )
    cat_map = feature_category_map(ontology)
    for feat in features:
        cat = cat_map.get(feat)
        if not cat:
            continue
        prior = float(ontology["categories"][cat].get("stroke_prior_weight", 0.2))
        edges.append(
            {
                "source": feat,
                "target": OUTCOME_NODE,
                "weight": prior * 0.25,
                "source_type": "category",
                "relation": f"category_prior:{cat}",
                "category": cat,
            }
        )
    return edges


def _edge_distance(weight: float) -> float:
    """Lower distance = stronger evidence (for shortest-path algorithms)."""
    return 1.0 / (float(weight) + 1e-6)


def build_graph(edge_records: list[dict[str, Any]], features: list[str]) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node(OUTCOME_NODE, node_type="outcome")
    for feat in features:
        g.add_node(feat, node_type="feature")
    for rec in edge_records:
        src, tgt, w = rec["source"], rec["target"], float(rec["weight"])
        if src not in g or tgt not in g or w <= 0:
            continue
        if g.has_edge(src, tgt):
            g[src][tgt]["weight"] += w
            g[src][tgt]["distance"] = _edge_distance(g[src][tgt]["weight"])
            g[src][tgt]["sources"].add(rec["source_type"])
        else:
            g.add_edge(
                src,
                tgt,
                weight=w,
                distance=_edge_distance(w),
                sources={rec["source_type"]},
                relation=rec.get("relation", ""),
            )
        if rec["source_type"] == "partial_corr":
            if g.has_edge(tgt, src):
                g[tgt][src]["weight"] += w
                g[tgt][src]["distance"] = _edge_distance(g[tgt][src]["weight"])
                g[tgt][src]["sources"].add(rec["source_type"])
            else:
                g.add_edge(
                    tgt,
                    src,
                    weight=w,
                    distance=_edge_distance(w),
                    sources={rec["source_type"]},
                    relation=rec.get("relation", ""),
                )
    return g


def score_nodes(g: nx.DiGraph, features: list[str]) -> pd.DataFrame:
    """Rank features by multi-metric pathway contribution toward stroke."""
    pr = nx.pagerank(g, weight="weight", alpha=0.85)
    try:
        between = nx.betweenness_centrality(g, weight="weight", normalized=True)
    except Exception:
        between = {n: 0.0 for n in g.nodes}

    personalization = {n: 1.0 if n == OUTCOME_NODE else 0.0 for n in g.nodes}
    try:
        ppr = nx.pagerank(
            g, weight="weight", alpha=0.85, personalization=personalization
        )
    except Exception:
        ppr = pr

    rows: list[dict[str, Any]] = []
    for feat in features:
        if feat not in g:
            continue
        direct = 0.0
        glm_w = perm_w = 0.0
        if g.has_edge(feat, OUTCOME_NODE):
            direct = float(g[feat][OUTCOME_NODE]["weight"])
            srcs = g[feat][OUTCOME_NODE].get("sources", set())
            if "glm" in srcs:
                glm_w = direct
            if "permutation" in srcs:
                perm_w = direct

        path_len = np.inf
        try:
            path_len = nx.shortest_path_length(g, feat, OUTCOME_NODE, weight="distance")
        except nx.NetworkXNoPath:
            pass
        path_score = 1.0 / (1.0 + path_len) if np.isfinite(path_len) else 0.0

        in_deg = float(g.in_degree(feat, weight="weight"))
        out_deg = float(g.out_degree(feat, weight="weight"))

        direct_norm = direct / (direct + 1.0)
        composite = (
            0.25 * pr.get(feat, 0.0)
            + 0.15 * ppr.get(feat, 0.0)
            + 0.15 * path_score
            + 0.35 * direct_norm
            + 0.10 * between.get(feat, 0.0)
        )
        rows.append(
            {
                "feature": feat,
                "pathway_score": composite,
                "pagerank": pr.get(feat, 0.0),
                "personalized_pagerank_to_stroke": ppr.get(feat, 0.0),
                "betweenness": between.get(feat, 0.0),
                "shortest_path_weight_to_stroke": path_len if np.isfinite(path_len) else None,
                "path_proximity_score": path_score,
                "direct_edge_weight_to_stroke": direct,
                "in_strength": in_deg,
                "out_strength": out_deg,
                "glm_edge_weight": glm_w,
                "permutation_edge_weight": perm_w,
            }
        )

    df = pd.DataFrame(rows).sort_values("pathway_score", ascending=False)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def top_pathways(
    g: nx.DiGraph, features: list[str], *, top_n: int = 25
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feat in features:
        if feat not in g:
            continue
        try:
            path = nx.shortest_path(g, feat, OUTCOME_NODE, weight="distance")
            length = nx.shortest_path_length(g, feat, OUTCOME_NODE, weight="distance")
        except nx.NetworkXNoPath:
            continue
        rows.append(
            {
                "feature": feat,
                "path_length": length,
                "path": " → ".join(path),
                "hops": len(path) - 1,
            }
        )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values(["path_length", "feature"])
        .head(top_n)
        .reset_index(drop=True)
    )


def edges_to_dataframe(g: nx.DiGraph) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for u, v, data in g.edges(data=True):
        rows.append(
            {
                "source": u,
                "target": v,
                "weight": data.get("weight", 0.0),
                "sources": ",".join(sorted(data.get("sources", []))),
                "relation": data.get("relation", ""),
            }
        )
    return pd.DataFrame(rows).sort_values(["source", "target"]).reset_index(drop=True)
