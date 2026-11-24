"""Export static and interactive knowledge-graph views."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import numpy as np

OUTCOME = "Stroke"

# Publication radial network (Figure 9 style)
STROKE_COLOR = "#c0392b"
NODE_FILL = "#2c6faa"
NODE_STROKE = "#1f4e79"
EDGE_STRONG = "#1f4e79"
EDGE_WEAK = "#d35400"
EDGE_FEATURE = "#7b1fa2"
LABEL_COLOR = "#1a252f"
PUB_FONT = "Times New Roman"
PUB_FONT_PT = 12.0
# Match composite panel slot (build_publication_figures KG_PANEL_W × KG_PANEL_PT).
KG_EMBED_PANEL_W = 738.0
KG_EMBED_PANEL_H = 400.0
FIG_IN_W = KG_EMBED_PANEL_W / 72.0
FIG_IN_H = KG_EMBED_PANEL_H / 72.0
NODE_SCATTER = 200
STROKE_SCATTER = 340
NODE_R_DATA = 0.038
LABEL_GAP = 0.016
RADIAL_R_MIN = 0.34
RADIAL_R_SPAN = 1.08
ZOOM_PAD = 0.035

# Optional per-node label nudges in data coordinates (dx, dy).
_LABEL_NUDGES: dict[str, dict[str, tuple[float, float]]] = {
    "all_states": {},
    "ny": {},
}


def export_graphml(g: nx.DiGraph, path: Path) -> None:
    serial = g.copy()
    for _, _, data in serial.edges(data=True):
        if "sources" in data and isinstance(data["sources"], set):
            data["sources"] = ",".join(sorted(data["sources"]))
    nx.write_graphml(serial, path)


def _pretty_label(name: str) -> str:
    return name.replace("_", " ")


def _path_lengths(g: nx.DiGraph, features: list[str]) -> dict[str, float]:
    lengths: dict[str, float] = {}
    for feat in features:
        try:
            lengths[feat] = float(
                nx.shortest_path_length(g, feat, OUTCOME, weight="distance")
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            lengths[feat] = float("inf")
    finite = [v for v in lengths.values() if np.isfinite(v)]
    fallback = max(finite) if finite else 1.0
    return {k: (v if np.isfinite(v) else fallback * 1.25) for k, v in lengths.items()}


def _radial_positions(
    features: list[str],
    path_len: dict[str, float],
    *,
    cohort: str,
) -> dict[str, tuple[float, float]]:
    if not features:
        return {OUTCOME: (0.0, 0.0)}

    vals = [path_len[f] for f in features]
    lo, hi = min(vals), max(vals)
    n = len(features)
    pos: dict[str, tuple[float, float]] = {OUTCOME: (0.0, 0.0)}
    for i, feat in enumerate(features):
        angle = 2.0 * np.pi * i / n - np.pi / 2.0
        if hi > lo:
            t = (path_len[feat] - lo) / (hi - lo)
        else:
            t = 0.5
        r = RADIAL_R_MIN + RADIAL_R_SPAN * t
        dx, dy = r * np.cos(angle), r * np.sin(angle)
        nudge = _LABEL_NUDGES.get(cohort, {}).get(feat, (0.0, 0.0))
        pos[feat] = (dx + nudge[0], dy + nudge[1])
    return pos


def _label_extent_data() -> float:
    """Approximate label box reach in data units for zoom padding (12 pt labels)."""
    return NODE_R_DATA + LABEL_GAP + 0.055


def _zoom_limits(
    pos: dict[str, tuple[float, float]],
    features: list[str],
    cx: float,
    cy: float,
    *,
    fig_aspect: float,
    pad_frac: float = ZOOM_PAD,
) -> tuple[float, float, float, float]:
    """Tight data limits; expand to panel aspect so graph fills the embed slot."""
    label_ext = _label_extent_data()
    xs = [cx]
    ys = [cy]
    for feat in features:
        fx, fy = pos[feat]
        xs.append(fx)
        ys.append(fy)
        dx, dy = fx - cx, fy - cy
        n = np.hypot(dx, dy) or 1.0
        xs.append(fx + dx / n * label_ext)
        ys.append(fy + dy / n * label_ext)

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    mx, my = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    half_x = (xmax - xmin) / 2.0
    half_y = (ymax - ymin) / 2.0
    pad = pad_frac * max(half_x, half_y, 0.05)
    half_x += pad
    half_y += pad
    # Fill wide panel: expand limits to figure aspect (zoom-in, less empty margin).
    if half_x < half_y * fig_aspect:
        half_x = half_y * fig_aspect
    else:
        half_y = half_x / fig_aspect
    return mx - half_x, mx + half_x, my - half_y, my + half_y


def _label_anchor(x: float, y: float) -> tuple[str, str]:
    if abs(x) < 0.08:
        ha = "center"
    elif x > 0:
        ha = "left"
    else:
        ha = "right"
    if abs(y) < 0.08:
        va = "center"
    elif y > 0:
        va = "bottom"
    else:
        va = "top"
    return ha, va


def _draw_label_box(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    is_outcome: bool = False,
    font_size: float = PUB_FONT_PT,
    ha: str | None = None,
    va: str | None = None,
) -> None:
    if ha is None or va is None:
        ha, va = _label_anchor(x, y)
    if is_outcome:
        ha, va = "center", "center"
        bbox = dict(
            boxstyle="round,pad=0.18",
            facecolor="#fff6f5",
            edgecolor=STROKE_COLOR,
            linewidth=0.8,
            alpha=0.96,
        )
        color = STROKE_COLOR
        weight = "bold"
    else:
        bbox = dict(
            boxstyle="round,pad=0.12",
            facecolor="#ffffff",
            edgecolor="#cfd8dc",
            linewidth=0.5,
            alpha=0.94,
        )
        color = LABEL_COLOR
        weight = "normal"
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=font_size,
        fontfamily=PUB_FONT,
        fontweight=weight,
        color=color,
        bbox=bbox,
        zorder=6,
        clip_on=False,
    )


def plot_publication_kg(
    g: nx.DiGraph,
    scores: dict[str, float],
    path: Path,
    *,
    cohort: str = "all_states",
    top_k: int = 18,
    title: str | None = None,
) -> None:
    """Central-hub radial network (Figure 9 style): PNG + PDF + SVG."""
    ranked = sorted(
        ((n, scores.get(n, 0.0)) for n in g.nodes if n != OUTCOME),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]
    features = [n for n, _ in ranked]
    keep = set(features) | {OUTCOME}
    sub = g.subgraph(keep).copy()

    path_len = _path_lengths(sub, features)
    outcome_paths = [path_len[f] for f in features if sub.has_edge(f, OUTCOME)]
    split = float(np.median(outcome_paths)) if outcome_paths else 0.0

    pos = _radial_positions(features, path_len, cohort=cohort)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [PUB_FONT, "Times", "DejaVu Serif"],
        "font.size": PUB_FONT_PT,
        "svg.fonttype": "none",
    })

    fig, ax = plt.subplots(figsize=(FIG_IN_W, FIG_IN_H))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_position([0, 0, 1, 1])
    ax.set_aspect("equal", anchor="C")
    ax.axis("off")
    fig_aspect = FIG_IN_W / FIG_IN_H
    x0, x1, y0, y1 = _zoom_limits(pos, features, pos[OUTCOME][0], pos[OUTCOME][1], fig_aspect=fig_aspect)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    ff_edges = [
        (u, v)
        for u, v in sub.edges()
        if u in features and v in features and u != OUTCOME and v != OUTCOME
    ]
    outcome_edges = [
        (u, v)
        for u, v in sub.edges()
        if OUTCOME in (u, v) and (u in features or v in features)
    ]

    for i, (u, v) in enumerate(ff_edges):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        rad = 0.12 * ((i % 5) - 2) / 2.0
        arrow = FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=8,
            color=EDGE_FEATURE,
            linewidth=1.2,
            linestyle=(0, (4.44, 1.92)),
            alpha=0.92,
            connectionstyle=f"arc3,rad={rad:.3f}",
            shrinkA=6,
            shrinkB=6,
            zorder=1,
        )
        ax.add_patch(arrow)

    for u, v in outcome_edges:
        feat = u if v == OUTCOME else v
        if feat not in pos:
            continue
        x0, y0 = pos[feat]
        x1, y1 = pos[OUTCOME]
        pl = path_len.get(feat, split)
        strong = pl <= split
        color = EDGE_STRONG if strong else EDGE_WEAK
        lw = 3.2 if strong else 2.2
        arrow = FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=10 if strong else 8,
            color=color,
            linewidth=lw,
            alpha=0.95 if strong else 0.92,
            connectionstyle="arc3,rad=0.0",
            shrinkA=6,
            shrinkB=8,
            zorder=2,
        )
        ax.add_patch(arrow)

    ax.scatter(
        [pos[f][0] for f in features],
        [pos[f][1] for f in features],
        s=NODE_SCATTER,
        c=NODE_FILL,
        edgecolors=NODE_STROKE,
        linewidths=1.0,
        alpha=0.95,
        zorder=4,
    )
    ax.scatter(
        [pos[OUTCOME][0]],
        [pos[OUTCOME][1]],
        s=STROKE_SCATTER,
        c=STROKE_COLOR,
        edgecolors="#1a252f",
        linewidths=1.0,
        alpha=0.95,
        zorder=5,
    )

    cx, cy = pos[OUTCOME]
    for feat in features:
        fx, fy = pos[feat]
        dx, dy = fx - cx, fy - cy
        norm = np.hypot(dx, dy) or 1.0
        ux, uy = dx / norm, dy / norm
        lx = fx + ux * (NODE_R_DATA + LABEL_GAP)
        ly = fy + uy * (NODE_R_DATA + LABEL_GAP)
        if ux > 0.12:
            ha = "left"
        elif ux < -0.12:
            ha = "right"
        else:
            ha = "center"
        if uy > 0.12:
            va = "bottom"
        elif uy < -0.12:
            va = "top"
        else:
            va = "center"
        _draw_label_box(
            ax, lx, ly, _pretty_label(feat),
            font_size=PUB_FONT_PT, ha=ha, va=va,
        )

    _draw_label_box(ax, cx, cy, OUTCOME, is_outcome=True, font_size=PUB_FONT_PT)

    path = Path(path)
    if path.suffix:
        base = path.with_suffix("")
    else:
        base = path
    base.parent.mkdir(parents=True, exist_ok=True)
    save_kw = dict(facecolor="white", bbox_inches=None, pad_inches=0)
    for ext in (".svg", ".png", ".pdf"):
        kwargs = dict(save_kw)
        if ext != ".svg":
            kwargs["dpi"] = 300
        fig.savefig(base.with_suffix(ext), **kwargs)
    plt.close(fig)


def plot_static_kg(
    g: nx.DiGraph,
    scores: dict[str, float],
    path: Path,
    *,
    title: str,
    top_k: int = 18,
) -> None:
    """Spring-layout preview (interactive exploration)."""
    ranked = sorted(
        ((n, scores.get(n, 0.0)) for n in g.nodes if n != OUTCOME),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]
    keep = {n for n, _ in ranked} | {OUTCOME}
    sub = g.subgraph(keep).copy()

    pos = nx.spring_layout(sub, seed=42, k=1.4, weight="weight")
    sizes = [900 if n == OUTCOME else 300 + 1200 * scores.get(n, 0.0) for n in sub.nodes]
    colors = [STROKE_COLOR if n == OUTCOME else "#2980b9" for n in sub.nodes]

    fig, ax = plt.subplots(figsize=(11, 8))
    nx.draw_networkx_edges(
        sub,
        pos,
        ax=ax,
        edge_color="#95a5a6",
        arrows=True,
        arrowsize=12,
        width=[min(3.0, 0.5 + d.get("weight", 0.1)) for _, _, d in sub.edges(data=True)],
        alpha=0.55,
    )
    nx.draw_networkx_nodes(sub, pos, node_size=sizes, node_color=colors, ax=ax, alpha=0.92)
    labels = {n: n.replace("_", "\n") for n in sub.nodes}
    nx.draw_networkx_labels(sub, pos, labels=labels, font_size=7, ax=ax)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_interactive_html(g: nx.DiGraph, scores: dict[str, float], path: Path, *, title: str) -> None:
    try:
        from pyvis.network import Network
    except ImportError as exc:
        raise ImportError("pyvis is required for HTML export. pip install pyvis") from exc

    net = Network(height="720px", width="100%", directed=True, notebook=False)
    net.barnes_hut(gravity=-2800, central_gravity=0.25, spring_length=140)

    for node in g.nodes:
        score = scores.get(node, 0.0)
        color = "#e74c3c" if node == OUTCOME else "#3498db"
        size = 34 if node == OUTCOME else 14 + 26 * min(score / (score + 0.05), 1.0)
        net.add_node(
            node,
            label=node.replace("_", " "),
            title=f"{node}\npathway_score={score:.4f}",
            color=color,
            size=size,
        )
    for u, v, data in g.edges(data=True):
        w = float(data.get("weight", 0.1))
        net.add_edge(u, v, value=w, title=data.get("relation", ""))

    net.set_options(
        """
        var options = {
          "physics": {"enabled": true, "stabilization": {"iterations": 120}},
          "interaction": {"hover": true, "navigationButtons": true}
        }
        """
    )
    net.write_html(str(path), open_browser=False, notebook=False)
    html = path.read_text(encoding="utf-8")
    html = html.replace("<title>Pyvis Network</title>", f"<title>{title}</title>")
    path.write_text(html, encoding="utf-8")
