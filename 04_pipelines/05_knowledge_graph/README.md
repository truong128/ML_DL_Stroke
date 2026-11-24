# Pipeline 5 — Knowledge graph pathway analysis (Python)

**Purpose:** Rank BRFSS predictors by how strongly they sit on **associational pathways** toward subjective cognitive decline (Stroke), combining:

| Evidence layer | Source | Edge meaning |
|----------------|--------|--------------|
| Epidemiologic GLM | `results/posthoc/glm_{cohort}/` | Adjusted OR → Stroke |
| ML permutation | `results/ml/{cohort}/09_permutation_importance.csv` | Predictive importance → Stroke |
| Data-driven | Training-split partial correlations | Conditional feature–feature links |
| Domain ontology | `domain_ontology.json` | Literature-informed priors & groupings |

**Important:** Pathway scores indicate **statistical and structural proximity** to stroke in a knowledge graph. They do **not** establish causal disease mechanisms. Use alongside GLM (Table 6), permutation importance (Figure 7), and SHAP (Figure 8).

## Prerequisites

1. Pipeline 1 completed (`results/ml/{ny,all_states}/`)
2. Post-hoc GLMs fitted (`scripts/fit_interpretable_glm.R` or `.py`)
3. KG dependencies installed (see below)

## Install

```bash
source ~/.venvs/15_NY_Neurological_Diseases/bin/activate
pip install -r 04_pipelines/05_knowledge_graph/requirements-kg.txt
```

## Run

From `project/` (after `bootstrap_project.sh`):

```bash
python knowledge_graph_pipeline/run_kg_pipeline.py --dataset both
python knowledge_graph_pipeline/run_kg_pipeline.py --dataset ny --quick
```

Or directly from the package:

```bash
python 04_pipelines/05_knowledge_graph/run_kg_pipeline.py --dataset both
```

## Outputs

`05_results/05_knowledge_graph/{ny,all_states}/`

| File | Description |
|------|-------------|
| `01_kg_feature_pathway_scores.csv` | Ranked features (composite pathway score) |
| `02_kg_edges.csv` | Full edge list with evidence sources |
| `03_kg_shortest_pathways_to_stroke.csv` | Shortest paths feature → Stroke |
| `04_kg_graph.graphml` | GraphML for Gephi / Cytoscape |
| `05_kg_network.png` | Static network (top features) |
| `06_kg_network_interactive.html` | Interactive PyVis graph |
| `pipeline_config.json` | Run metadata |

## Scoring

Composite **pathway_score** = weighted blend of PageRank, personalized PageRank (target = Stroke), shortest-path proximity, direct edge strength to stroke, and betweenness centrality.
