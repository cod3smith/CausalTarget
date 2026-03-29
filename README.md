# 🧬 CausalTarget — Causal Drug Target Discovery

> *"Correlation is not causation — CausalTarget knows the difference."*

[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-108%20passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()

**CausalTarget** is an AI-driven drug target discovery pipeline that applies
**Pearl's causal inference framework** to distinguish genuine causal drug
targets from correlational bystanders.  Unlike conventional association-based
pipelines, CausalTarget asks:

> *"If we intervene on this protein (inhibit or activate it with a drug),
> does it causally affect the disease outcome?"*

## The Problem

Gene–disease association databases (Monarch Initiative, Open Targets) report that
**TNF-α** is strongly associated with HIV.  TNF-α levels *are* elevated
during HIV infection.  But TNF-α elevation is a **downstream consequence**
of immune activation — inhibiting it does not treat HIV; it makes it worse.

Conversely, **CCR5** is the HIV-1 co-receptor.  A loss-of-function mutation
(CCR5-Δ32) confers near-complete resistance.  Maraviroc, which blocks CCR5,
is an approved antiretroviral.  CCR5 is a **causal** target.

CausalTarget distinguishes these two cases automatically.

## Pipeline Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  7 Biomedical   │────▶│  Causal Knowledge │────▶│  Do-Calculus     │
│  Databases      │     │  Graph (NetworkX) │     │  Analysis        │
│  Monarch Init.  │     │                  │     │  • Backdoor      │
│  Open Targets   │     │  Genes, Pathways, │     │  • d-Separation  │
│  KEGG, Reactome │     │  PPIs, Structures │     │  • Sensitivity   │
│  STRING, UniProt│     │                  │     │                 │
│  RCSB PDB       │     └──────────────────┘     └────────┬────────┘
└─────────────────┘                                       │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  HTML Report    │◀────│  Composite Scorer │◀────│  GenMol +        │
│  Interactive    │     │  6D Ranking:      │     │  MolScreen +     │
│  Graph + Tables │     │  Causal > Binding │     │  DockBot         │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Six-Step Workflow

1. **Build Causal Knowledge Graph** — Query 7 biomedical databases in
   parallel, merge into a unified directed graph.

2. **Identify Causal Targets** — Apply the backdoor criterion and
   d-separation analysis to distinguish causal from correlational targets.
   Multi-source evidence triangulation and leave-one-source-out sensitivity
   analysis for robustness.

3. **Generate Candidate Molecules** — GenMol variational autoencoder
   generates novel drug-like molecules targeting each causal protein.

4. **Screen Candidates** — MolScreen evaluates drug-likeness (Lipinski,
   QED, SA) and DockBot estimates binding affinity via AutoDock Vina.

5. **Score & Rank** — Composite scorer integrates six dimensions with
   **causal confidence weighted highest** (0.30):

   | Dimension           | Weight |
   |---------------------|--------|
   | Causal confidence   | 0.30   |
   | Binding affinity    | 0.25   |
   | Drug-likeness (QED) | 0.15   |
   | Synthetic access.   | 0.10   |
   | ADMET               | 0.10   |
   | Novelty             | 0.10   |

6. **Report** — Interactive HTML report with causal knowledge graph
   visualisation, target reasoning, and ranked candidate table.

## Installation

```bash
# From PyPI
pip install causaltarget

# From source with uv
git clone https://github.com/NeoForge/CausalTarget.git
cd CausalTarget
uv sync

# From source with pip
git clone https://github.com/NeoForge/CausalTarget.git
cd CausalTarget
pip install -e .
```

### Optional Dependencies

```bash
uv sync --extra docking   # AutoDock Vina support
uv sync --extra mirror    # ESM-2 protein language model
```

### Docker (with persistence)

```bash
# Start Redis + PostgreSQL + API
cp .env.example .env
docker compose up -d

# API available at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

## Quick Start

### CLI

```bash
# Full pipeline for HIV
causaltarget run HIV --top-n 5

# Build the causal knowledge graph only
causaltarget graph "Type 2 Diabetes"

# Identify causal targets (no molecule generation)
causaltarget identify "Alzheimer Disease" --top-n 10

# Export graph to Cytoscape format
causaltarget graph HIV --export cytoscape

# Start API server
causaltarget serve --port 8000
```

### Python API

```python
from causaltarget import (
    build_disease_graph,
    identify_causal_targets,
    run_pipeline,
)

# Step-by-step
graph = build_disease_graph("HIV")
targets = identify_causal_targets(graph, top_n=10)

for t in targets:
    print(f"{t.gene_name}: {t.classification.value} "
          f"(confidence={t.causal_confidence:.3f})")

# Or run the full pipeline
result = run_pipeline("HIV", top_n_targets=5)
print(f"Found {result.n_causal_targets} causal targets")
print(f"Scored {len(result.scored_candidates)} candidates")
```

### REST API

```bash
# Full pipeline
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{"disease": "HIV", "top_n_targets": 5}'

# Build graph only
curl -X POST http://localhost:8000/graph \
  -d '{"disease": "Type 2 Diabetes"}'

# Identify causal targets
curl -X POST http://localhost:8000/identify \
  -d '{"disease": "HIV", "top_n": 10}'
```

## Data Sources

| Database     | Data Type                | API Style    | Caching |
|--------------|--------------------------|--------------|---------|
| Monarch Init.| Gene–disease associations| REST (v3)    | ✅ 24h   |
| Open Targets | Target–disease evidence  | GraphQL      | ✅ 24h   |
| KEGG         | Pathway membership       | REST         | ✅ 24h   |
| Reactome     | Pathway membership       | REST         | ✅ 24h   |
| STRING       | Protein–protein interactions| REST (batch)| ✅ 24h   |
| UniProt      | Protein metadata         | REST         | ✅ 24h   |
| RCSB PDB     | 3D structures            | REST/Search  | ✅ 24h   |

All clients include curated mock fallback data for offline use and testing.

## Methodology

### Causal Inference Approach

CausalTarget uses **multi-source evidence triangulation** rather than
fabricated synthetic data:

1. **Graph-Topological Causal Analysis** — Path strength from treatment
   to disease computed as the product of edge weights.  Shorter, more
   direct paths indicate stronger causal mechanisms.

2. **d-Separation (Backdoor Criterion)** — Proper implementation using
   NetworkX's `d_separated()` to verify that causal effects are
   identifiable after conditioning on a valid adjustment set.

3. **Multi-Source Triangulation** — Evidence from independent databases
   is aggregated.  A target confirmed by Monarch Initiative, Open Targets, *and*
   STRING is more credible than one from a single source.

4. **Leave-One-Source-Out Sensitivity** — For each data source, we
   remove its evidence and recompute the effect.  Stable estimates
   across source removals indicate robustness.

5. **Bootstrap Confidence Intervals** — 95% CIs on causal confidence
   via bootstrap resampling over evidence components.

### Target Classification

- **Causal**: confidence ≥ 0.6, robust, identifiable via backdoor criterion
- **Correlational**: confidence < 0.4 or not robust
- **Inconclusive**: insufficient evidence for definitive classification

## Graph Persistence & Export

```python
from causaltarget import save_graph, load_graph

# Save to JSON (default: ~/.causaltarget/graphs/)
path = save_graph(graph)

# Export for Cytoscape
save_graph(graph, fmt="cytoscape")

# Export for Gephi
save_graph(graph, fmt="gexf")

# Reload later
graph = load_graph(path)
```

When running with Docker, graphs are also persisted to PostgreSQL
for querying and sharing across sessions.

## Project Structure

```
CausalTarget/
├── main.py                     # Entry point
├── pyproject.toml              # Dependencies, build config & CLI scripts
├── LICENSE                     # MIT License
├── docker-compose.yml          # Redis + PostgreSQL + API
├── Dockerfile
├── init/init.sql               # Database schema
│
├── causaltarget/               # Public re-export package (pip install)
│   └── __init__.py             # `from causaltarget import run_pipeline`
│
├── modules/causal_target/
│   ├── models.py               # Pydantic data models
│   ├── graph_builder.py        # Parallel multi-source graph assembly
│   ├── identifier.py           # Causal inference (d-separation, triangulation)
│   ├── scorer.py               # 6D composite scoring (configurable weights)
│   ├── pipeline.py             # End-to-end orchestration
│   ├── report.py               # Interactive HTML reports
│   ├── api.py                  # FastAPI REST service
│   ├── cache.py                # File/Redis caching layer
│   ├── persistence.py          # Graph save/load/export
│   ├── admet.py                # Multi-rule ADMET prediction
│   ├── __main__.py             # Typer CLI
│   ├── data_sources/
│   │   ├── monarch.py          # Monarch Initiative client
│   │   ├── open_targets.py     # Open Targets GraphQL
│   │   ├── kegg.py             # KEGG REST
│   │   ├── reactome.py         # Reactome Content Service
│   │   ├── string_db.py        # STRING batch PPI queries
│   │   ├── uniprot.py          # UniProt metadata enrichment
│   │   └── pdb.py              # RCSB PDB structure search
│   ├── templates/
│   │   └── report.html         # Jinja2 report template
│   └── tests/                  # 108 tests (pytest)
│
├── modules/genmol/             # Molecular generation (VAE)
├── modules/molscreen/          # Drug-likeness screening
├── modules/dockbot/            # Molecular docking (Vina)
└── modules/mirrorfold/         # Protein structure prediction
```

## Testing

```bash
# Run all causal_target tests
uv run python -m pytest modules/causal_target/tests/ -v

# Run with coverage
uv run python -m pytest modules/causal_target/tests/ --cov=modules.causal_target
```

## Configuration

### Environment Variables

| Variable                       | Default     | Description                      |
|--------------------------------|-------------|----------------------------------|
| `CAUSALTARGET_CACHE_BACKEND`   | `file`      | `file` or `redis`                |
| `CAUSALTARGET_SEED`            | `42`        | RNG seed for reproducibility     |
| `REDIS_URL`                    | —           | Redis connection URL             |
| `DATABASE_URL`                 | —           | PostgreSQL connection URL        |


### Score Weights

Customise via environment or the Python API:

```python
from causaltarget import score_candidate

# Weights are configurable per-call
custom_weights = {
    "causal_confidence": 0.35,
    "binding_affinity": 0.20,
    "qed": 0.15,
    "sa": 0.10,
    "admet": 0.10,
    "novelty": 0.10,
}
```

## Citation

If you use CausalTarget in your research, please cite:

```bibtex
@software{causaltarget2026,
  title  = {CausalTarget: Causal Drug Target Discovery via Do-Calculus},
  author = {NeoForge Bio-AI Platform},
  year   = {2026},
  url    = {https://github.com/NeoForge/CausalTarget}
}
```

## License

MIT
