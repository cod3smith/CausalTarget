"""
CausalTarget Report Generator
===============================

Generates a self-contained HTML report summarising the pipeline
output.  The report includes:

1. **Executive Summary** — disease, targets found, top candidates
2. **Causal Graph Visualisation** — interactive node-link diagram
3. **Target Deep-Dive** — per-target causal reasoning
4. **Candidate Table** — ranked candidates with score breakdown
5. **Methodology** — explanation of causal reasoning approach

The report uses Jinja2 templating with inline CSS (NeoForge
brand: Deep Navy #0D1B2A, Neon Teal #00D4AA).  No external
dependencies are required to view it.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import (
    DiseaseGraph,
    CausalTargetResult,
    ScoredCandidate,
    NodeType,
)

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"


def generate_report(
    disease: str,
    graph: DiseaseGraph,
    causal_targets: list[CausalTargetResult],
    candidates: list[ScoredCandidate],
    output_dir: str | Path | None = None,
) -> tuple[str, str]:
    """Generate an HTML report for a pipeline run.

    Parameters
    ----------
    disease : str
        Disease name.
    graph : DiseaseGraph
        The assembled causal knowledge graph.
    causal_targets : list[CausalTargetResult]
        Identified causal targets.
    candidates : list[ScoredCandidate]
        Scored and ranked candidates.
    output_dir : str or Path, optional
        Directory to save the report.  Defaults to ``./reports/``.

    Returns
    -------
    tuple[str, str]
        (report_html, report_path)
    """
    if output_dir is None:
        output_dir = Path("reports")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build context for template
    context = _build_context(disease, graph, causal_targets, candidates)

    # Render template
    html = _render_template(context)

    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_disease = disease.lower().replace(" ", "_").replace("/", "_")
    filename = f"causaltarget_{safe_disease}_{timestamp}.html"
    filepath = output_dir / filename
    filepath.write_text(html, encoding="utf-8")

    logger.info("Report saved to %s.", filepath)
    return html, str(filepath)


def _build_context(
    disease: str,
    graph: DiseaseGraph,
    causal_targets: list[CausalTargetResult],
    candidates: list[ScoredCandidate],
) -> dict[str, Any]:
    """Build the template context dictionary."""
    n_causal = sum(1 for t in causal_targets if t.is_causal_target)
    n_correlational = sum(
        1 for t in causal_targets if not t.is_causal_target
    )
    top_candidates = candidates[:20]

    # Build graph data for visualisation
    graph_nodes = []
    for node in graph.nodes[:100]:  # Limit for performance
        graph_nodes.append({
            "id": node.node_id,
            "label": node.name,
            "type": node.node_type.value,
            "score": node.score,
        })

    graph_edges = []
    for edge in graph.edges[:200]:
        graph_edges.append({
            "from": edge.source_id,
            "to": edge.target_id,
            "type": edge.edge_type.value,
            "weight": edge.weight,
        })

    return {
        "disease": disease,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "graph": {
            "n_nodes": len(graph.nodes),
            "n_edges": len(graph.edges),
            "n_genes": graph.n_genes,
            "n_proteins": graph.n_proteins,
            "n_pathways": graph.n_pathways,
            "sources": graph.sources_queried,
            "nodes_json": graph_nodes,
            "edges_json": graph_edges,
        },
        "targets": {
            "total": len(causal_targets),
            "n_causal": n_causal,
            "n_correlational": n_correlational,
            "items": [
                {
                    "gene_name": t.gene_name,
                    "protein_name": t.protein_name,
                    "classification": t.classification.value,
                    "is_causal": t.is_causal_target,
                    "causal_confidence": t.causal_confidence,
                    "causal_effect": t.causal_effect,
                    "robustness": t.robustness_score,
                    "druggability": t.druggability_score,
                    "reasoning": t.reasoning,
                    "n_pathways": t.n_supporting_pathways,
                    "n_interactions": t.n_protein_interactions,
                    "pdb_ids": t.pdb_ids[:3],
                }
                for t in causal_targets
            ],
        },
        "candidates": {
            "total": len(candidates),
            "n_drug_like": sum(1 for c in candidates if c.is_drug_like),
            "n_novel": sum(1 for c in candidates if c.is_novel),
            "items": [
                {
                    "rank": c.rank,
                    "smiles": c.smiles,
                    "target": c.target_protein_name,
                    "composite": c.composite_score,
                    "causal_conf": c.causal_confidence,
                    "binding": c.binding_affinity,
                    "qed": c.qed_score,
                    "sa": c.sa_score,
                    "admet": c.admet_score,
                    "novelty": c.novelty_score,
                    "mw": c.molecular_weight,
                    "is_drug_like": c.is_drug_like,
                    "is_novel": c.is_novel,
                }
                for c in top_candidates
            ],
        },
    }


def _render_template(context: dict[str, Any]) -> str:
    """Render the HTML report template.

    Uses Jinja2 if available, otherwise falls back to a simple
    string-based template.
    """
    try:
        from jinja2 import Environment, FileSystemLoader
        template_path = TEMPLATE_DIR / "report.html"
        if template_path.exists():
            env = Environment(
                loader=FileSystemLoader(str(TEMPLATE_DIR)),
                autoescape=True,
            )
            template = env.get_template("report.html")
            return template.render(**context)
    except ImportError:
        logger.info("Jinja2 not installed. Using built-in template.")
    except Exception as e:
        logger.warning("Jinja2 rendering failed: %s. Using built-in.", e)

    # Fallback: built-in template
    return _builtin_template(context)


def _builtin_template(ctx: dict[str, Any]) -> str:
    """A self-contained HTML report without Jinja2."""
    targets_html = ""
    for t in ctx["targets"]["items"]:
        badge = "causal" if t["is_causal"] else t["classification"]
        badge_color = "#00D4AA" if t["is_causal"] else "#ff6b6b"
        targets_html += f"""
        <tr>
            <td><strong>{t["gene_name"]}</strong></td>
            <td>{t["protein_name"]}</td>
            <td><span style="background:{badge_color};color:#fff;
                padding:2px 8px;border-radius:4px;font-size:0.85em">
                {badge}</span></td>
            <td>{t["causal_confidence"]:.3f}</td>
            <td>{t["robustness"]:.3f}</td>
            <td>{t["druggability"]:.3f}</td>
            <td>{t["n_pathways"]}</td>
        </tr>"""

    candidates_html = ""
    for c in ctx["candidates"]["items"]:
        candidates_html += f"""
        <tr>
            <td>{c["rank"]}</td>
            <td style="font-family:monospace;font-size:0.85em">{c["smiles"][:40]}</td>
            <td>{c["target"]}</td>
            <td><strong>{c["composite"]:.4f}</strong></td>
            <td>{c["causal_conf"]:.3f}</td>
            <td>{c["binding"] if c["binding"] else "N/A"}</td>
            <td>{c["qed"]:.2f if c["qed"] else "N/A"}</td>
            <td>{"✓" if c["is_drug_like"] else "✗"}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CausalTarget Report — {ctx["disease"]}</title>
<style>
  :root {{ --navy: #0D1B2A; --teal: #00D4AA; --bg: #f8f9fa; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif;
         background: var(--bg); color: #333; }}
  .header {{ background: var(--navy); color: #fff; padding: 2rem;
             text-align: center; }}
  .header h1 {{ color: var(--teal); font-size: 2rem; }}
  .header p {{ opacity: 0.8; margin-top: 0.5rem; }}
  .container {{ max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }}
  .card {{ background: #fff; border-radius: 12px;
           box-shadow: 0 2px 12px rgba(0,0,0,0.08);
           padding: 1.5rem; margin-bottom: 1.5rem; }}
  .card h2 {{ color: var(--navy); margin-bottom: 1rem;
              border-bottom: 2px solid var(--teal);
              padding-bottom: 0.5rem; }}
  .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem; margin-bottom: 1rem; }}
  .stat {{ background: var(--navy); color: #fff; padding: 1rem;
           border-radius: 8px; text-align: center; }}
  .stat .value {{ font-size: 2rem; color: var(--teal); font-weight: bold; }}
  .stat .label {{ font-size: 0.85rem; opacity: 0.8; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th {{ background: var(--navy); color: #fff; padding: 0.75rem;
       text-align: left; }}
  td {{ padding: 0.75rem; border-bottom: 1px solid #e9ecef; }}
  tr:hover {{ background: #f1f3f5; }}
  .footer {{ text-align: center; padding: 2rem; color: #999;
             font-size: 0.85rem; }}
</style>
</head>
<body>
<div class="header">
  <h1>🧬 CausalTarget Report</h1>
  <p>{ctx["disease"]} — {ctx["timestamp"]}</p>
</div>
<div class="container">
  <div class="card">
    <h2>Executive Summary</h2>
    <div class="stats">
      <div class="stat">
        <div class="value">{ctx["graph"]["n_nodes"]}</div>
        <div class="label">Graph Nodes</div>
      </div>
      <div class="stat">
        <div class="value">{ctx["graph"]["n_edges"]}</div>
        <div class="label">Graph Edges</div>
      </div>
      <div class="stat">
        <div class="value">{ctx["targets"]["n_causal"]}</div>
        <div class="label">Causal Targets</div>
      </div>
      <div class="stat">
        <div class="value">{ctx["targets"]["n_correlational"]}</div>
        <div class="label">Correlational</div>
      </div>
      <div class="stat">
        <div class="value">{ctx["candidates"]["total"]}</div>
        <div class="label">Candidates Scored</div>
      </div>
      <div class="stat">
        <div class="value">{ctx["candidates"]["n_drug_like"]}</div>
        <div class="label">Drug-Like</div>
      </div>
    </div>
    <p>Data sources: {", ".join(ctx["graph"]["sources"])}</p>
  </div>

  <div class="card">
    <h2>Causal Target Analysis</h2>
    <table>
      <thead>
        <tr><th>Gene</th><th>Protein</th><th>Classification</th>
            <th>Causal Conf.</th><th>Robustness</th>
            <th>Druggability</th><th>Pathways</th></tr>
      </thead>
      <tbody>{targets_html}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>Top Drug Candidates</h2>
    <table>
      <thead>
        <tr><th>#</th><th>SMILES</th><th>Target</th>
            <th>Composite</th><th>Causal</th><th>Binding</th>
            <th>QED</th><th>Drug-like</th></tr>
      </thead>
      <tbody>{candidates_html}</tbody>
    </table>
  </div>

  <div class="card">
    <h2>Methodology</h2>
    <p>CausalTarget applies Pearl's do-calculus to distinguish genuine
    causal drug targets from correlational bystanders. The pipeline:</p>
    <ol style="margin:1rem 0 0 1.5rem">
      <li>Builds a causal knowledge graph from 7 biomedical databases</li>
      <li>Tests identifiability via the backdoor criterion</li>
      <li>Estimates causal effects using DoWhy</li>
      <li>Validates robustness through sensitivity analysis</li>
      <li>Generates novel molecules with GenMol (VAE)</li>
      <li>Screens for drug-likeness (MolScreen) and binding (DockBot)</li>
      <li>Ranks by composite score with causal confidence weighted highest (0.30)</li>
    </ol>
  </div>
</div>
<div class="footer">
  Generated by CausalTarget &middot; NeoForge Bio-AI Platform
</div>
</body>
</html>"""
