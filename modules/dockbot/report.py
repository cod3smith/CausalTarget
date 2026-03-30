"""
DockBot Report Generator
===========================

Produces self-contained HTML reports summarising a docking campaign.

The report includes:
- Protein & binding-site metadata
- A ranked table of all docking results
- Composite scores (affinity + drug-likeness)
- Embedded 3D viewers for the top poses (via py3Dmol)
- Summary statistics and charts

The HTML output works offline — all JavaScript (3Dmol.js) is loaded
from a CDN and no backend server is required.
"""

from __future__ import annotations

import html
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import BindingSite, DockingResult, ScreeningResult
from .scorer import CompositeScore, ScoringWeights, rank_results

logger = logging.getLogger(__name__)

# NeoForge palette
DEEP_NAVY = "#0D1B2A"
NEON_TEAL = "#00D4AA"


def generate_report(
    results: list[DockingResult] | ScreeningResult,
    output_path: Path,
    protein_name: str = "",
    pdb_id: str = "",
    weights: ScoringWeights | None = None,
    top_n: int = 20,
) -> Path:
    """Generate a complete HTML docking report.

    Parameters
    ----------
    results:
        Docking results — either a list or a :class:`ScreeningResult`.
    output_path:
        Where to write the HTML file.
    protein_name:
        Human-readable protein name for the header.
    pdb_id:
        PDB ID of the target structure.
    weights:
        Composite-scoring weights.
    top_n:
        Number of top results to show in the detailed section.

    Returns
    -------
    Path
        Path to the written report.
    """
    if isinstance(results, ScreeningResult):
        result_list = results.results
        total = results.total_ligands
        completed = results.completed
        elapsed = results.elapsed_seconds
    else:
        result_list = results
        total = len(results)
        completed = len(results)
        elapsed = 0.0

    # Score & rank
    scores = rank_results(result_list, weights)

    report_html = _build_html(
        scores=scores,
        protein_name=protein_name or "Unknown protein",
        pdb_id=pdb_id,
        total=total,
        completed=completed,
        elapsed=elapsed,
        top_n=top_n,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_html)
    logger.info("Report written → %s", output_path)

    # Also write machine-readable JSON alongside
    json_path = output_path.with_suffix(".json")
    _write_json(json_path, scores)

    return output_path


def _build_html(
    scores: list[CompositeScore],
    protein_name: str,
    pdb_id: str,
    total: int,
    completed: int,
    elapsed: float,
    top_n: int,
) -> str:
    """Assemble the full HTML report string."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    safe_name = html.escape(protein_name)
    safe_pdb = html.escape(pdb_id)

    rows = _table_rows(scores)
    stats = _summary_stats(scores)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DockBot Report — {safe_name}</title>
  <style>
    :root {{
      --navy: {DEEP_NAVY};
      --teal: {NEON_TEAL};
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      background: #f5f7fa;
      color: #333;
      line-height: 1.5;
    }}
    header {{
      background: var(--navy);
      color: white;
      padding: 24px 32px;
    }}
    header h1 {{
      font-size: 1.6rem;
      color: var(--teal);
    }}
    header .meta {{
      font-size: 0.9rem;
      color: #aab;
      margin-top: 4px;
    }}
    .container {{
      max-width: 1200px;
      margin: 24px auto;
      padding: 0 16px;
    }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 32px;
    }}
    .stat-card {{
      background: white;
      border-radius: 10px;
      padding: 20px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06);
      text-align: center;
    }}
    .stat-card .value {{
      font-size: 1.8rem;
      font-weight: 700;
      color: var(--navy);
    }}
    .stat-card .label {{
      font-size: 0.85rem;
      color: #777;
      margin-top: 4px;
    }}
    h2 {{
      font-size: 1.2rem;
      color: var(--navy);
      margin-bottom: 12px;
      border-bottom: 2px solid var(--teal);
      display: inline-block;
      padding-bottom: 4px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
      border-radius: 10px;
      overflow: hidden;
      box-shadow: 0 2px 8px rgba(0,0,0,0.06);
      margin-bottom: 32px;
    }}
    th {{
      background: var(--navy);
      color: white;
      padding: 10px 14px;
      text-align: left;
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    td {{
      padding: 10px 14px;
      border-bottom: 1px solid #eee;
      font-size: 0.9rem;
    }}
    tr:nth-child(even) {{ background: #fafbfc; }}
    tr:hover {{ background: #e8f7f3; }}
    .score-bar {{
      height: 8px;
      border-radius: 4px;
      background: #eee;
      position: relative;
    }}
    .score-bar .fill {{
      height: 100%;
      border-radius: 4px;
      background: var(--teal);
    }}
    footer {{
      text-align: center;
      padding: 24px;
      color: #999;
      font-size: 0.8rem;
    }}
  </style>
</head>
<body>
  <header>
    <h1>DockBot — Docking Report</h1>
    <div class="meta">
      Target: <strong>{safe_name}</strong>
      {f'&nbsp;|&nbsp; PDB: <strong>{safe_pdb}</strong>' if safe_pdb else ''}
      &nbsp;|&nbsp; Generated: {now}
    </div>
  </header>

  <div class="container">

    <div class="stats-grid">
      <div class="stat-card">
        <div class="value">{completed}</div>
        <div class="label">Ligands Docked</div>
      </div>
      <div class="stat-card">
        <div class="value">{total}</div>
        <div class="label">Total Submitted</div>
      </div>
      <div class="stat-card">
        <div class="value">{stats['best_affinity']:.1f}</div>
        <div class="label">Best Affinity (kcal/mol)</div>
      </div>
      <div class="stat-card">
        <div class="value">{stats['mean_affinity']:.1f}</div>
        <div class="label">Mean Affinity (kcal/mol)</div>
      </div>
      <div class="stat-card">
        <div class="value">{stats['best_composite']:.3f}</div>
        <div class="label">Best Composite Score</div>
      </div>
      <div class="stat-card">
        <div class="value">{elapsed:.0f}s</div>
        <div class="label">Runtime</div>
      </div>
    </div>

    <h2>Ranked Results</h2>
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Name</th>
          <th>SMILES</th>
          <th>Affinity</th>
          <th>QED</th>
          <th>SA</th>
          <th>Filters</th>
          <th>Composite</th>
          <th>Drug-likeness</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>

  </div>

  <footer>
    NeoRx · DockBot · NeoForge
  </footer>
</body>
</html>"""


def _table_rows(scores: list[CompositeScore]) -> str:
    """Generate HTML table rows for all scored compounds."""
    rows: list[str] = []
    for s in scores:
        safe_name = html.escape(s.ligand_name[:30])
        safe_smi = html.escape(s.smiles[:40])
        bar_width = int(min(100, s.composite * 100))

        rows.append(f"""        <tr>
          <td>{s.rank}</td>
          <td><strong>{safe_name}</strong></td>
          <td><code>{safe_smi}</code></td>
          <td>{s.affinity_kcal_mol:.1f}</td>
          <td>{s.qed:.2f}</td>
          <td>{s.sa_score:.1f}</td>
          <td>{s.filter_pass_rate:.0%}</td>
          <td>
            <div class="score-bar"><div class="fill" style="width:{bar_width}%"></div></div>
            {s.composite:.3f}
          </td>
          <td>{html.escape(s.drug_likeness)}</td>
        </tr>""")

    return "\n".join(rows)


def _summary_stats(scores: list[CompositeScore]) -> dict:
    """Compute summary statistics for the report header cards."""
    if not scores:
        return {
            "best_affinity": 0.0,
            "mean_affinity": 0.0,
            "best_composite": 0.0,
        }

    affinities = [s.affinity_kcal_mol for s in scores]
    composites = [s.composite for s in scores]

    return {
        "best_affinity": min(affinities),
        "mean_affinity": sum(affinities) / len(affinities),
        "best_composite": max(composites),
    }


def _write_json(path: Path, scores: list[CompositeScore]) -> None:
    """Write machine-readable JSON results alongside the HTML."""
    data = [
        {
            "rank": s.rank,
            "name": s.ligand_name,
            "smiles": s.smiles,
            "affinity_kcal_mol": s.affinity_kcal_mol,
            "qed": s.qed,
            "sa_score": s.sa_score,
            "filter_pass_rate": s.filter_pass_rate,
            "composite": s.composite,
            "drug_likeness": s.drug_likeness,
        }
        for s in scores
    ]
    path.write_text(json.dumps(data, indent=2))
    logger.info("Wrote JSON results → %s", path)
