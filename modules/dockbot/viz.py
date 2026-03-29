"""
DockBot Visualisation
=======================

Generate interactive 3D visualisations of docking poses and summary
charts for screening campaigns.

Uses **py3Dmol** for 3D molecular rendering (renders in Jupyter
notebooks and exports standalone HTML) and **matplotlib** for 2D
summary figures.

NeoForge brand palette
----------------------
- Deep Navy:  ``#0D1B2A``  (backgrounds, text)
- Neon Teal:  ``#00D4AA``  (highlights, accents)
"""

from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Optional

from .models import BindingSite, DockingPose, DockingResult

logger = logging.getLogger(__name__)

# NeoForge palette
DEEP_NAVY = "#0D1B2A"
NEON_TEAL = "#00D4AA"
ACCENT_2 = "#1B98E0"  # complementary blue


def view_pose(
    receptor_pdb: str | Path,
    pose: DockingPose,
    site: BindingSite | None = None,
    width: int = 800,
    height: int = 600,
) -> object:
    """Render a docking pose interactively with py3Dmol.

    Parameters
    ----------
    receptor_pdb:
        Path to a PDB file of the receptor.
    pose:
        A docking pose (contains PDBQT of the ligand).
    site:
        If provided, draw the binding-site box.
    width, height:
        Widget dimensions.

    Returns
    -------
    py3Dmol.view
        Interactive 3D viewer (renders in Jupyter).
    """
    try:
        import py3Dmol  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "py3Dmol is required for 3D visualisation.  "
            "Install it with: pip install py3Dmol"
        )

    receptor_text = Path(receptor_pdb).read_text()

    viewer = py3Dmol.view(width=width, height=height)

    # Protein — cartoon + transparent surface
    viewer.addModel(receptor_text, "pdb")
    viewer.setStyle(
        {"model": 0},
        {"cartoon": {"color": DEEP_NAVY, "opacity": 0.85}},
    )
    viewer.addSurface(
        py3Dmol.VDW,
        {"opacity": 0.15, "color": DEEP_NAVY},
        {"model": 0},
    )

    # Ligand — sticks coloured in Neon Teal
    viewer.addModel(pose.pdbqt, "pdbqt")
    viewer.setStyle(
        {"model": 1},
        {
            "stick": {
                "radius": 0.25,
                "colorscheme": {"prop": "elem", "map": {"C": NEON_TEAL}},
            },
        },
    )

    # Binding-site box
    if site is not None:
        _add_box(viewer, site)

    viewer.zoomTo({"model": 1})
    viewer.setBackgroundColor("white")

    # Add label with affinity
    label = f"Rank {pose.rank}  |  {pose.affinity_kcal_mol:.1f} kcal/mol"
    viewer.addLabel(
        label,
        {
            "position": {"x": site.center_x, "y": site.center_y, "z": site.center_z}
            if site else {},
            "backgroundColor": DEEP_NAVY,
            "fontColor": NEON_TEAL,
            "fontSize": 12,
        },
    )

    return viewer


def _add_box(viewer, site: BindingSite) -> None:
    """Draw a wireframe box representing the binding site."""
    viewer.addBox(
        {
            "center": {
                "x": site.center_x,
                "y": site.center_y,
                "z": site.center_z,
            },
            "dimensions": {
                "w": site.size_x,
                "h": site.size_y,
                "d": site.size_z,
            },
            "color": NEON_TEAL,
            "opacity": 0.15,
            "wireframe": True,
        }
    )


def pose_to_html(
    receptor_pdb: str | Path,
    pose: DockingPose,
    site: BindingSite | None = None,
    title: str = "",
) -> str:
    """Generate a standalone HTML file with an embedded 3D viewer.

    Can be opened in any browser without a running notebook kernel.

    Parameters
    ----------
    receptor_pdb:
        Path to receptor PDB.
    pose:
        Docking pose.
    site:
        Optional binding-site box.
    title:
        Page title.

    Returns
    -------
    str
        Complete HTML document string.
    """
    viewer = view_pose(receptor_pdb, pose, site)
    inner_html = viewer._make_html()  # type: ignore[attr-defined]

    safe_title = html.escape(title or f"Pose rank {pose.rank}")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{safe_title}</title>
  <style>
    body {{
      margin: 0;
      background: {DEEP_NAVY};
      font-family: 'Inter', system-ui, sans-serif;
      color: white;
    }}
    h1 {{
      text-align: center;
      padding: 12px;
      color: {NEON_TEAL};
      font-size: 1.4rem;
    }}
    .info {{
      text-align: center;
      font-size: 0.95rem;
      color: #ccc;
    }}
    .viewer-container {{
      display: flex;
      justify-content: center;
      padding: 16px;
    }}
  </style>
</head>
<body>
  <h1>{safe_title}</h1>
  <p class="info">Affinity: {pose.affinity_kcal_mol:.1f} kcal/mol
    &nbsp;|&nbsp; RMSD lb: {pose.rmsd_lb:.2f}
    &nbsp;|&nbsp; RMSD ub: {pose.rmsd_ub:.2f}</p>
  <div class="viewer-container">
    {inner_html}
  </div>
</body>
</html>"""


def save_pose_html(
    receptor_pdb: str | Path,
    pose: DockingPose,
    output_path: Path,
    site: BindingSite | None = None,
    title: str = "",
) -> Path:
    """Write a standalone HTML visualisation to disk."""
    content = pose_to_html(receptor_pdb, pose, site, title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content)
    logger.info("Saved pose HTML → %s", output_path)
    return output_path


def screening_summary_figure(
    results: list[DockingResult],
    output_path: Path | None = None,
    top_n: int = 30,
):
    """Create a summary bar chart of top screening hits.

    Parameters
    ----------
    results:
        Docking results sorted by affinity.
    output_path:
        Where to save the figure (PNG).  If ``None``, returns the
        matplotlib Figure without saving.
    top_n:
        Number of top hits to display.

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        raise ImportError("matplotlib required for summary figures.")

    top = results[:top_n]
    names = [r.ligand_name[:20] for r in top]
    affinities = [
        r.poses[0].affinity_kcal_mol if r.poses else 0.0 for r in top
    ]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.35)))
    bars = ax.barh(names[::-1], affinities[::-1], color=NEON_TEAL, edgecolor=DEEP_NAVY)

    ax.set_xlabel("Vina Affinity (kcal/mol)", fontsize=11)
    ax.set_title("Top Docking Hits", fontsize=13, fontweight="bold", color=DEEP_NAVY)
    ax.axvline(x=-7.0, color="red", linestyle="--", alpha=0.5, label="–7 kcal/mol")
    ax.legend(fontsize=9)
    ax.invert_xaxis()

    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved summary figure → %s", output_path)

    return fig
