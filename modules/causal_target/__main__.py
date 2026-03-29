"""
CausalTarget CLI
=================

Typer-based command-line interface.

Commands
--------
- ``causaltarget run <disease>``     — Full pipeline
- ``causaltarget graph <disease>``   — Build graph only
- ``causaltarget identify <disease>``— Identify causal targets
- ``causaltarget report <disease>``  — Generate report from cache
- ``causaltarget serve``             — Start FastAPI server

Examples::

    $ causaltarget run HIV --top-n 5
    $ causaltarget graph "Type 2 Diabetes"
    $ causaltarget identify HIV --top-n 10
    $ causaltarget serve --port 8000
"""

from __future__ import annotations

import logging
import sys

import typer

app = typer.Typer(
    name="causaltarget",
    help="🧬 CausalTarget — Causal Drug Target Discovery Pipeline",
    add_completion=False,
)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(name)-25s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command()
def run(
    disease: str = typer.Argument(..., help="Disease name (e.g. HIV)"),
    top_n: int = typer.Option(5, "--top-n", "-n", help="Number of top targets"),
    candidates: int = typer.Option(100, "--candidates", "-c", help="Candidates per target"),
    no_docking: bool = typer.Option(False, "--no-docking", help="Skip molecular docking"),
    no_generation: bool = typer.Option(False, "--no-generation", help="Skip molecule generation"),
    no_report: bool = typer.Option(False, "--no-report", help="Skip report generation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Run the full CausalTarget pipeline."""
    _setup_logging(verbose)

    from .pipeline import run_pipeline

    typer.echo(f"\n🧬 CausalTarget Pipeline — {disease}")
    typer.echo("═" * 50)

    result = run_pipeline(
        disease=disease,
        top_n_targets=top_n,
        candidates_per_target=candidates,
        generate_molecules=not no_generation,
        run_docking=not no_docking,
        generate_report=not no_report,
    )

    typer.echo(f"\n✅ Pipeline complete: {result.job.status.value}")
    typer.echo(f"   Causal targets: {result.n_causal_targets}")
    typer.echo(f"   Candidates scored: {len(result.scored_candidates)}")

    if result.report_path:
        typer.echo(f"   Report: {result.report_path}")

    if result.scored_candidates:
        typer.echo("\n🏆 Top 5 Candidates:")
        for c in result.top_candidates[:5]:
            typer.echo(
                f"   #{c.rank}  {c.composite_score:.4f}  "
                f"{c.smiles[:40]}  ({c.target_protein_name})"
            )


@app.command()
def graph(
    disease: str = typer.Argument(..., help="Disease name"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build a disease causal knowledge graph."""
    _setup_logging(verbose)

    from .graph_builder import build_disease_graph

    typer.echo(f"\n🧬 Building causal graph for '{disease}'…")
    g = build_disease_graph(disease)

    typer.echo(f"\n✅ Graph built:")
    typer.echo(f"   Nodes: {len(g.nodes)} ({g.n_genes} genes, {g.n_proteins} proteins, {g.n_pathways} pathways)")
    typer.echo(f"   Edges: {len(g.edges)}")
    typer.echo(f"   Sources: {', '.join(g.sources_queried)}")


@app.command()
def identify(
    disease: str = typer.Argument(..., help="Disease name"),
    top_n: int = typer.Option(10, "--top-n", "-n"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Identify causal drug targets for a disease."""
    _setup_logging(verbose)

    from .graph_builder import build_disease_graph
    from .identifier import identify_causal_targets

    typer.echo(f"\n🧬 Identifying causal targets for '{disease}'…")
    g = build_disease_graph(disease)
    targets = identify_causal_targets(g, top_n=top_n)

    n_causal = sum(1 for t in targets if t.is_causal_target)
    typer.echo(f"\n✅ Found {n_causal} causal targets (of {len(targets)} evaluated):\n")

    for t in targets:
        icon = "✓" if t.is_causal_target else "✗"
        typer.echo(
            f"  {icon} {t.gene_name:12s} "
            f"conf={t.causal_confidence:.3f}  "
            f"robust={t.robustness_score:.3f}  "
            f"drug={t.druggability_score:.3f}  "
            f"[{t.classification.value}]"
        )


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port", "-p"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Start the CausalTarget FastAPI server."""
    _setup_logging(verbose)

    try:
        import uvicorn
        from .api import app as fastapi_app  # noqa: F811
        typer.echo(f"\n🧬 CausalTarget API server starting on {host}:{port}")
        uvicorn.run(fastapi_app, host=host, port=port)
    except ImportError:
        typer.echo("Error: uvicorn not installed. Run: pip install uvicorn", err=True)
        raise typer.Exit(1)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
