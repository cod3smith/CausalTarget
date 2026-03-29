"""
DockBot CLI
=============

Command-line interface for the DockBot docking pipeline.

Usage examples::

    # Prepare a protein target
    python -m modules.dockbot prepare-protein 1BNA --output-dir ./docking

    # Detect binding site from co-crystallised ligand
    python -m modules.dockbot binding-site ./docking/1BNA_clean.pdb --method ligand

    # Dock a single ligand
    python -m modules.dockbot dock 1BNA "CC(=O)Oc1ccccc1C(O)=O" --name aspirin

    # Screen a library
    python -m modules.dockbot screen 1BNA library.csv --output-dir results
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

try:
    import typer  # type: ignore[import-untyped]
except ImportError:
    print(
        "DockBot CLI requires typer.  Install it with: pip install typer",
        file=sys.stderr,
    )
    sys.exit(1)

app = typer.Typer(
    name="dockbot",
    help="🤖 DockBot — automated molecular docking pipeline.",
    add_completion=False,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dockbot")


# ── prepare-protein ─────────────────────────────────────────────────

@app.command()
def prepare_protein(
    pdb_id: str = typer.Argument(..., help="PDB ID to fetch and prepare."),
    output_dir: Path = typer.Option(
        Path("docking"), "--output-dir", "-o",
        help="Directory for output files.",
    ),
    keep_water: bool = typer.Option(
        False, "--keep-water", help="Keep crystallographic water molecules.",
    ),
):
    """Fetch a PDB structure and prepare it for docking."""
    from .protein_prep import prepare_protein as _prep

    result = _prep(pdb_id, output_dir=output_dir)

    typer.echo(f"\n✅  Protein prepared: {result['pdb_id']}")
    typer.echo(f"   Title       : {result.get('title', 'N/A')}")
    typer.echo(f"   Residues    : {result.get('residues', 'N/A')}")
    typer.echo(f"   Resolution  : {result.get('resolution', 'N/A')}")
    typer.echo(f"   Clean PDB   : {result['clean_pdb']}")
    typer.echo(f"   PDBQT       : {result['pdbqt']}")

    if result.get("ligands"):
        typer.echo(f"   Ligands     : {', '.join(result['ligands'])}")


# ── binding-site ────────────────────────────────────────────────────

@app.command()
def binding_site(
    pdb_path: Path = typer.Argument(..., help="Path to a PDB file."),
    method: str = typer.Option(
        "ligand", "--method", "-m",
        help="Detection method: ligand, fpocket, or manual.",
    ),
    ligand: Optional[str] = typer.Option(
        None, "--ligand", "-l",
        help="Three-letter ligand residue name (for ligand method).",
    ),
    pocket_rank: int = typer.Option(
        1, "--pocket-rank", help="fpocket pocket rank (1 = best).",
    ),
    center: Optional[str] = typer.Option(
        None, "--center",
        help="Manual centre as 'x,y,z' (in Å).",
    ),
    size: str = typer.Option(
        "25,25,25", "--size",
        help="Box size as 'sx,sy,sz' (in Å).",
    ),
    padding: float = typer.Option(
        10.0, "--padding", help="Extra padding around detected site (Å).",
    ),
):
    """Detect or specify a binding site on the protein."""
    from .binding_site import detect_binding_site

    center_tuple = None
    if center:
        parts = [float(x.strip()) for x in center.split(",")]
        center_tuple = (parts[0], parts[1], parts[2])

    size_parts = [float(x.strip()) for x in size.split(",")]
    size_tuple = (size_parts[0], size_parts[1], size_parts[2])

    site = detect_binding_site(
        pdb_path,
        method=method,
        ligand_resname=ligand,
        pocket_rank=pocket_rank,
        center=center_tuple,
        size=size_tuple,
        padding=padding,
    )

    typer.echo(f"\n📍  Binding site ({site.method.value}):")
    typer.echo(f"   Centre : ({site.center_x:.1f}, {site.center_y:.1f}, {site.center_z:.1f})")
    typer.echo(f"   Size   : ({site.size_x:.1f}, {site.size_y:.1f}, {site.size_z:.1f})")
    if site.residues:
        typer.echo(f"   Residues: {', '.join(site.residues[:10])}")

    # Write site to JSON for re-use
    site_json = pdb_path.parent / f"{pdb_path.stem}_site.json"
    site_json.write_text(site.model_dump_json(indent=2))
    typer.echo(f"   Saved → {site_json}")


# ── dock ────────────────────────────────────────────────────────────

@app.command()
def dock_cmd(
    pdb_id: str = typer.Argument(..., help="PDB ID or path to receptor PDBQT."),
    smiles: str = typer.Argument(..., help="SMILES string of the ligand."),
    name: str = typer.Option("ligand", "--name", "-n", help="Ligand name."),
    site_json: Optional[Path] = typer.Option(
        None, "--site", "-s",
        help="Path to binding-site JSON. Auto-detected if omitted.",
    ),
    method: str = typer.Option(
        "ligand", "--method", "-m",
        help="Binding-site detection method if no --site provided.",
    ),
    exhaustiveness: int = typer.Option(32, "--exhaustiveness", "-e"),
    n_poses: int = typer.Option(9, "--n-poses"),
    output_dir: Path = typer.Option(
        Path("docking"), "--output-dir", "-o",
    ),
):
    """Dock a single ligand against a protein target."""
    from .binding_site import detect_binding_site
    from .docker import VinaConfig, dock
    from .ligand_prep import prepare_ligand_pdbqt
    from .models import BindingSite
    from .protein_prep import prepare_protein as _prep

    # 1. Receptor
    receptor_pdbqt = Path(pdb_id)
    if not receptor_pdbqt.exists():
        # Treat as PDB ID
        result = _prep(pdb_id, output_dir=output_dir)
        receptor_pdbqt = Path(result["pdbqt"])
        clean_pdb = Path(result["clean_pdb"])
    else:
        clean_pdb = receptor_pdbqt.with_suffix(".pdb")

    # 2. Binding site
    if site_json and site_json.exists():
        site = BindingSite.model_validate_json(site_json.read_text())
    else:
        site = detect_binding_site(clean_pdb, method=method)

    # 3. Ligand
    typer.echo(f"\n🧪  Preparing ligand: {name} ({smiles})")
    mol, lig_pdbqt = prepare_ligand_pdbqt(smiles, name=name, output_dir=output_dir)
    if mol is None:
        typer.echo("❌  Ligand preparation failed.", err=True)
        raise typer.Exit(1)

    # 4. Dock
    typer.echo("🔬  Docking...")
    config = VinaConfig(exhaustiveness=exhaustiveness, n_poses=n_poses)
    result = dock(
        receptor_pdbqt=receptor_pdbqt,
        ligand_pdbqt=lig_pdbqt,
        site=site,
        config=config,
        ligand_name=name,
        ligand_smiles=smiles,
    )

    # 5. Output
    typer.echo(f"\n✅  Docking complete — {len(result.poses)} poses:")
    for pose in result.poses:
        typer.echo(
            f"   Rank {pose.rank:2d}  |  "
            f"{pose.affinity_kcal_mol:6.1f} kcal/mol  |  "
            f"RMSD lb={pose.rmsd_lb:.2f}  ub={pose.rmsd_ub:.2f}"
        )

    # Save result
    result_json = output_dir / f"{name}_result.json"
    result_json.parent.mkdir(parents=True, exist_ok=True)
    result_json.write_text(result.model_dump_json(indent=2))
    typer.echo(f"\n   Saved → {result_json}")


# ── screen ──────────────────────────────────────────────────────────

@app.command()
def screen(
    pdb_id: str = typer.Argument(..., help="PDB ID or receptor PDBQT path."),
    library: Path = typer.Argument(
        ..., help="CSV file with 'smiles' and 'name' columns.",
    ),
    site_json: Optional[Path] = typer.Option(
        None, "--site", "-s",
        help="Binding-site JSON.",
    ),
    method: str = typer.Option("ligand", "--method", "-m"),
    exhaustiveness: int = typer.Option(16, "--exhaustiveness", "-e"),
    workers: int = typer.Option(0, "--workers", "-w", help="Parallel workers (0=auto)."),
    output_dir: Path = typer.Option(
        Path("screening_results"), "--output-dir", "-o",
    ),
):
    """Screen a library of compounds against a protein target."""
    import csv

    from .binding_site import detect_binding_site
    from .docker import VinaConfig
    from .models import BindingSite
    from .parallel import ScreeningJob, run_screening
    from .protein_prep import prepare_protein as _prep
    from .report import generate_report

    # 1. Receptor
    receptor_pdbqt = Path(pdb_id)
    if not receptor_pdbqt.exists():
        result = _prep(pdb_id, output_dir=output_dir)
        receptor_pdbqt = Path(result["pdbqt"])
        clean_pdb = Path(result["clean_pdb"])
    else:
        clean_pdb = receptor_pdbqt.with_suffix(".pdb")

    # 2. Binding site
    if site_json and site_json.exists():
        site = BindingSite.model_validate_json(site_json.read_text())
    else:
        site = detect_binding_site(clean_pdb, method=method)

    # 3. Load library
    ligands: list[tuple[str, str]] = []
    with library.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row.get("smiles", row.get("SMILES", "")).strip()
            nm = row.get("name", row.get("Name", row.get("NAME", ""))).strip()
            if smi:
                ligands.append((smi, nm or smi[:30]))

    typer.echo(f"\n📚  Loaded {len(ligands)} ligands from {library}")

    # 4. Screen
    job = ScreeningJob(
        receptor_pdbqt=receptor_pdbqt,
        binding_site=site,
        ligands=ligands,
        output_dir=output_dir,
        config=VinaConfig(exhaustiveness=exhaustiveness),
        n_workers=workers,
    )

    screening_result = run_screening(job)

    typer.echo(
        f"\n✅  Screening complete: {screening_result.completed}/"
        f"{screening_result.total_ligands} in "
        f"{screening_result.elapsed_seconds:.0f}s"
    )

    # 5. Generate report
    report_path = output_dir / "report.html"
    generate_report(
        screening_result,
        output_path=report_path,
        protein_name=pdb_id,
        pdb_id=pdb_id,
    )
    typer.echo(f"📄  Report → {report_path}")


# ── Entry point ─────────────────────────────────────────────────────

def main():
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
