"""
DockBot Docking Engine
========================

Wraps AutoDock Vina to perform molecular docking.

What is molecular docking?
--------------------------
Molecular docking predicts how a small molecule (ligand) fits inside a
protein's binding pocket.  The algorithm:

1. Places the ligand inside the search box using stochastic global
   optimisation (Vina uses iterated local search with a Monte Carlo
   component).
2. Evaluates each pose with a scoring function that approximates the
   binding free energy (ΔG, in kcal/mol).  More negative = tighter
   binding.
3. Returns the top-ranked poses sorted by predicted affinity.

Vina's scoring function
-----------------------
Vina's empirical scoring function combines:

* **Gauss** — steric interactions (van der Waals attraction).
* **Repulsion** — penalises atomic overlap.
* **Hydrophobic** — favours burying hydrophobic groups.
* **Hydrogen bonding** — rewards H-bonds between ligand and protein.
* **Torsional penalty** — penalises ligand flexibility (entropy cost
  of freezing rotatable bonds).

The weights were fitted to reproduce experimental binding affinities
from the PDBbind database.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .models import BindingSite, DockingPose, DockingResult

logger = logging.getLogger(__name__)


@dataclass
class VinaConfig:
    """Configuration knobs for a Vina docking run.

    Attributes
    ----------
    exhaustiveness:
        How thoroughly to search.  Higher = slower but more accurate.
        Default 32 is a good balance; use 64–128 for production.
    n_poses:
        Maximum number of binding modes to return.
    energy_range:
        Maximum energy difference (kcal/mol) between the best pose and
        the worst returned pose.
    seed:
        Random seed for reproducibility.
    cpu:
        Number of CPU cores to use (0 = all available).
    """

    exhaustiveness: int = 32
    n_poses: int = 9
    energy_range: float = 3.0
    seed: int = 42
    cpu: int = 0


def dock(
    receptor_pdbqt: str | Path,
    ligand_pdbqt: str,
    site: BindingSite,
    config: VinaConfig | None = None,
    ligand_name: str = "",
    ligand_smiles: str = "",
) -> DockingResult:
    """Run AutoDock Vina and return parsed results.

    Parameters
    ----------
    receptor_pdbqt:
        Path to the receptor PDBQT file prepared by
        :mod:`dockbot.protein_prep`.
    ligand_pdbqt:
        PDBQT string of the ligand (from :mod:`dockbot.ligand_prep`).
    site:
        Binding site that defines the search box.
    config:
        Vina run parameters.  Defaults are sensible for quick screens.
    ligand_name:
        Human-readable name for the ligand.
    ligand_smiles:
        Canonical SMILES of the ligand.

    Returns
    -------
    DockingResult
        Contains predicted poses with affinities.
    """
    if config is None:
        config = VinaConfig()

    receptor_path = Path(receptor_pdbqt).resolve()

    # ── Try Vina Python bindings first ──────────────────────────────
    try:
        return _dock_with_vina_bindings(
            receptor_path, ligand_pdbqt, site, config,
            ligand_name, ligand_smiles,
        )
    except ImportError:
        logger.info("Vina Python bindings not available — trying CLI.")

    # ── Fallback to Vina CLI ────────────────────────────────────────
    try:
        return _dock_with_vina_cli(
            receptor_path, ligand_pdbqt, site, config,
            ligand_name, ligand_smiles,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "AutoDock Vina is not installed.  Install the Python "
            "bindings (`pip install vina`) or put `vina` on $PATH."
        )


# ── Vina Python bindings ────────────────────────────────────────────

def _dock_with_vina_bindings(
    receptor_path: Path,
    ligand_pdbqt: str,
    site: BindingSite,
    config: VinaConfig,
    ligand_name: str,
    ligand_smiles: str,
) -> DockingResult:
    """Dock using the ``vina`` Python package."""
    from vina import Vina  # type: ignore[import-untyped]

    v = Vina(sf_name="vina", cpu=config.cpu, seed=config.seed)

    v.set_receptor(str(receptor_path))

    # Vina Python API accepts PDBQT string via a temp file
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", mode="w", delete=False) as f:
        f.write(ligand_pdbqt)
        lig_tmp = f.name

    v.set_ligand_from_file(lig_tmp)

    v.compute_vina_maps(
        center=[site.center_x, site.center_y, site.center_z],
        box_size=[site.size_x, site.size_y, site.size_z],
    )

    v.dock(
        exhaustiveness=config.exhaustiveness,
        n_poses=config.n_poses,
        min_rmsd=1.0,
    )

    energies = v.energies(n_poses=config.n_poses)
    poses_pdbqt = v.poses(n_poses=config.n_poses)

    # Parse poses
    poses = _parse_vina_poses(poses_pdbqt, energies)

    # Clean up
    Path(lig_tmp).unlink(missing_ok=True)

    return DockingResult(
        ligand_name=ligand_name or "unknown",
        ligand_smiles=ligand_smiles,
        poses=poses,
        receptor_path=str(receptor_path),
        binding_site=site,
    )


def _parse_vina_poses(
    poses_pdbqt: str,
    energies: list | None = None,
) -> list[DockingPose]:
    """Split Vina multi-model PDBQT output into individual poses."""
    models: list[str] = []
    current_lines: list[str] = []

    for line in poses_pdbqt.splitlines():
        if line.startswith("MODEL"):
            current_lines = []
        elif line.startswith("ENDMDL"):
            models.append("\n".join(current_lines))
        else:
            current_lines.append(line)

    poses: list[DockingPose] = []
    for i, model_pdbqt in enumerate(models):
        affinity = 0.0
        rmsd_lb = 0.0
        rmsd_ub = 0.0

        if energies is not None and i < len(energies):
            row = energies[i]
            affinity = float(row[0])
            if len(row) > 1:
                rmsd_lb = float(row[1])
            if len(row) > 2:
                rmsd_ub = float(row[2])

        # Try to parse REMARK line for affinity if energies not given
        if energies is None:
            for line in model_pdbqt.splitlines():
                if "VINA RESULT" in line:
                    parts = line.split()
                    try:
                        idx = parts.index("RESULT") + 1
                        affinity = float(parts[idx])
                        if idx + 1 < len(parts):
                            rmsd_lb = float(parts[idx + 1])
                        if idx + 2 < len(parts):
                            rmsd_ub = float(parts[idx + 2])
                    except (ValueError, IndexError):
                        pass
                    break

        poses.append(
            DockingPose(
                rank=i + 1,
                affinity_kcal_mol=affinity,
                rmsd_lb=rmsd_lb,
                rmsd_ub=rmsd_ub,
                pdbqt=model_pdbqt,
            )
        )

    return poses


# ── Vina CLI fallback ───────────────────────────────────────────────

def _dock_with_vina_cli(
    receptor_path: Path,
    ligand_pdbqt: str,
    site: BindingSite,
    config: VinaConfig,
    ligand_name: str,
    ligand_smiles: str,
) -> DockingResult:
    """Dock using the ``vina`` command-line executable."""
    import subprocess

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        lig_in = tmp / "ligand.pdbqt"
        lig_out = tmp / "output.pdbqt"
        lig_in.write_text(ligand_pdbqt)

        cmd = [
            "vina",
            "--receptor", str(receptor_path),
            "--ligand", str(lig_in),
            "--out", str(lig_out),
            "--center_x", str(site.center_x),
            "--center_y", str(site.center_y),
            "--center_z", str(site.center_z),
            "--size_x", str(site.size_x),
            "--size_y", str(site.size_y),
            "--size_z", str(site.size_z),
            "--exhaustiveness", str(config.exhaustiveness),
            "--num_modes", str(config.n_poses),
            "--energy_range", str(config.energy_range),
            "--seed", str(config.seed),
        ]

        if config.cpu > 0:
            cmd.extend(["--cpu", str(config.cpu)])

        logger.info("Running Vina CLI: %s", " ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.debug("Vina stdout:\n%s", result.stdout)

        if not lig_out.exists():
            raise RuntimeError(f"Vina produced no output.  stderr:\n{result.stderr}")

        poses_pdbqt = lig_out.read_text()

    poses = _parse_vina_poses(poses_pdbqt)

    return DockingResult(
        ligand_name=ligand_name or "unknown",
        ligand_smiles=ligand_smiles,
        poses=poses,
        receptor_path=str(receptor_path),
        binding_site=site,
    )


# ── Score-only mode ─────────────────────────────────────────────────

def score_pose(
    receptor_pdbqt: str | Path,
    ligand_pdbqt: str,
    site: BindingSite,
) -> float:
    """Score a pre-docked pose without re-docking.

    Returns the Vina score in kcal/mol.  Requires the Vina Python
    bindings.
    """
    from vina import Vina  # type: ignore[import-untyped]

    receptor_path = Path(receptor_pdbqt).resolve()
    v = Vina(sf_name="vina")
    v.set_receptor(str(receptor_path))

    with tempfile.NamedTemporaryFile(suffix=".pdbqt", mode="w", delete=False) as f:
        f.write(ligand_pdbqt)
        lig_tmp = f.name

    v.set_ligand_from_file(lig_tmp)
    v.compute_vina_maps(
        center=[site.center_x, site.center_y, site.center_z],
        box_size=[site.size_x, site.size_y, site.size_z],
    )

    energy = v.score()
    Path(lig_tmp).unlink(missing_ok=True)
    return float(energy[0])
