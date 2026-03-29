"""
DockBot Binding-Site Detection
================================

Identifies the 3D pocket on a protein where a small-molecule ligand
is most likely to bind.  Knowing the binding site is essential for
molecular docking because AutoDock Vina (and most docking engines)
require a search box that defines the region of interest.

Three strategies are provided
-----------------------------

1. **Co-crystallised ligand** — if the PDB structure was solved with a
   ligand already bound, we extract it and compute a bounding box
   centred on the ligand's centre of mass with a user-defined padding.
   This is the gold-standard approach when structural data is
   available.

2. **fpocket** — a fast, geometry-based pocket-detection algorithm that
   identifies concave surface regions using Voronoi tessellation and
   α-spheres.  No ligand information is needed.

3. **Manual** — the user specifies the centre (x, y, z) and size of
   the search box directly.

All three return a :class:`BindingSite` model.
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from .models import BindingSite, BindingSiteMethod

logger = logging.getLogger(__name__)


# ─── Co-crystallised ligand ─────────────────────────────────────────

def from_ligand(
    pdb_path: str | Path,
    ligand_resname: str | None = None,
    padding: float = 10.0,
) -> BindingSite:
    """Derive a binding site from a co-crystallised ligand.

    Parameters
    ----------
    pdb_path:
        Path to the PDB file (may contain protein + ligand).
    ligand_resname:
        Three-letter residue name of the ligand (e.g. ``"ATP"``).
        If ``None``, the first non-water, non-standard residue is used.
    padding:
        Extra Ångströms added to each side of the ligand bounding box.

    Returns
    -------
    BindingSite
        Binding site centred on the ligand with the specified padding.
    """
    from Bio.PDB import PDBParser  # type: ignore[import-untyped]

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))

    STANDARD_RESIDUES = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
        "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
        "THR", "TRP", "TYR", "VAL", "HOH", "WAT",
    }

    ligand_atoms: list[np.ndarray] = []

    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip()

                if ligand_resname:
                    if resname != ligand_resname:
                        continue
                else:
                    if resname in STANDARD_RESIDUES:
                        continue

                for atom in residue:
                    ligand_atoms.append(atom.get_vector().get_array())

                if not ligand_resname:
                    ligand_resname = resname

        if ligand_atoms:
            break  # use first model only

    if not ligand_atoms:
        raise ValueError(
            f"No ligand found in {pdb_path}. "
            "Specify ligand_resname or use a different detection method."
        )

    coords = np.array(ligand_atoms)
    centre = coords.mean(axis=0).tolist()
    span = (coords.max(axis=0) - coords.min(axis=0) + 2 * padding).tolist()

    logger.info(
        "Binding site from ligand '%s': centre=(%.1f, %.1f, %.1f)  "
        "size=(%.1f, %.1f, %.1f)",
        ligand_resname, *centre, *span,
    )

    return BindingSite(
        center_x=round(centre[0], 2),
        center_y=round(centre[1], 2),
        center_z=round(centre[2], 2),
        size_x=round(span[0], 2),
        size_y=round(span[1], 2),
        size_z=round(span[2], 2),
        method=BindingSiteMethod.LIGAND,
        residues=_nearby_residues(structure, centre, max_dist=5.0),
    )


def _nearby_residues(
    structure,
    centre: list[float],
    max_dist: float = 5.0,
) -> list[str]:
    """List protein residues within *max_dist* Å of the centre."""
    STANDARD_AA = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
        "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
        "THR", "TRP", "TYR", "VAL",
    }
    centre_arr = np.array(centre)
    seen: set[str] = set()

    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip()
                if resname not in STANDARD_AA:
                    continue
                for atom in residue:
                    dist = np.linalg.norm(atom.get_vector().get_array() - centre_arr)
                    if dist <= max_dist:
                        rid = residue.get_id()
                        label = f"{chain.get_id()}:{resname}{rid[1]}"
                        seen.add(label)
                        break  # one hit per residue is enough
        break  # first model only

    return sorted(seen)


# ─── fpocket ────────────────────────────────────────────────────────

def from_fpocket(
    pdb_path: str | Path,
    pocket_rank: int = 1,
    padding: float = 5.0,
) -> BindingSite:
    """Detect a binding pocket with *fpocket*.

    fpocket is a very fast, open-source pocket-detection program based
    on Voronoi tessellation and α-sphere clustering.

    Parameters
    ----------
    pdb_path:
        Path to a cleaned PDB file (protein only, no ligands).
    pocket_rank:
        Rank of the desired pocket (1 = best-scoring pocket).
    padding:
        Extra Å added to the bounding box of the pocket α-spheres.

    Returns
    -------
    BindingSite

    Raises
    ------
    FileNotFoundError
        If *fpocket* is not on ``$PATH``.
    RuntimeError
        If fpocket fails or the requested pocket is not found.
    """
    pdb_path = Path(pdb_path).resolve()

    # fpocket writes output next to the input file, so work in a
    # temporary directory to avoid cluttering the project.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_pdb = Path(tmp) / pdb_path.name
        tmp_pdb.write_text(pdb_path.read_text())

        try:
            subprocess.run(
                ["fpocket", "-f", str(tmp_pdb)],
                capture_output=True,
                text=True,
                check=True,
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "fpocket not found.  Install it from "
                "https://github.com/Discngine/fpocket  or use a "
                "different binding-site method."
            )

        # fpocket creates a folder like  <name>_out/pockets/
        stem = tmp_pdb.stem
        pocket_dir = Path(tmp) / f"{stem}_out" / "pockets"
        if not pocket_dir.is_dir():
            raise RuntimeError("fpocket produced no output directory.")

        # Find the pocket PDB for the requested rank
        pocket_file = pocket_dir / f"pocket{pocket_rank}_atm.pdb"
        if not pocket_file.exists():
            available = sorted(pocket_dir.glob("pocket*_atm.pdb"))
            raise RuntimeError(
                f"Pocket rank {pocket_rank} not found.  "
                f"Available: {[p.name for p in available]}"
            )

        return _pocket_file_to_binding_site(pocket_file, padding)


def _pocket_file_to_binding_site(
    pocket_pdb: Path,
    padding: float,
) -> BindingSite:
    """Parse an fpocket pocket PDB and compute a bounding box."""
    coords: list[list[float]] = []

    for line in pocket_pdb.read_text().splitlines():
        if line.startswith(("ATOM", "HETATM")):
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append([x, y, z])

    if not coords:
        raise RuntimeError(f"No atoms found in {pocket_pdb}.")

    arr = np.array(coords)
    centre = arr.mean(axis=0).tolist()
    span = (arr.max(axis=0) - arr.min(axis=0) + 2 * padding).tolist()

    return BindingSite(
        center_x=round(centre[0], 2),
        center_y=round(centre[1], 2),
        center_z=round(centre[2], 2),
        size_x=round(span[0], 2),
        size_y=round(span[1], 2),
        size_z=round(span[2], 2),
        method=BindingSiteMethod.FPOCKET,
    )


# ─── Manual specification ───────────────────────────────────────────

def from_manual(
    center_x: float,
    center_y: float,
    center_z: float,
    size_x: float = 25.0,
    size_y: float = 25.0,
    size_z: float = 25.0,
) -> BindingSite:
    """Create a binding site from manually specified coordinates.

    Parameters
    ----------
    center_x, center_y, center_z:
        Centre of the docking search box in Ångströms.
    size_x, size_y, size_z:
        Dimensions of the search box (default 25 Å per axis).

    Returns
    -------
    BindingSite
    """
    return BindingSite(
        center_x=center_x,
        center_y=center_y,
        center_z=center_z,
        size_x=size_x,
        size_y=size_y,
        size_z=size_z,
        method=BindingSiteMethod.MANUAL,
    )


# ─── High-level convenience ────────────────────────────────────────

def detect_binding_site(
    pdb_path: str | Path,
    method: str = "ligand",
    ligand_resname: str | None = None,
    pocket_rank: int = 1,
    center: tuple[float, float, float] | None = None,
    size: tuple[float, float, float] = (25.0, 25.0, 25.0),
    padding: float = 10.0,
) -> BindingSite:
    """Auto-detect or specify a binding site.

    Parameters
    ----------
    pdb_path:
        Path to the PDB file.
    method:
        ``"ligand"`` (default), ``"fpocket"``, or ``"manual"``.
    ligand_resname:
        For the ligand method — the three-letter code.
    pocket_rank:
        For fpocket — which pocket to select (1 = top-ranked).
    center:
        For manual — ``(x, y, z)`` in Å.
    size:
        For manual — ``(sx, sy, sz)`` in Å.
    padding:
        Extra space around the detected pocket.

    Returns
    -------
    BindingSite
    """
    method_lower = method.lower().strip()

    if method_lower == "ligand":
        return from_ligand(pdb_path, ligand_resname=ligand_resname, padding=padding)

    if method_lower == "fpocket":
        return from_fpocket(pdb_path, pocket_rank=pocket_rank, padding=padding)

    if method_lower == "manual":
        if center is None:
            raise ValueError("Manual method requires 'center=(x, y, z)'.")
        return from_manual(*center, *size)

    raise ValueError(
        f"Unknown binding-site method '{method}'.  "
        "Choose 'ligand', 'fpocket', or 'manual'."
    )
