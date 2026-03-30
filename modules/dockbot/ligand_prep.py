"""
DockBot Ligand Preparation
============================

Converts SMILES strings into 3D molecular structures ready for docking.

The pipeline
------------
1. **Parse SMILES** → 2D molecular graph (atoms + bonds).
2. **Add hydrogens** — explicit H atoms needed for force-field calculations.
3. **Generate 3D conformer** — use the ETKDG (Experimental-Torsion Knowledge
   Distance Geometry) algorithm to produce a realistic 3D geometry.
4. **Energy minimise** — relax the structure using the MMFF94 force field
   to remove steric clashes and find a local energy minimum.
5. **Convert to PDBQT** — add partial charges and AutoDock atom types
   using Meeko, the format Vina requires.

Why ETKDG?
----------
ETKDG combines distance geometry (random embedding in 3D space subject
to bond-length / angle constraints) with torsion-angle preferences
learned from the Cambridge Structural Database.  It produces much
better starting geometries than older distance geometry methods.

Why MMFF94?
-----------
The Merck Molecular Force Field (MMFF94) is parameterised for drug-like
organic molecules.  It balances accuracy and speed for the geometry
optimisation step.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers

logger = logging.getLogger(__name__)


def prepare_ligand(
    smiles: str,
    name: str = "",
    num_conformers: int = 1,
    minimize: bool = True,
    max_iterations: int = 500,
) -> Optional[Chem.Mol]:
    """Prepare a ligand from SMILES: parse → 3D → minimise.

    Parameters
    ----------
    smiles:
        SMILES string of the ligand.
    name:
        Optional molecule name (stored as a property).
    num_conformers:
        Number of 3D conformers to generate.  The lowest-energy one is
        kept after minimisation.
    minimize:
        Whether to energy-minimise with MMFF94.
    max_iterations:
        Maximum force-field optimisation iterations.

    Returns
    -------
    Mol | None
        RDKit Mol with 3D coordinates, or ``None`` on failure.
    """
    # 1. Parse SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning("Invalid SMILES: '%s'", smiles)
        return None

    try:
        # 2. Add explicit hydrogens (needed for force field + PDBQT)
        mol = Chem.AddHs(mol)

        # 3. Generate 3D conformer(s) using ETKDG v3
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = 42  # reproducibility
        params.numThreads = 0   # use all available cores

        conf_ids = rdDistGeom.EmbedMultipleConfs(
            mol, numConfs=num_conformers, params=params
        )

        if not conf_ids:
            logger.warning("3D embedding failed for '%s' — trying fallback.", smiles)
            # Fallback: allow random coordinates as seed
            params.useRandomCoords = True
            conf_ids = rdDistGeom.EmbedMultipleConfs(
                mol, numConfs=num_conformers, params=params
            )
            if not conf_ids:
                logger.error("3D embedding failed completely for '%s'.", smiles)
                return None

        # 4. Energy minimise with MMFF94
        if minimize:
            best_conf_id = _minimize_conformers(mol, conf_ids, max_iterations)
            if best_conf_id is not None and num_conformers > 1:
                # Keep only the best conformer
                _keep_conformer(mol, best_conf_id)

        # 5. Set metadata
        if name:
            mol.SetProp("_Name", name)
        mol.SetProp("SMILES", Chem.MolToSmiles(Chem.RemoveHs(mol)))

        return mol

    except Exception:
        logger.exception("Ligand preparation failed for '%s'.", smiles)
        return None


def _minimize_conformers(
    mol: Chem.Mol,
    conf_ids: list[int],
    max_iterations: int,
) -> int | None:
    """Minimise all conformers and return the ID of the lowest-energy one.

    Uses MMFF94 if parameters are available, otherwise falls back to UFF.
    """
    energies: list[tuple[int, float]] = []

    # Try MMFF94 first
    if rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol):
        for cid in conf_ids:
            result = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
                mol, numThreads=0, maxIters=max_iterations
            )
            # result is list of (converged, energy) tuples
            for i, (converged, energy) in enumerate(result):
                energies.append((i, energy))
    else:
        # Fallback to UFF (Universal Force Field)
        logger.debug("MMFF94 params not available — falling back to UFF.")
        for cid in conf_ids:
            try:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
                if ff is not None:
                    ff.Minimize(maxIts=max_iterations)
                    energies.append((cid, ff.CalcEnergy()))
            except Exception:
                pass

    if not energies:
        return None if not conf_ids else conf_ids[0]

    # Return the conformer with the lowest energy
    best_cid, best_energy = min(energies, key=lambda x: x[1])
    logger.debug("Best conformer: %d (energy=%.2f kcal/mol).", best_cid, best_energy)
    return best_cid


def _keep_conformer(mol: Chem.Mol, conf_id: int) -> None:
    """Remove all conformers except *conf_id*."""
    all_ids = [c.GetId() for c in mol.GetConformers()]
    for cid in all_ids:
        if cid != conf_id:
            mol.RemoveConformer(cid)


def mol_to_pdbqt_string(mol: Chem.Mol) -> str:
    """Convert an RDKit Mol (with 3D coords) to a PDBQT string using Meeko.

    Parameters
    ----------
    mol:
        RDKit Mol with at least one 3D conformer and explicit hydrogens.

    Returns
    -------
    str
        PDBQT-format string ready for Vina.

    Raises
    ------
    RuntimeError
        If Meeko is not installed.
    """
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy  # type: ignore[import-untyped]

        preparator = MoleculePreparation()
        mol_setup_list = preparator.prepare(mol)

        if not mol_setup_list:
            raise RuntimeError("Meeko preparation returned empty result.")

        pdbqt_strings = []
        for mol_setup in mol_setup_list:
            pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setup)
            if is_ok:
                pdbqt_strings.append(pdbqt_string)
            else:
                logger.warning("Meeko PDBQT write warning: %s", error_msg)
                pdbqt_strings.append(pdbqt_string)

        return "\n".join(pdbqt_strings)

    except ImportError as exc:
        logger.warning(
            "Meeko import failed: %s — using basic PDB→PDBQT fallback. "
            "Ensure meeko and its dependencies are installed: "
            "pip install meeko gemmi",
            exc,
        )
        return _basic_mol_to_pdbqt(mol)


def _basic_mol_to_pdbqt(mol: Chem.Mol) -> str:
    """Fallback PDB→PDBQT conversion when Meeko is not available.

    Writes a basic PDBQT by converting to PDB block and appending
    AutoDock atom types.  Not as accurate as Meeko but functional for
    testing.
    """
    pdb_block = Chem.MolToPDBBlock(mol)
    if not pdb_block:
        return ""

    _AD_TYPE_MAP = {
        "C": "C", "N": "NA", "O": "OA", "S": "SA", "H": "HD",
        "F": "F", "P": "P", "Cl": "Cl", "Br": "Br", "I": "I",
    }

    lines: list[str] = []
    for line in pdb_block.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            element = line[76:78].strip() if len(line) > 76 else "C"
            ad_type = _AD_TYPE_MAP.get(element, "C")
            pdbqt_line = f"{line[:54]}  0.000 {line[54:66]}    {ad_type:>2s}"
            lines.append(pdbqt_line)
        elif line.startswith(("END", "TER")):
            lines.append(line)

    return "\n".join(lines)


def mol_to_pdbqt_file(mol: Chem.Mol, output_path: Path) -> Path:
    """Write an RDKit Mol to a PDBQT file.

    Parameters
    ----------
    mol:
        RDKit Mol with 3D coords.
    output_path:
        Output file path.

    Returns
    -------
    Path
        The written file path.
    """
    pdbqt = mol_to_pdbqt_string(mol)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(pdbqt)
    logger.info("Wrote ligand PDBQT → %s", output_path)
    return output_path


def prepare_ligand_pdbqt(
    smiles: str,
    name: str = "",
    output_dir: Path | None = None,
) -> tuple[Chem.Mol | None, str]:
    """Full ligand preparation: SMILES → 3D → minimise → PDBQT string.

    Convenience function that combines :func:`prepare_ligand` and
    :func:`mol_to_pdbqt_string`.

    Parameters
    ----------
    smiles:
        SMILES string.
    name:
        Optional molecule name.
    output_dir:
        If provided, also write the PDBQT to a file in this directory.

    Returns
    -------
    tuple[Mol | None, str]
        ``(mol_with_3d_coords, pdbqt_string)`` — both ``None``/empty on
        failure.
    """
    mol = prepare_ligand(smiles, name=name)
    if mol is None:
        return None, ""

    pdbqt = mol_to_pdbqt_string(mol)

    if output_dir:
        safe_name = name or Chem.MolToSmiles(Chem.RemoveHs(mol))
        safe_name = "".join(c if c.isalnum() else "_" for c in safe_name)[:50]
        mol_to_pdbqt_file(mol, output_dir / f"{safe_name}.pdbqt")

    return mol, pdbqt
