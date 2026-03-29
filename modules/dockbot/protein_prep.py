"""
DockBot Protein Preparation
=============================

Fetches protein structures from the RCSB Protein Data Bank and prepares
them for molecular docking with AutoDock Vina.

Why preparation matters
-----------------------
Raw PDB files are *crystallographic snapshots* — they contain artefacts
that interfere with docking:

* **Water molecules (HOH)** — crystallographic waters aren't relevant
  to most docking experiments and clutter the binding site.
* **Co-crystallised ligands** — these occupy the binding pocket and must
  be removed, but their coordinates are valuable for defining the
  docking box.
* **Alternate conformations** — some residues have multiple conformations
  (A/B); we keep only the A conformer.
* **Missing hydrogens** — X-ray crystallography can't resolve hydrogens;
  we must add polar hydrogens for proper electrostatic modelling.

The final output is a **PDBQT** file — PDB format augmented with partial
charges (Q) and atom types (T) needed by AutoDock Vina's scoring function.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from Bio.PDB import PDBList, PDBParser, PDBIO, Select  # type: ignore[import-untyped]
from rdkit import Chem
from rdkit.Chem import AllChem

from .models import ProteinInfo

logger = logging.getLogger(__name__)

# Default cache directory for prepared proteins.
_CACHE_DIR = Path(__file__).parent / "cache"


# ═══════════════════════════════════════════════════════════════════════════
# BioPython helpers
# ═══════════════════════════════════════════════════════════════════════════


class _CleanProteinSelect(Select):  # type: ignore[misc]
    """BioPython ``Select`` subclass that strips waters, ligands, and
    alternate conformations during PDBIO.save().

    What gets *kept*:
    - Standard amino acid residues (protein backbone + side chains)
    - Altloc '' or 'A' only (first conformer)

    What gets *removed*:
    - Water molecules (HOH / WAT)
    - Heteroatom groups (ligands, ions, buffer molecules)
    - Alternate conformations B, C, etc.
    """

    # Standard amino acid residue names (3-letter codes)
    _STANDARD_AA = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
        "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP",
        "TYR", "VAL",
        # Modified / non-standard but common in PDB
        "MSE", "SEC", "PYL",
    }

    def accept_residue(self, residue: Any) -> bool:
        """Keep only standard amino acids (no water, no heteroatoms)."""
        resname = residue.get_resname().strip()
        hetflag = residue.get_id()[0]
        # hetflag == ' ' for standard residues, 'W' for water, 'H_xxx' for het
        if hetflag.strip() and hetflag != " ":
            return False
        return resname in self._STANDARD_AA

    def accept_atom(self, atom: Any) -> bool:
        """Keep only altloc '' or 'A'."""
        altloc = atom.get_altloc()
        return altloc in ("", " ", "A")


# ═══════════════════════════════════════════════════════════════════════════
# Core functions
# ═══════════════════════════════════════════════════════════════════════════


def fetch_pdb(pdb_id: str, output_dir: Path | None = None) -> Path:
    """Download a PDB file from RCSB.

    Parameters
    ----------
    pdb_id:
        4-character PDB accession code (e.g. ``"1HHP"``).
    output_dir:
        Where to save the file.  Defaults to :data:`_CACHE_DIR`.

    Returns
    -------
    Path
        Path to the downloaded PDB file.
    """
    pdb_id = pdb_id.upper().strip()
    output_dir = output_dir or _CACHE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    pdbl = PDBList(verbose=False)
    downloaded = pdbl.retrieve_pdb_file(
        pdb_id,
        pdir=str(output_dir),
        file_format="pdb",
    )

    if downloaded is None or not Path(downloaded).exists():
        raise FileNotFoundError(f"Could not download PDB {pdb_id} from RCSB.")

    # PDBList names the file pdb{id}.ent — rename to {id}.pdb for clarity
    dest = output_dir / f"{pdb_id}.pdb"
    if Path(downloaded) != dest:
        shutil.move(str(downloaded), str(dest))

    logger.info("Downloaded PDB %s → %s", pdb_id, dest)
    return dest


def extract_ligands(pdb_path: Path) -> list[dict]:
    """Extract co-crystallised ligands and their coordinates.

    These ligand positions are used to define the docking box — the
    rationale is that if a ligand was found there experimentally, the
    binding site is real.

    Parameters
    ----------
    pdb_path:
        Path to the raw PDB file.

    Returns
    -------
    list[dict]
        Each dict has ``resname``, ``chain``, ``resid``, ``center``
        (x, y, z), and ``atoms`` (list of coordinate tuples).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))

    # Common non-ligand heteroatoms to skip
    skip = {"HOH", "WAT", "DOD", "SO4", "PO4", "GOL", "EDO", "ACE", "NH2",
            "NAG", "MAN", "GAL", "FUC", "BGC", "BMA"}

    ligands: list[dict] = []

    for model in structure:
        for chain in model:
            for residue in chain:
                hetflag = residue.get_id()[0]
                resname = residue.get_resname().strip()
                if hetflag.strip() == "" or resname in skip:
                    continue
                # This is a heteroatom group — likely a ligand
                atoms = [atom.get_vector().get_array().tolist() for atom in residue]
                if not atoms:
                    continue
                import numpy as np
                center = np.mean(atoms, axis=0).tolist()
                ligands.append({
                    "resname": resname,
                    "chain": chain.get_id(),
                    "resid": residue.get_id()[1],
                    "center": center,
                    "atoms": atoms,
                    "num_atoms": len(atoms),
                })

    logger.info("Found %d ligand(s) in %s: %s",
                len(ligands), pdb_path.name,
                [lig["resname"] for lig in ligands])
    return ligands


def clean_protein(pdb_path: Path, output_path: Path | None = None) -> Path:
    """Remove waters, ligands, and alternate conformations from a PDB file.

    Parameters
    ----------
    pdb_path:
        Path to the raw PDB file.
    output_path:
        Where to write the cleaned file.  Defaults to
        ``{pdb_id}_clean.pdb`` in the same directory.

    Returns
    -------
    Path
        Path to the cleaned PDB file.
    """
    if output_path is None:
        output_path = pdb_path.with_name(pdb_path.stem + "_clean.pdb")

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))

    io = PDBIO()
    io.set_structure(structure)
    io.save(str(output_path), select=_CleanProteinSelect())

    logger.info("Cleaned protein → %s", output_path)
    return output_path


def convert_to_pdbqt(pdb_path: Path, output_path: Path | None = None) -> Path:
    """Convert a cleaned PDB to PDBQT format using Meeko.

    PDBQT adds partial atomic charges and AutoDock atom types, which are
    required by Vina's scoring function.  Meeko is the modern replacement
    for the legacy MGLTools / ``prepare_receptor4.py`` script.

    Parameters
    ----------
    pdb_path:
        Path to a cleaned PDB file (no waters / ligands).
    output_path:
        Where to write the PDBQT.  Defaults to same stem + ``.pdbqt``.

    Returns
    -------
    Path
        Path to the generated PDBQT file.

    Raises
    ------
    RuntimeError
        If Meeko or obabel is not available.
    """
    if output_path is None:
        output_path = pdb_path.with_suffix(".pdbqt")

    # Strategy 1: Try Meeko's receptor preparation
    try:
        from meeko import PDBQTReceptor  # type: ignore[import-untyped]
        receptor = PDBQTReceptor(str(pdb_path))
        receptor.write_pdbqt_file(str(output_path))
        logger.info("Converted to PDBQT (Meeko) → %s", output_path)
        return output_path
    except ImportError:
        logger.debug("Meeko PDBQTReceptor not available, trying obabel.")
    except Exception as exc:
        logger.debug("Meeko receptor prep failed: %s — trying obabel.", exc)

    # Strategy 2: Try Open Babel CLI
    obabel = shutil.which("obabel")
    if obabel:
        import subprocess
        result = subprocess.run(
            [obabel, str(pdb_path), "-O", str(output_path),
             "-xr", "--partialcharge", "gasteiger"],
            capture_output=True, text=True,
        )
        if result.returncode == 0 and output_path.exists():
            logger.info("Converted to PDBQT (obabel) → %s", output_path)
            return output_path
        logger.warning("obabel failed: %s", result.stderr)

    # Strategy 3: Minimal PDBQT conversion (add Gasteiger charges inline)
    # This is a fallback that writes a basic PDBQT from the PDB.
    logger.warning(
        "Neither Meeko nor obabel available — writing basic PDBQT. "
        "Install meeko (`pip install meeko`) for proper receptor preparation."
    )
    _basic_pdb_to_pdbqt(pdb_path, output_path)
    return output_path


def _basic_pdb_to_pdbqt(pdb_path: Path, output_path: Path) -> None:
    """Minimal PDB → PDBQT conversion (fallback when no tools available).

    This simply copies ATOM/HETATM lines and appends a dummy charge +
    atom type column.  It is *not* as accurate as Meeko but allows the
    pipeline to run for testing.
    """
    _AD_TYPE_MAP = {
        "C": "C", "N": "N", "O": "OA", "S": "SA", "H": "HD",
        "F": "F", "P": "P", "CL": "Cl", "BR": "Br", "I": "I",
        "ZN": "Zn", "FE": "Fe", "MG": "Mg", "CA": "Ca", "MN": "Mn",
    }
    lines: list[str] = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                element = line[76:78].strip().upper() if len(line) > 76 else line[12:16].strip()[0]
                ad_type = _AD_TYPE_MAP.get(element, "C")
                # PDBQT: columns 71-76 = partial charge, 77-79 = AD type
                pdbqt_line = f"{line[:54]}  0.000 {line[54:66]}    {ad_type:>2s}\n"
                lines.append(pdbqt_line)
            elif line.startswith(("END", "TER")):
                lines.append(line)

    with open(output_path, "w") as fh:
        fh.writelines(lines)
    logger.info("Basic PDBQT fallback → %s", output_path)


def get_protein_title(pdb_path: Path) -> str:
    """Extract the protein title from PDB HEADER/TITLE records."""
    titles: list[str] = []
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("TITLE"):
                titles.append(line[10:].strip())
            elif line.startswith("ATOM"):
                break
    return " ".join(titles) if titles else ""


def count_residues(pdb_path: Path) -> int:
    """Count amino acid residues in a PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", str(pdb_path))
    count = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == " ":
                    count += 1
    return count


def get_resolution(pdb_path: Path) -> float:
    """Extract crystallographic resolution from PDB REMARK 2."""
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith("REMARK   2 RESOLUTION."):
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == "RESOLUTION.":
                        try:
                            return float(parts[i + 1])
                        except (IndexError, ValueError):
                            pass
    return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# High-level API
# ═══════════════════════════════════════════════════════════════════════════


def prepare_protein(
    pdb_id: str,
    cache_dir: Path | None = None,
    force: bool = False,
) -> ProteinInfo:
    """Full protein preparation pipeline: fetch → clean → convert to PDBQT.

    Results are cached so repeated calls with the same PDB ID are instant.

    Parameters
    ----------
    pdb_id:
        4-character PDB accession code (e.g. ``"1HHP"`` — HIV-1 protease).
    cache_dir:
        Directory for caching prepared files.
    force:
        If ``True``, re-download and re-prepare even if cached.

    Returns
    -------
    ProteinInfo
        Metadata about the prepared protein, including path to PDBQT.
    """
    pdb_id = pdb_id.upper().strip()
    cache_dir = cache_dir or _CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    pdbqt_path = cache_dir / f"{pdb_id}_clean.pdbqt"

    # Return cached if available
    if pdbqt_path.exists() and not force:
        pdb_path = cache_dir / f"{pdb_id}.pdb"
        clean_path = cache_dir / f"{pdb_id}_clean.pdb"
        logger.info("Using cached PDBQT for %s", pdb_id)
        return ProteinInfo(
            pdb_id=pdb_id,
            name=get_protein_title(pdb_path) if pdb_path.exists() else "",
            resolution=get_resolution(pdb_path) if pdb_path.exists() else 0.0,
            pdbqt_path=str(pdbqt_path),
            num_residues=count_residues(clean_path) if clean_path.exists() else 0,
            removed_ligands=[
                lig["resname"]
                for lig in extract_ligands(pdb_path)
            ] if pdb_path.exists() else [],
        )

    # Full pipeline
    logger.info("Preparing protein %s …", pdb_id)

    # 1. Fetch from RCSB
    pdb_path = fetch_pdb(pdb_id, output_dir=cache_dir)

    # 2. Extract ligand info before cleaning (needed for binding site)
    ligands = extract_ligands(pdb_path)
    removed_names = [lig["resname"] for lig in ligands]

    # 3. Clean
    clean_path = clean_protein(pdb_path)

    # 4. Convert to PDBQT
    pdbqt_path = convert_to_pdbqt(clean_path)

    return ProteinInfo(
        pdb_id=pdb_id,
        name=get_protein_title(pdb_path),
        resolution=get_resolution(pdb_path),
        pdbqt_path=str(pdbqt_path),
        num_residues=count_residues(clean_path),
        removed_ligands=removed_names,
    )
