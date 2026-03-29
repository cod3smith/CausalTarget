"""
ChEMBL Data Download
=====================

Downloads drug-like molecules from the ChEMBL database for training
the generative model.

Why ChEMBL?
-----------
ChEMBL is the largest open-access database of bioactive drug-like
small molecules (~2.4M compounds).  It's curated by the EMBL-EBI
and is the gold standard for cheminformatics research.

We filter for "drug-like" molecules using these criteria:

* **Molecular weight** 150–500 Da — too small = fragments,
  too large = poor oral bioavailability.
* **LogP** −1 to 5 — Lipinski range for membrane permeability.
* **HBD** ≤ 5, **HBA** ≤ 10 — hydrogen bond donors/acceptors
  (Lipinski's Rule of Five).
* **Valid SMILES** that RDKit can parse and canonicalize.

The resulting dataset (~500 K–1 M molecules) provides a rich
distribution of drug-like chemical space for VAE training.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

from rdkit import Chem, RDLogger

logger = logging.getLogger(__name__)

# Silence RDKit warnings during bulk parsing
RDLogger.DisableLog("rdApp.*")

# ── Default paths ───────────────────────────────────────────────────
_DATA_DIR = Path(__file__).parent / "cache"
_DEFAULT_CSV = _DATA_DIR / "chembl_druglike.csv"

# ── Filtering thresholds ────────────────────────────────────────────
_DEFAULT_FILTERS = {
    "mw_min": 150.0,
    "mw_max": 500.0,
    "logp_min": -1.0,
    "logp_max": 5.0,
    "hbd_max": 5,
    "hba_max": 10,
    "max_molecules": 500_000,
}


def _is_drug_like(
    mol: Chem.Mol,
    *,
    mw_min: float = 150.0,
    mw_max: float = 500.0,
    logp_min: float = -1.0,
    logp_max: float = 5.0,
    hbd_max: int = 5,
    hba_max: int = 10,
) -> bool:
    """Check whether a molecule passes basic drug-likeness filters.

    Uses RDKit descriptor calculations rather than the MolScreen module
    to keep the data pipeline self-contained and fast (no Pydantic model
    overhead for millions of molecules).
    """
    from rdkit.Chem import Descriptors, Lipinski

    mw = Descriptors.ExactMolWt(mol)
    if not (mw_min <= mw <= mw_max):
        return False

    logp = Descriptors.MolLogP(mol)
    if not (logp_min <= logp <= logp_max):
        return False

    hbd = Lipinski.NumHDonors(mol)
    if hbd > hbd_max:
        return False

    hba = Lipinski.NumHAcceptors(mol)
    if hba > hba_max:
        return False

    return True


def download_chembl(
    output_path: Optional[str | Path] = None,
    max_molecules: int = 500_000,
    **filter_kwargs,
) -> Path:
    """Download drug-like SMILES from ChEMBL via the REST API.

    Uses ``chembl_webresource_client`` to query the ChEMBL database.
    Results are cached as a CSV so the download only happens once.

    Parameters
    ----------
    output_path : str or Path, optional
        Where to save the CSV.  Defaults to
        ``modules/genmol/data/cache/chembl_druglike.csv``.
    max_molecules : int
        Maximum number of molecules to download.
    **filter_kwargs
        Overrides for drug-likeness thresholds (``mw_min``, ``mw_max``,
        ``logp_min``, ``logp_max``, ``hbd_max``, ``hba_max``).

    Returns
    -------
    Path
        Path to the output CSV file.
    """
    output = Path(output_path) if output_path else _DEFAULT_CSV

    if output.exists():
        # Count existing rows
        with open(output) as f:
            n = sum(1 for _ in f) - 1  # subtract header
        logger.info("Cache exists with %d molecules → %s", n, output)
        return output

    output.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading drug-like molecules from ChEMBL…")

    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        raise ImportError(
            "chembl_webresource_client is required for download. "
            "Install it with: uv add chembl-webresource-client"
        )

    # Merge filter defaults with user overrides
    filters = {**_DEFAULT_FILTERS, **filter_kwargs}

    molecule_api = new_client.molecule  # type: ignore[attr-defined]

    # Query ChEMBL for drug-like molecules
    # The API supports server-side filtering on some properties
    results = molecule_api.filter(
        molecule_properties__mw_freebase__gte=filters["mw_min"],
        molecule_properties__mw_freebase__lte=filters["mw_max"],
        molecule_properties__alogp__gte=filters["logp_min"],
        molecule_properties__alogp__lte=filters["logp_max"],
        molecule_properties__hbd__lte=filters["hbd_max"],
        molecule_properties__hba__lte=filters["hba_max"],
    ).only(["molecule_structures"])

    smiles_set: set[str] = set()
    count = 0

    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles"])

        for record in results:
            if count >= max_molecules:
                break

            structures = record.get("molecule_structures")
            if not structures:
                continue

            smi = structures.get("canonical_smiles")
            if not smi or smi in smiles_set:
                continue

            # Validate with RDKit
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            # Re-canonicalize
            canonical = Chem.MolToSmiles(mol)
            if canonical in smiles_set:
                continue

            # Client-side drug-likeness re-check
            if not _is_drug_like(mol, **{
                k: v for k, v in filters.items()
                if k != "max_molecules"
            }):
                continue

            smiles_set.add(canonical)
            writer.writerow([canonical])
            count += 1

            if count % 10_000 == 0:
                logger.info("  … %d molecules collected.", count)

    logger.info(
        "Download complete: %d drug-like molecules → %s",
        count,
        output,
    )
    return output


def load_smiles(path: Optional[str | Path] = None) -> list[str]:
    """Load SMILES from a CSV file.

    Parameters
    ----------
    path : str or Path, optional
        Path to a CSV with a ``smiles`` column.
        Defaults to the cached ChEMBL file.

    Returns
    -------
    list[str]
        List of canonical SMILES strings.
    """
    csv_path = Path(path) if path else _DEFAULT_CSV

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {csv_path}\n"
            "Run `genmol download` or call download_chembl() first."
        )

    smiles: list[str] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row.get("smiles", "").strip()
            if smi:
                smiles.append(smi)

    logger.info("Loaded %d SMILES from %s.", len(smiles), csv_path)
    return smiles
