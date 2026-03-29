"""
MolScreen Similarity Search
=============================

Compares a query molecule against a reference set of approved drugs from
the **ChEMBL** database using molecular fingerprints and the Tanimoto
similarity coefficient.

Data source
-----------
Approved drugs (``max_phase = 4``) are fetched from ChEMBL via the
official ``chembl_webresource_client``.  Results are cached locally as a
CSV so the API is only hit once (or when you explicitly refresh).

How it works
------------
1. Each molecule is encoded as a **Morgan fingerprint** (circular
   fingerprint, radius=2, 2048 bits).  This captures the local chemical
   environment around every atom up to two bonds away — similar to the
   ECFP4 fingerprints widely used in pharma.

2. Fingerprints are compared using the **Tanimoto coefficient** (a.k.a.
   Jaccard index):

   .. math::

       T(A, B) = \\frac{|A \\cap B|}{|A \\cup B|}

   A Tanimoto score of 1.0 means the fingerprints are identical; 0.0
   means no overlap.

Why this matters
----------------
Finding approved drugs that are structurally similar to a new compound
gives clues about its pharmacology, toxicity, and patent landscape.  If
your molecule is 0.85+ similar to an approved drug, it may share the
same target — useful for drug repurposing.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .models import SimilarDrug
from .parser import parse_smiles

logger = logging.getLogger(__name__)

# Morgan fingerprint generator (modern API, replaces deprecated
# GetMorganFingerprintAsBitVect).  Radius 2 ≈ ECFP4, 2048 bits.
_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Local cache directory — stores the ChEMBL extract so we don't re-fetch.
_DATA_DIR = Path(__file__).parent / "data"
_CACHE_CSV = _DATA_DIR / "approved_drugs_chembl.csv"

# Module-level runtime caches (populated once per process).
_drug_cache: list[dict] | None = None
_fp_cache: list[tuple[dict, DataStructs.ExplicitBitVect]] | None = None


# ═══════════════════════════════════════════════════════════════════════════
# ChEMBL data fetching
# ═══════════════════════════════════════════════════════════════════════════


def fetch_approved_drugs_from_chembl(limit: int | None = None) -> list[dict]:
    """Fetch approved drugs (max_phase 4) from ChEMBL via the REST API.

    Uses ``chembl_webresource_client`` to query molecules that have
    reached **phase 4** (approved for human use).

    Parameters
    ----------
    limit:
        Maximum number of drugs to fetch.  ``None`` means all available
        (typically ~2 500+).  Use a smaller number for faster initial
        setup.

    Returns
    -------
    list[dict]
        Each dict has keys: ``chembl_id``, ``name``, ``smiles``,
        ``indication``.
    """
    from chembl_webresource_client.new_client import new_client  # type: ignore[import-untyped]

    logger.info("Fetching approved drugs from ChEMBL (max_phase=4) …")

    molecule_resource = new_client.molecule  # type: ignore[attr-defined]
    query = molecule_resource.filter(max_phase=4).only(
        [
            "molecule_chembl_id",
            "pref_name",
            "molecule_structures",
            "indication_class",
        ]
    )

    drugs: list[dict] = []
    seen: set[str] = set()

    for record in query:
        structures = record.get("molecule_structures")
        if not structures:
            continue
        smiles = structures.get("canonical_smiles")
        if not smiles:
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol)
        if canonical in seen:
            continue
        seen.add(canonical)

        name = record.get("pref_name") or record.get("molecule_chembl_id", "Unknown")
        chembl_id = record.get("molecule_chembl_id", "")
        indication = record.get("indication_class") or ""

        drugs.append(
            {
                "chembl_id": chembl_id,
                "name": name,
                "smiles": canonical,
                "indication": indication,
            }
        )

        if limit and len(drugs) >= limit:
            break

    logger.info("Fetched %d approved drugs from ChEMBL.", len(drugs))
    return drugs


def build_cache(limit: int | None = None, force: bool = False) -> Path:
    """Fetch approved drugs from ChEMBL and save to a local CSV cache.

    Parameters
    ----------
    limit:
        Max drugs to fetch (``None`` = all).
    force:
        If ``True``, re-fetch even if the cache file already exists.

    Returns
    -------
    Path
        Path to the generated cache CSV.
    """
    if _CACHE_CSV.exists() and not force:
        logger.info("Cache already exists at %s — use force=True to refresh.", _CACHE_CSV)
        return _CACHE_CSV

    drugs = fetch_approved_drugs_from_chembl(limit=limit)
    if not drugs:
        logger.warning("No drugs fetched from ChEMBL; cache not written.")
        return _CACHE_CSV

    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["chembl_id", "name", "smiles", "indication"])
        writer.writeheader()
        writer.writerows(drugs)

    logger.info("Wrote %d drugs to cache: %s", len(drugs), _CACHE_CSV)
    return _CACHE_CSV


def clear_cache() -> None:
    """Clear both the in-memory and on-disk caches.

    Call :func:`build_cache` with ``force=True`` afterwards to do a
    full refresh from ChEMBL.
    """
    global _drug_cache, _fp_cache
    _drug_cache = None
    _fp_cache = None
    if _CACHE_CSV.exists():
        _CACHE_CSV.unlink()
        logger.info("Deleted cache file: %s", _CACHE_CSV)


# ═══════════════════════════════════════════════════════════════════════════
# Reference data loading (from local cache)
# ═══════════════════════════════════════════════════════════════════════════


def _load_approved_drugs() -> list[dict]:
    """Load approved drugs from the local ChEMBL cache CSV.

    If the cache doesn't exist yet, it will be fetched from ChEMBL
    automatically on first use.

    Returns
    -------
    list[dict]
        Each dict has ``name``, ``smiles``, ``indication``, ``mol``
        (RDKit Mol object).
    """
    global _drug_cache
    if _drug_cache is not None:
        return _drug_cache

    # Auto-build cache on first use if it doesn't exist
    if not _CACHE_CSV.exists():
        logger.info("No local cache found — fetching from ChEMBL …")
        try:
            build_cache()
        except Exception:
            logger.exception(
                "Failed to fetch from ChEMBL. Run "
                "`python -m molscreen build-cache` manually when online."
            )
            return []

    if not _CACHE_CSV.exists():
        logger.error("Cache CSV still missing after fetch attempt.")
        return []

    drugs: list[dict] = []
    seen_smiles: set[str] = set()

    with open(_CACHE_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smiles = row.get("smiles", "").strip()
            if not smiles or smiles in seen_smiles:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            canonical = Chem.MolToSmiles(mol)
            if canonical in seen_smiles:
                continue
            seen_smiles.add(canonical)
            drugs.append(
                {
                    "chembl_id": row.get("chembl_id", ""),
                    "name": row.get("name", "Unknown").strip(),
                    "smiles": canonical,
                    "indication": row.get("indication", "").strip(),
                    "mol": mol,
                }
            )

    _drug_cache = drugs
    logger.info("Loaded %d approved drugs for similarity search.", len(drugs))
    return drugs


def _get_fingerprint_cache() -> list[tuple[dict, DataStructs.ExplicitBitVect]]:
    """Precompute Morgan fingerprints for all reference drugs.

    This is expensive on first call (~200 molecules) but cached afterwards.
    """
    global _fp_cache
    if _fp_cache is not None:
        return _fp_cache

    drugs = _load_approved_drugs()
    cache: list[tuple[dict, DataStructs.ExplicitBitVect]] = []

    for drug in drugs:
        try:
            fp = _MORGAN_GEN.GetFingerprint(drug["mol"])
            cache.append((drug, fp))
        except Exception:
            logger.debug("Could not fingerprint %s", drug["name"])

    _fp_cache = cache
    return cache


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


def find_similar_drugs(
    mol_or_smiles: Chem.Mol | str,
    top_n: int = 5,
) -> list[SimilarDrug]:
    """Find the most similar FDA-approved drugs to a query molecule.

    Parameters
    ----------
    mol_or_smiles:
        Query molecule (RDKit Mol or SMILES string).
    top_n:
        Number of results to return (default 5).

    Returns
    -------
    list[SimilarDrug]
        Top-N most similar approved drugs, sorted by descending Tanimoto
        similarity.

    Examples
    --------
    >>> results = find_similar_drugs("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    >>> results[0].name
    'Aspirin'
    """
    # Resolve to Mol
    if isinstance(mol_or_smiles, str):
        mol = parse_smiles(mol_or_smiles)
        if mol is None:
            logger.warning("Invalid SMILES for similarity search.")
            return []
    else:
        mol = mol_or_smiles

    # Compute query fingerprint
    try:
        query_fp = _MORGAN_GEN.GetFingerprint(mol)
    except Exception:
        logger.exception("Could not compute fingerprint for query molecule.")
        return []

    # Compare against all reference drugs
    cache = _get_fingerprint_cache()
    similarities: list[tuple[float, dict]] = []

    for drug_info, ref_fp in cache:
        tanimoto = DataStructs.TanimotoSimilarity(query_fp, ref_fp)
        similarities.append((tanimoto, drug_info))

    # Sort descending by similarity and take top N
    similarities.sort(key=lambda x: x[0], reverse=True)

    results: list[SimilarDrug] = []
    for sim, drug_info in similarities[:top_n]:
        results.append(
            SimilarDrug(
                name=drug_info["name"],
                smiles=drug_info["smiles"],
                similarity=round(sim, 4),
                indication=drug_info.get("indication", ""),
            )
        )

    return results


def tanimoto_similarity(
    smiles_a: str,
    smiles_b: str,
) -> Optional[float]:
    """Calculate Tanimoto similarity between two molecules.

    Parameters
    ----------
    smiles_a:
        First SMILES string.
    smiles_b:
        Second SMILES string.

    Returns
    -------
    float | None
        Tanimoto coefficient (0–1), or ``None`` if either SMILES is invalid.
    """
    mol_a = parse_smiles(smiles_a)
    mol_b = parse_smiles(smiles_b)
    if mol_a is None or mol_b is None:
        return None

    try:
        fp_a = _MORGAN_GEN.GetFingerprint(mol_a)
        fp_b = _MORGAN_GEN.GetFingerprint(mol_b)
        return round(DataStructs.TanimotoSimilarity(fp_a, fp_b), 4)
    except Exception:
        logger.exception("Tanimoto calculation failed.")
        return None
