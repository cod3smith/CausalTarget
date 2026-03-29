"""
MolScreen SMILES Parser
========================

Parses and validates SMILES (Simplified Molecular-Input Line-Entry System)
strings using RDKit, and provides name-to-SMILES lookup via the PubChem
PUG REST API.

SMILES is the *lingua franca* of cheminformatics — a line notation that
encodes molecular structure as a string.  For example, ``CCO`` is ethanol
and ``c1ccccc1`` is benzene.

This module is the first step in the MolScreen pipeline: every molecule
enters as a SMILES string and must be validated before any properties
can be calculated.
"""

from __future__ import annotations

import logging
from typing import Optional

import requests
from rdkit import Chem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PubChem PUG REST configuration
# ---------------------------------------------------------------------------
_PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_PUBCHEM_TIMEOUT = 10  # seconds


def parse_smiles(smiles: str) -> Optional[Chem.Mol]:
    """Parse a SMILES string and return an RDKit ``Mol`` object.

    Parameters
    ----------
    smiles:
        A SMILES string representing a chemical structure.

    Returns
    -------
    Chem.Mol | None
        The parsed molecule, or ``None`` if the SMILES is invalid.

    Notes
    -----
    RDKit's ``MolFromSmiles`` returns ``None`` for un-parseable strings
    rather than raising an exception.  We log a warning so callers can
    trace failures without try/except boilerplate.

    Examples
    --------
    >>> mol = parse_smiles("CCO")  # ethanol
    >>> mol is not None
    True
    >>> parse_smiles("not_a_smiles") is None
    True
    """
    if not smiles or not isinstance(smiles, str):
        logger.warning("Empty or non-string input passed to parse_smiles.")
        return None

    smiles = smiles.strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning("Invalid SMILES string: '%s'", smiles)
    return mol


def validate_smiles(smiles: str) -> bool:
    """Return ``True`` if *smiles* can be parsed into a valid molecule.

    This is a thin convenience wrapper around :func:`parse_smiles`.

    Parameters
    ----------
    smiles:
        SMILES string to validate.

    Returns
    -------
    bool
    """
    return parse_smiles(smiles) is not None


def canonicalise(smiles: str) -> str | None:
    """Return the *canonical* SMILES for a molecule.

    Canonical SMILES is a unique, deterministic representation — two
    molecules with the same structure will always produce the same
    canonical SMILES.  This is essential for de-duplication and database
    lookups.

    Parameters
    ----------
    smiles:
        Input SMILES (may be non-canonical).

    Returns
    -------
    str | None
        Canonical SMILES string, or ``None`` if input is invalid.
    """
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def name_to_smiles(name: str) -> str | None:
    """Look up a drug / compound name and return its canonical SMILES.

    Uses the `PubChem PUG REST API
    <https://pubchem.ncbi.nlm.nih.gov/rest/pug>`_ to resolve common names
    (e.g. "aspirin", "ibuprofen") to SMILES.

    Parameters
    ----------
    name:
        Common or IUPAC name of the compound.

    Returns
    -------
    str | None
        Canonical SMILES if found, otherwise ``None``.

    Notes
    -----
    This makes an HTTP request to an external service.  It will time out
    after ``_PUBCHEM_TIMEOUT`` seconds and return ``None`` on any network
    error.
    """
    if not name or not isinstance(name, str):
        logger.warning("Empty or non-string name passed to name_to_smiles.")
        return None

    name = name.strip()
    url = f"{_PUBCHEM_BASE}/compound/name/{name}/property/CanonicalSMILES/JSON"

    try:
        response = requests.get(url, timeout=_PUBCHEM_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        smiles = (
            data.get("PropertyTable", {})
            .get("Properties", [{}])[0]
            .get("CanonicalSMILES")
        )
        if smiles:
            logger.info("Resolved '%s' → '%s' via PubChem.", name, smiles)
            return smiles
        logger.warning("PubChem returned no SMILES for '%s'.", name)
        return None
    except requests.exceptions.RequestException as exc:
        logger.error("PubChem lookup failed for '%s': %s", name, exc)
        return None
    except (KeyError, IndexError, ValueError) as exc:
        logger.error("Unexpected PubChem response for '%s': %s", name, exc)
        return None


def smart_parse(input_str: str) -> tuple[Chem.Mol | None, str | None]:
    """Attempt to interpret *input_str* as a SMILES string **or** a drug name.

    Strategy:
    1. Try parsing as SMILES directly.
    2. If that fails, treat it as a compound name and query PubChem.

    Parameters
    ----------
    input_str:
        A SMILES string or compound name (e.g. ``"aspirin"``).

    Returns
    -------
    tuple[Mol | None, str | None]
        ``(mol_object, canonical_smiles)`` — both ``None`` if resolution
        fails.
    """
    if not input_str or not isinstance(input_str, str):
        return None, None

    input_str = input_str.strip()

    # 1) Try direct SMILES parse
    mol = parse_smiles(input_str)
    if mol is not None:
        return mol, Chem.MolToSmiles(mol)

    # 2) Treat as a compound name
    logger.info(
        "Could not parse '%s' as SMILES — attempting PubChem name lookup.",
        input_str,
    )
    smiles = name_to_smiles(input_str)
    if smiles is None:
        return None, None

    mol = parse_smiles(smiles)
    return mol, smiles
