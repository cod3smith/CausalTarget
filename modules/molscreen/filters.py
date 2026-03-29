"""
MolScreen Drug-Likeness Filters
================================

Pharmaceutical industry filters that encode decades of empirical knowledge
about what makes a small molecule likely to succeed as an oral drug.

Each filter captures a different facet of "drug-likeness":

* **Lipinski's Rule of Five (1997)** — The original and most famous filter.
  Based on an analysis of ~2 200 drugs: oral drugs tend to have MW ≤ 500,
  LogP ≤ 5, HBD ≤ 5, HBA ≤ 10.  One violation is tolerated.

* **Veber Rules (2002)** — Adds flexibility and polarity: rotatable bonds
  ≤ 10 and TPSA ≤ 140 Å².  Predicts rat oral bioavailability.

* **Ghose Filter (1999)** — Tighter multi-dimensional box: MW 160–480,
  LogP −0.4–5.6, MR 40–130, atoms 20–70.

* **Egan Filter (2000)** — Simple 2D ellipse in LogP/TPSA space that
  predicts intestinal absorption.

* **PAINS (2010)** — Pan-Assay Interference Compounds.  Substructures that
  give false positives in biological assays (e.g. rhodanines, quinones).
  These waste enormous amounts of drug-discovery resources.

* **Brenk Filter (2008)** — Structural alerts for reactive, toxic, or
  metabolically unstable functional groups (e.g. Michael acceptors,
  alkyl halides, epoxides).
"""

from __future__ import annotations

import logging

from rdkit import Chem
from rdkit.Chem import Descriptors, FilterCatalog, Lipinski, rdMolDescriptors

from .models import FilterResult

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Filter implementations
# ═══════════════════════════════════════════════════════════════════════════


def lipinski_filter(mol: Chem.Mol) -> FilterResult:
    """Apply **Lipinski's Rule of Five**.

    The rule states that poor absorption or permeation is more likely when:

    * Molecular weight > 500 Da
    * LogP > 5
    * Hydrogen-bond donors > 5
    * Hydrogen-bond acceptors > 10

    A molecule **passes** if it has ≤ 1 violation (the original paper
    allows one).

    Parameters
    ----------
    mol:
        RDKit Mol object.

    Returns
    -------
    FilterResult
    """
    mw = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)

    violations: list[str] = []

    if mw > 500:
        violations.append(f"MW {mw:.1f} > 500")
    if logp > 5:
        violations.append(f"LogP {logp:.2f} > 5")
    if hbd > 5:
        violations.append(f"HBD {hbd} > 5")
    if hba > 10:
        violations.append(f"HBA {hba} > 10")

    return FilterResult(
        name="Lipinski",
        passed=len(violations) <= 1,
        violations=violations,
    )


def veber_filter(mol: Chem.Mol) -> FilterResult:
    """Apply **Veber Rules** for oral bioavailability.

    Both conditions must be satisfied:

    * Rotatable bonds ≤ 10
    * TPSA ≤ 140 Å²

    Veber et al. (2002) showed these two descriptors are sufficient to
    predict rat oral bioavailability, independent of molecular weight.

    Parameters
    ----------
    mol:
        RDKit Mol object.

    Returns
    -------
    FilterResult
    """
    rot = Lipinski.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)

    violations: list[str] = []

    if rot > 10:
        violations.append(f"Rotatable bonds {rot} > 10")
    if tpsa > 140:
        violations.append(f"TPSA {tpsa:.1f} > 140 Å²")

    return FilterResult(
        name="Veber",
        passed=len(violations) == 0,
        violations=violations,
    )


def ghose_filter(mol: Chem.Mol) -> FilterResult:
    """Apply the **Ghose Filter**.

    A four-dimensional box defining drug-like chemical space:

    * 160 ≤ MW ≤ 480
    * −0.4 ≤ LogP ≤ 5.6
    * 40 ≤ Molar Refractivity ≤ 130
    * 20 ≤ Total atom count ≤ 70

    Parameters
    ----------
    mol:
        RDKit Mol object.

    Returns
    -------
    FilterResult
    """
    mw = Descriptors.ExactMolWt(mol)
    logp = Descriptors.MolLogP(mol)
    mr = Descriptors.MolMR(mol)
    n_atoms = mol.GetNumAtoms()

    violations: list[str] = []

    if not (160 <= mw <= 480):
        violations.append(f"MW {mw:.1f} outside [160, 480]")
    if not (-0.4 <= logp <= 5.6):
        violations.append(f"LogP {logp:.2f} outside [-0.4, 5.6]")
    if not (40 <= mr <= 130):
        violations.append(f"MR {mr:.1f} outside [40, 130]")
    if not (20 <= n_atoms <= 70):
        violations.append(f"Atom count {n_atoms} outside [20, 70]")

    return FilterResult(
        name="Ghose",
        passed=len(violations) == 0,
        violations=violations,
    )


def egan_filter(mol: Chem.Mol) -> FilterResult:
    """Apply the **Egan Filter** for intestinal absorption.

    Defines an elliptical region in TPSA / LogP space:

    * TPSA ≤ 131.6 Å²
    * LogP ≤ 5.88

    Molecules within this region are predicted to have good human
    intestinal absorption (HIA).

    Parameters
    ----------
    mol:
        RDKit Mol object.

    Returns
    -------
    FilterResult
    """
    tpsa = Descriptors.TPSA(mol)
    logp = Descriptors.MolLogP(mol)

    violations: list[str] = []

    if tpsa > 131.6:
        violations.append(f"TPSA {tpsa:.1f} > 131.6 Å²")
    if logp > 5.88:
        violations.append(f"LogP {logp:.2f} > 5.88")

    return FilterResult(
        name="Egan",
        passed=len(violations) == 0,
        violations=violations,
    )


# ---------------------------------------------------------------------------
# Substructure-based filters (PAINS, Brenk) via RDKit FilterCatalog
# ---------------------------------------------------------------------------

# Build PAINS catalog once at import time (thread-safe, read-only).
_pains_params = FilterCatalog.FilterCatalogParams()
_pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
_pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
_pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
_PAINS_CATALOG = FilterCatalog.FilterCatalog(_pains_params)

# Build Brenk catalog once.
_brenk_params = FilterCatalog.FilterCatalogParams()
_brenk_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
_BRENK_CATALOG = FilterCatalog.FilterCatalog(_brenk_params)


def pains_filter(mol: Chem.Mol) -> FilterResult:
    """Screen for **PAINS** (Pan-Assay Interference Compounds).

    PAINS are substructures known to interfere with many biological
    assays — they appear as "hits" but are actually artefacts.  Common
    PAINS motifs include:

    * Rhodanines
    * Catechols
    * Quinones
    * Hydroxyphenyl hydrazones

    Publishing a PAINS compound as a "hit" is considered a serious
    credibility issue in modern drug discovery.

    Parameters
    ----------
    mol:
        RDKit Mol object.

    Returns
    -------
    FilterResult
    """
    matches = _PAINS_CATALOG.GetMatches(mol)
    violations = [m.GetDescription() for m in matches]

    return FilterResult(
        name="PAINS",
        passed=len(violations) == 0,
        violations=violations,
    )


def brenk_filter(mol: Chem.Mol) -> FilterResult:
    """Screen for **Brenk structural alerts**.

    Brenk et al. (2008) compiled a list of 105 substructural features
    that are undesirable in drug candidates because they are:

    * Chemically reactive (e.g. Michael acceptors, epoxides)
    * Metabolically unstable (e.g. thiols, aldehydes)
    * Toxic or mutagenic (e.g. nitro groups on aromatics)

    Parameters
    ----------
    mol:
        RDKit Mol object.

    Returns
    -------
    FilterResult
    """
    matches = _BRENK_CATALOG.GetMatches(mol)
    violations = [m.GetDescription() for m in matches]

    return FilterResult(
        name="Brenk",
        passed=len(violations) == 0,
        violations=violations,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation
# ═══════════════════════════════════════════════════════════════════════════


def run_all_filters(mol: Chem.Mol) -> list[FilterResult]:
    """Run **all** drug-likeness filters on a molecule.

    Parameters
    ----------
    mol:
        RDKit Mol object.

    Returns
    -------
    list[FilterResult]
        Results for every filter, in a fixed order.
    """
    return [
        lipinski_filter(mol),
        veber_filter(mol),
        ghose_filter(mol),
        egan_filter(mol),
        pains_filter(mol),
        brenk_filter(mol),
    ]


def classify_drug_likeness(filters: list[FilterResult]) -> str:
    """Classify a molecule based on its filter results.

    Classification logic
    --------------------
    * **Drug-like** — passes Lipinski *and* Veber *and* no PAINS.
    * **Lead-like** — fails one of the above but passes at least 4/6 filters.
    * **Not drug-like** — fails more than 2 filters.

    Parameters
    ----------
    filters:
        Output of :func:`run_all_filters`.

    Returns
    -------
    str
        One of ``"Drug-like"``, ``"Lead-like"``, ``"Not drug-like"``.
    """
    by_name = {f.name: f.passed for f in filters}
    total_passed = sum(1 for f in filters if f.passed)

    core_pass = (
        by_name.get("Lipinski", False)
        and by_name.get("Veber", False)
        and by_name.get("PAINS", True)
    )

    if core_pass and total_passed >= 5:
        return "Drug-like"
    if total_passed >= 4:
        return "Lead-like"
    return "Not drug-like"
