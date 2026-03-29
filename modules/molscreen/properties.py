"""
MolScreen Property Calculator
==============================

Calculates physico-chemical molecular properties using RDKit descriptors.
These properties are the foundation of drug-likeness assessment — they
determine whether a molecule can be absorbed, distributed, metabolised,
and excreted (ADME) in the human body.

Every property calculated here maps to a biological reality:

* **Molecular Weight** — bigger molecules struggle to cross cell membranes.
* **LogP** — the oil/water balance controls membrane permeation.
* **HBD / HBA** — hydrogen bonds to water must be broken to enter cells.
* **TPSA** — polar surface area predicts oral absorption and BBB crossing.
* **Rotatable Bonds** — flexibility affects binding entropy and bioavailability.
* **Rings / Aromatic Rings** — shape and planarity affect selectivity.
* **Fsp3** — three-dimensionality correlates with clinical success.
* **Molar Refractivity** — volume and polarisability proxy.
"""

from __future__ import annotations

import logging
from typing import Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

from .models import MolecularProperties
from .parser import parse_smiles

logger = logging.getLogger(__name__)


def calculate_properties(mol_or_smiles: Chem.Mol | str) -> Optional[MolecularProperties]:
    """Calculate a full suite of molecular properties.

    Parameters
    ----------
    mol_or_smiles:
        An RDKit ``Mol`` object **or** a SMILES string.  If a string is
        provided it will be parsed first.

    Returns
    -------
    MolecularProperties | None
        A populated data model, or ``None`` if the input is invalid.

    Notes
    -----
    All descriptors are computed via RDKit's fast C++ backend.  The
    function is deliberately stateless so it can be safely parallelised
    for batch processing.
    """
    # Resolve input to an RDKit Mol
    if isinstance(mol_or_smiles, str):
        mol = parse_smiles(mol_or_smiles)
        if mol is None:
            return None
    else:
        mol = mol_or_smiles

    try:
        canonical_smiles = Chem.MolToSmiles(mol)

        return MolecularProperties(
            smiles=canonical_smiles,
            molecular_weight=_molecular_weight(mol),
            logp=_logp(mol),
            hbd=_hbd(mol),
            hba=_hba(mol),
            tpsa=_tpsa(mol),
            rotatable_bonds=_rotatable_bonds(mol),
            num_rings=_num_rings(mol),
            num_aromatic_rings=_num_aromatic_rings(mol),
            fsp3=_fsp3(mol),
            molar_refractivity=_molar_refractivity(mol),
        )
    except Exception:
        logger.exception("Failed to calculate properties for molecule.")
        return None


# ---------------------------------------------------------------------------
# Individual descriptor functions
# ---------------------------------------------------------------------------
# Wrapped in thin helpers for readability, testability, and consistent
# rounding.


def _molecular_weight(mol: Chem.Mol) -> float:
    """Exact molecular weight (monoisotopic masses).

    Why it matters
    --------------
    The gut lining and cell membranes act as molecular sieves — large
    molecules (>500 Da) have difficulty passing through.  This is the
    first of Lipinski's four rules.
    """
    return round(Descriptors.ExactMolWt(mol), 2)


def _logp(mol: Chem.Mol) -> float:
    """Wildman-Crippen LogP — octanol/water partition coefficient.

    Why it matters
    --------------
    LogP quantifies hydrophobicity.  Cell membranes are lipid bilayers,
    so a drug must have *some* lipophilicity to permeate.  But too much
    (LogP > 5) leads to poor aqueous solubility, plasma protein binding,
    and metabolic instability.  The sweet spot is roughly 1–3.
    """
    return round(Descriptors.MolLogP(mol), 2)


def _hbd(mol: Chem.Mol) -> int:
    """Number of hydrogen-bond donors (NH + OH groups).

    Why it matters
    --------------
    Each donor forms a hydrogen bond with surrounding water molecules.
    To cross a lipid membrane, these bonds must be desolvated — an
    energetically costly process.  Lipinski's limit: ≤ 5 donors.
    """
    return Lipinski.NumHDonors(mol)


def _hba(mol: Chem.Mol) -> int:
    """Number of hydrogen-bond acceptors (N + O atoms).

    Why it matters
    --------------
    Same desolvation penalty as donors.  Lipinski's limit: ≤ 10 acceptors.
    """
    return Lipinski.NumHAcceptors(mol)


def _tpsa(mol: Chem.Mol) -> float:
    """Topological Polar Surface Area (Å²).

    Why it matters
    --------------
    TPSA is the best single predictor of oral drug absorption.
    * ≤ 140 Å² → good intestinal absorption.
    * ≤  90 Å² → can cross the blood-brain barrier (important for CNS drugs).
    * >  140 Å² → likely poor oral bioavailability.
    """
    return round(Descriptors.TPSA(mol), 2)


def _rotatable_bonds(mol: Chem.Mol) -> int:
    """Number of rotatable (freely spinning) bonds.

    Why it matters
    --------------
    Rotatable bonds define molecular flexibility.  When a flexible
    molecule binds its target, it must freeze into one conformation —
    losing entropy.  Veber showed that ≤ 10 rotatable bonds correlates
    with good oral bioavailability in rats.
    """
    return Lipinski.NumRotatableBonds(mol)


def _num_rings(mol: Chem.Mol) -> int:
    """Total ring count (SSSR — Smallest Set of Smallest Rings).

    Why it matters
    --------------
    Rings add rigidity and define the 3D shape that lets a molecule fit
    a protein binding site.  Most oral drugs have 1–5 rings.
    """
    return Lipinski.RingCount(mol)


def _num_aromatic_rings(mol: Chem.Mol) -> int:
    """Number of aromatic rings.

    Why it matters
    --------------
    Aromatic rings enable π-π stacking with protein residues (Phe, Tyr,
    Trp, His).  However, too many aromatic rings (> 3) increase planarity
    and promote molecular aggregation, reducing solubility and increasing
    off-target toxicity.
    """
    return rdMolDescriptors.CalcNumAromaticRings(mol)


def _fsp3(mol: Chem.Mol) -> float:
    """Fraction of sp3-hybridised carbon atoms.

    Why it matters
    --------------
    sp3 carbons have tetrahedral geometry (3D), while sp2 carbons are flat.
    Higher Fsp3 (≥ 0.42) correlates with greater clinical success because
    3D molecules explore more chemical space, are more soluble, and show
    fewer off-target effects.  Flat molecules tend to be promiscuous binders.
    """
    return round(rdMolDescriptors.CalcFractionCSP3(mol), 3)


def _molar_refractivity(mol: Chem.Mol) -> float:
    """Wildman-Crippen Molar Refractivity.

    Why it matters
    --------------
    Molar refractivity (MR) is proportional to molecular volume and
    electronic polarisability.  The Ghose filter uses MR (40–130) as a
    size / steric descriptor complementary to molecular weight.
    """
    return round(Descriptors.MolMR(mol), 2)
