"""
MolScreen Synthetic Accessibility & Drug-Likeness Scores
=========================================================

Two complementary scores that answer the questions:

1. **SA Score (Synthetic Accessibility)** — *Can a chemist actually make
   this molecule?*  A score from 1 (trivially easy) to 10 (practically
   impossible).  Most approved drugs score 2–5.  The algorithm was
   designed by Ertl & Schuffenhauer (2009) and is bundled with RDKit.

2. **QED (Quantitative Estimate of Drug-likeness)** — *How drug-like is
   this molecule overall?*  A single number from 0 to 1 that integrates
   eight desirability functions (MW, LogP, HBA, HBD, TPSA, RotBonds,
   AromaticRings, Alerts) via a geometric mean.  Designed by Bickerton
   et al. (2012).  Higher = more drug-like.

Why both?
---------
SA Score captures *synthetic feasibility* (can we build it?), while QED
captures *pharmacological desirability* (should we build it?).  A molecule
needs to score well on both to be a viable drug candidate.
"""

from __future__ import annotations

import logging
from typing import Optional

from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Contrib.SA_Score import sascorer  # type: ignore[import-untyped]

from .parser import parse_smiles

logger = logging.getLogger(__name__)


def sa_score(mol_or_smiles: Chem.Mol | str) -> Optional[float]:
    """Calculate the **Synthetic Accessibility Score** (1–10).

    The SA Score estimates how easy it would be for a medicinal chemist to
    synthesise a molecule.  It combines two components:

    * **Fragment score** — how common the molecule's substructures are in
      known compounds (PubChem).  Common fragments → easy synthesis.
    * **Complexity penalty** — based on ring complexity, stereocentres,
      macrocycles, and spiro/bridged systems.

    Interpretation
    ~~~~~~~~~~~~~~
    * 1–3  — Easy to synthesise (simple commercial building blocks).
    * 3–5  — Moderate (most approved drugs live here).
    * 5–7  — Difficult (requires specialised chemistry).
    * 7–10 — Very difficult to near-impossible.

    Parameters
    ----------
    mol_or_smiles:
        RDKit Mol or SMILES string.

    Returns
    -------
    float | None
        SA score rounded to 2 decimal places, or ``None`` on failure.
    """
    mol = _resolve_mol(mol_or_smiles)
    if mol is None:
        return None

    try:
        score = sascorer.calculateScore(mol)
        return round(score, 2)
    except Exception:
        logger.exception("SA Score calculation failed.")
        return None


def qed_score(mol_or_smiles: Chem.Mol | str) -> Optional[float]:
    """Calculate the **QED** (Quantitative Estimate of Drug-likeness).

    QED was designed to replace the pass/fail nature of Lipinski with a
    continuous score.  It uses desirability functions for eight properties:

    1. Molecular weight
    2. LogP
    3. HBA
    4. HBD
    5. Polar surface area
    6. Rotatable bonds
    7. Aromatic rings
    8. Number of structural alerts

    Each property is scored 0–1 via a fitted asymmetric double-sigmoidal
    function, and the final QED is the weighted geometric mean.

    Interpretation
    ~~~~~~~~~~~~~~
    * > 0.67 — *Favourable* (in the "drug-like" sweet spot).
    * 0.49–0.67 — *Moderate* (some properties are sub-optimal).
    * < 0.49 — *Unfavourable* (multiple properties outside drug space).

    For reference: aspirin QED ≈ 0.55, atorvastatin (Lipitor) ≈ 0.15.

    Parameters
    ----------
    mol_or_smiles:
        RDKit Mol or SMILES string.

    Returns
    -------
    float | None
        QED score rounded to 3 decimal places, or ``None`` on failure.
    """
    mol = _resolve_mol(mol_or_smiles)
    if mol is None:
        return None

    try:
        score = QED.qed(mol)
        return round(score, 3)
    except Exception:
        logger.exception("QED calculation failed.")
        return None


def _resolve_mol(mol_or_smiles: Chem.Mol | str) -> Chem.Mol | None:
    """Resolve input to an RDKit Mol object."""
    if isinstance(mol_or_smiles, str):
        return parse_smiles(mol_or_smiles)
    return mol_or_smiles
