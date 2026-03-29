"""
DockBot Composite Scorer
===========================

Combines docking scores with drug-likeness assessments to rank
compounds holistically.

Why a composite score?
----------------------
A molecule may bind tightly (good Vina score) but still be a terrible
drug if it violates absorption / distribution / metabolism / excretion
(ADME) rules, is synthetically intractable, or triggers PAINS alerts.

The composite score integrates:

1. **Vina affinity** (kcal/mol) — predicted binding free energy.
2. **QED** (0–1) — Quantitative Estimate of Drug-likeness.
3. **SA Score** (1–10, lower = easier) — synthetic accessibility.
4. **Filter pass-rate** — fraction of Lipinski/Veber/Ghose/Egan/PAINS
   filters passed.

Scoring formula
---------------
.. math::

    S = w_a \\cdot \\text{norm}(\\Delta G) + w_q \\cdot \\text{QED}
        + w_s \\cdot (1 - \\frac{\\text{SA} - 1}{9})
        + w_f \\cdot \\text{filter\\_rate}

where *norm(ΔG)* maps the affinity to [0, 1] using a sigmoid centred
at −7 kcal/mol.  Higher composite score = better drug candidate.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from rdkit import Chem

from .models import DockingResult

logger = logging.getLogger(__name__)


@dataclass
class ScoringWeights:
    """Tunable weights for the composite scoring function.

    All weights should sum to 1.0 for the score to stay in [0, 1].
    """

    affinity: float = 0.40
    qed: float = 0.25
    sa: float = 0.15
    filters: float = 0.20


@dataclass
class CompositeScore:
    """Result of composite scoring for a single ligand."""

    ligand_name: str
    smiles: str

    # Raw components
    affinity_kcal_mol: float
    qed: float
    sa_score: float
    filter_pass_rate: float

    # Normalised [0, 1] components
    norm_affinity: float
    norm_qed: float
    norm_sa: float
    norm_filters: float

    # Final composite
    composite: float

    # Drug-likeness category from MolScreen
    drug_likeness: str = ""

    # Rank (filled after sorting)
    rank: int = 0


def normalise_affinity(
    affinity_kcal_mol: float,
    midpoint: float = -7.0,
    steepness: float = 1.0,
) -> float:
    """Map Vina ΔG to [0, 1] with a sigmoid.

    ``midpoint`` is the affinity at which the normalised score = 0.5.
    More negative affinities (stronger binding) → higher score.
    """
    # Sigmoid: 1 / (1 + exp(steepness * (affinity - midpoint)))
    return 1.0 / (1.0 + math.exp(steepness * (affinity_kcal_mol - midpoint)))


def normalise_sa(sa_score: float) -> float:
    """Map SA score (1–10) to [0, 1] where 1 (easy) → 1.0."""
    return max(0.0, min(1.0, 1.0 - (sa_score - 1.0) / 9.0))


def score_docking_result(
    result: DockingResult,
    weights: ScoringWeights | None = None,
) -> CompositeScore:
    """Compute a composite score for a docked ligand.

    Parameters
    ----------
    result:
        Docking result containing poses and SMILES.
    weights:
        Component weights (default: affinity 40%, QED 25%, SA 15%,
        filters 20%).

    Returns
    -------
    CompositeScore
    """
    if weights is None:
        weights = ScoringWeights()

    # ── Vina affinity ───────────────────────────────────────────────
    best_affinity = (
        result.poses[0].affinity_kcal_mol if result.poses else 0.0
    )
    norm_aff = normalise_affinity(best_affinity)

    # ── Drug-likeness from MolScreen ────────────────────────────────
    qed_val = 0.0
    sa_val = 5.0  # mid-range default
    filter_rate = 0.0
    drug_likeness = "unknown"

    smiles = result.ligand_smiles
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            qed_val, sa_val, filter_rate, drug_likeness = _molscreen_assess(mol)

    norm_q = qed_val  # QED already in [0, 1]
    norm_s = normalise_sa(sa_val)
    norm_f = filter_rate  # already a fraction

    composite = (
        weights.affinity * norm_aff
        + weights.qed * norm_q
        + weights.sa * norm_s
        + weights.filters * norm_f
    )

    return CompositeScore(
        ligand_name=result.ligand_name,
        smiles=smiles,
        affinity_kcal_mol=best_affinity,
        qed=qed_val,
        sa_score=sa_val,
        filter_pass_rate=filter_rate,
        norm_affinity=round(norm_aff, 4),
        norm_qed=round(norm_q, 4),
        norm_sa=round(norm_s, 4),
        norm_filters=round(norm_f, 4),
        composite=round(composite, 4),
        drug_likeness=drug_likeness,
    )


def rank_results(
    results: list[DockingResult],
    weights: ScoringWeights | None = None,
) -> list[CompositeScore]:
    """Score and rank a list of docking results.

    Parameters
    ----------
    results:
        Docking results (e.g. from a screening run).
    weights:
        Scoring weights.

    Returns
    -------
    list[CompositeScore]
        Sorted by descending composite score (best first).
    """
    scores = [score_docking_result(r, weights) for r in results]
    scores.sort(key=lambda s: s.composite, reverse=True)

    for i, s in enumerate(scores, 1):
        s.rank = i

    return scores


# ── MolScreen integration ───────────────────────────────────────────

def _molscreen_assess(mol: Chem.Mol) -> tuple[float, float, float, str]:
    """Get QED, SA score, filter pass rate and drug-likeness category.

    Returns
    -------
    tuple of (qed, sa_score, filter_pass_rate, drug_likeness_category)
    """
    try:
        from modules.molscreen.accessibility import qed_score, sa_score
        from modules.molscreen.filters import classify_drug_likeness, run_all_filters

        qed_val = qed_score(mol)
        sa_val = sa_score(mol)

        filter_results = run_all_filters(mol)
        total = len(filter_results)
        passed = sum(1 for f in filter_results if f.passed)
        filter_rate = passed / total if total > 0 else 0.0

        drug_cat = classify_drug_likeness(filter_results)

        return qed_val, sa_val, filter_rate, str(drug_cat)

    except ImportError:
        logger.debug("MolScreen not available — using RDKit built-ins only.")
        from rdkit.Chem.QED import qed

        qed_val = qed(mol)

        try:
            from rdkit.Contrib.SA_Score import sascorer  # type: ignore[import-untyped]
            sa_val = sascorer.calculateScore(mol)
        except ImportError:
            sa_val = 5.0

        return qed_val, sa_val, 0.5, "unknown"
