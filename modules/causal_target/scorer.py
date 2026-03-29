"""
Composite Drug-Candidate Scorer
================================

The scorer ranks drug candidates across **six dimensions**:

1. **Causal confidence** (0.30) — Is the target genuinely causal?
2. **Binding affinity** (0.25) — How tightly does it bind?
3. **Drug-likeness (QED)** (0.15) — Lipinski / Veber / QED
4. **Synthetic accessibility** (0.10) — Can it be synthesised?
5. **ADMET** (0.10) — Predicted pharmacokinetics
6. **Novelty** (0.10) — Structural novelty vs known drugs

The critical design decision:  **causal confidence has the
highest weight**.  A moderately-binding molecule aimed at a
validated causal target is ranked above a tightly-binding
molecule aimed at a correlational bystander.  This is what
distinguishes CausalTarget from conventional virtual screening
pipelines.

Normalisation
-------------
Raw scores live on different scales:
- Binding affinity:  −12 to 0 kcal/mol (more negative = better)
- QED:  0 to 1
- SA score:  1 to 10 (lower = easier to synthesise)
- ADMET:  0 to 1
- Novelty:  0 to 1

We normalise everything to [0, 1] where 1 = best before
applying weights.
"""

from __future__ import annotations

import logging
from typing import Any

from .models import ScoredCandidate

logger = logging.getLogger(__name__)

# ── Score Weights ──────────────────────────────────────────────────

WEIGHTS = {
    "causal_confidence": 0.30,
    "binding_affinity": 0.25,
    "qed": 0.15,
    "sa": 0.10,
    "admet": 0.10,
    "novelty": 0.10,
}

# ── Normalisation Constants ────────────────────────────────────────

# Binding affinity: typical range −12 to 0 kcal/mol
AFFINITY_BEST = -12.0   # Best possible
AFFINITY_WORST = 0.0     # No binding

# SA score: 1 (easy) to 10 (hard)
SA_BEST = 1.0
SA_WORST = 10.0


def score_candidate(
    smiles: str,
    target_protein_id: str,
    target_protein_name: str,
    causal_confidence: float,
    binding_affinity: float | None = None,
    qed_score: float | None = None,
    sa_score: float | None = None,
    admet_score: float = 0.5,
    novelty_score: float = 0.5,
    molecular_weight: float | None = None,
    logp: float | None = None,
    drug_likeness_class: str = "",
    n_filters_passed: int = 0,
) -> ScoredCandidate:
    """Score a single drug candidate across all dimensions.

    Parameters
    ----------
    smiles : str
        SMILES string of the candidate molecule.
    target_protein_id : str
        Target protein identifier.
    target_protein_name : str
        Target protein name.
    causal_confidence : float
        Causal confidence for the target (0–1).
    binding_affinity : float, optional
        Docking score in kcal/mol (negative = better).
    qed_score : float, optional
        Quantitative Estimate of Drug-likeness (0–1).
    sa_score : float, optional
        Synthetic accessibility (1=easy, 10=hard).
    admet_score : float
        Predicted ADMET score (0–1).
    novelty_score : float
        Structural novelty vs known drugs (0–1).
    molecular_weight : float, optional
        Molecular weight in Da.
    logp : float, optional
        Predicted logP.
    drug_likeness_class : str
        Drug-likeness classification.
    n_filters_passed : int
        Number of drug-likeness filters passed.

    Returns
    -------
    ScoredCandidate
        Fully scored candidate with composite score.
    """
    # Normalise individual scores to [0, 1]
    norm_causal = _clamp(causal_confidence)
    norm_binding = normalise_affinity(binding_affinity) if binding_affinity is not None else 0.5
    norm_qed = _clamp(qed_score) if qed_score is not None else 0.5
    norm_sa = normalise_sa(sa_score) if sa_score is not None else 0.5
    norm_admet = _clamp(admet_score)
    norm_novelty = _clamp(novelty_score)

    # Compute weighted composite
    composite = (
        WEIGHTS["causal_confidence"] * norm_causal
        + WEIGHTS["binding_affinity"] * norm_binding
        + WEIGHTS["qed"] * norm_qed
        + WEIGHTS["sa"] * norm_sa
        + WEIGHTS["admet"] * norm_admet
        + WEIGHTS["novelty"] * norm_novelty
    )

    # Build breakdown
    breakdown = {
        "causal_confidence": round(WEIGHTS["causal_confidence"] * norm_causal, 4),
        "binding_affinity": round(WEIGHTS["binding_affinity"] * norm_binding, 4),
        "qed": round(WEIGHTS["qed"] * norm_qed, 4),
        "sa": round(WEIGHTS["sa"] * norm_sa, 4),
        "admet": round(WEIGHTS["admet"] * norm_admet, 4),
        "novelty": round(WEIGHTS["novelty"] * norm_novelty, 4),
    }

    is_drug_like = (
        norm_qed >= 0.4
        and (molecular_weight is not None and 150 <= molecular_weight <= 500)
        and n_filters_passed >= 2
    )

    return ScoredCandidate(
        smiles=smiles,
        target_protein_id=target_protein_id,
        target_protein_name=target_protein_name,
        causal_confidence=causal_confidence,
        binding_affinity=binding_affinity,
        qed_score=qed_score,
        sa_score=sa_score,
        admet_score=admet_score,
        novelty_score=novelty_score,
        composite_score=round(composite, 4),
        score_breakdown=breakdown,
        molecular_weight=molecular_weight,
        logp=logp,
        drug_likeness_class=drug_likeness_class,
        n_filters_passed=n_filters_passed,
        is_drug_like=is_drug_like,
        is_novel=novelty_score >= 0.7,
    )


def rank_candidates(
    candidates: list[ScoredCandidate],
) -> list[ScoredCandidate]:
    """Sort candidates by composite score and assign ranks.

    Parameters
    ----------
    candidates : list[ScoredCandidate]
        Unranked scored candidates.

    Returns
    -------
    list[ScoredCandidate]
        Ranked (best-first) with ``.rank`` set.
    """
    sorted_cands = sorted(
        candidates, key=lambda c: c.composite_score, reverse=True,
    )
    for i, cand in enumerate(sorted_cands, 1):
        cand.rank = i
    return sorted_cands


# ── Normalisation Functions ────────────────────────────────────────

def normalise_affinity(affinity_kcal: float) -> float:
    """Normalise binding affinity to [0, 1].

    More negative → better → higher score.

    Examples:
    - −12 kcal/mol → 1.0  (excellent binder)
    - −6 kcal/mol  → 0.5  (moderate binder)
    -  0 kcal/mol  → 0.0  (no binding)
    """
    return _clamp(
        (affinity_kcal - AFFINITY_WORST) / (AFFINITY_BEST - AFFINITY_WORST)
    )


def normalise_sa(sa_score: float) -> float:
    """Normalise synthetic accessibility to [0, 1].

    Lower SA → easier to synthesise → higher score.

    Examples:
    - SA = 1  → 1.0  (trivially synthesisable)
    - SA = 5  → 0.56 (moderate)
    - SA = 10 → 0.0  (practically unsynthesisable)
    """
    return _clamp(
        (SA_WORST - sa_score) / (SA_WORST - SA_BEST)
    )


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to [lo, hi]."""
    return max(lo, min(hi, value))
