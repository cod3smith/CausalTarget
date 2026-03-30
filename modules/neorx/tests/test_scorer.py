"""
Tests for neorx.scorer
================================
"""

import pytest
from modules.neorx.scorer import (
    score_candidate,
    rank_candidates,
    normalise_affinity,
    normalise_sa,
    DEFAULT_WEIGHTS,
)
from modules.neorx.models import ScoredCandidate


class TestNormaliseAffinity:
    """Test binding affinity normalisation."""

    def test_excellent_binder(self):
        """−12 kcal/mol should normalise to ~1.0."""
        assert normalise_affinity(-12.0) == pytest.approx(1.0)

    def test_no_binding(self):
        """0 kcal/mol should normalise to 0.0."""
        assert normalise_affinity(0.0) == pytest.approx(0.0)

    def test_moderate_binder(self):
        """−6 kcal/mol should normalise to ~0.5."""
        assert normalise_affinity(-6.0) == pytest.approx(0.5)

    def test_clamped_positive(self):
        """Positive affinity (non-physical) → 0.0."""
        assert normalise_affinity(5.0) == pytest.approx(0.0)

    def test_clamped_extreme(self):
        """Beyond −12 should clamp to 1.0."""
        assert normalise_affinity(-15.0) == pytest.approx(1.0)


class TestNormaliseSA:
    """Test synthetic accessibility normalisation."""

    def test_easy_synthesis(self):
        """SA = 1 → 1.0 (trivially synthesisable)."""
        assert normalise_sa(1.0) == pytest.approx(1.0)

    def test_hard_synthesis(self):
        """SA = 10 → 0.0 (unsynthesisable)."""
        assert normalise_sa(10.0) == pytest.approx(0.0)

    def test_moderate(self):
        """SA = 5.5 → ~0.5."""
        assert normalise_sa(5.5) == pytest.approx(0.5)


class TestWeights:
    """Test that weights sum to 1.0."""

    def test_weights_sum_to_one(self):
        total = sum(DEFAULT_WEIGHTS.values())
        assert total == pytest.approx(1.0)

    def test_causal_confidence_highest(self):
        """Causal confidence must be the highest-weighted dimension."""
        max_weight = max(DEFAULT_WEIGHTS.values())
        assert DEFAULT_WEIGHTS["causal_confidence"] == max_weight


class TestScoreCandidate:
    """Test single-candidate scoring."""

    def test_basic_scoring(self):
        cand = score_candidate(
            smiles="c1ccccc1",
            target_protein_id="gene:CCR5",
            target_protein_name="CCR5",
            causal_confidence=0.9,
            binding_affinity=-8.0,
            qed_score=0.7,
            sa_score=3.0,
            admet_score=0.8,
            novelty_score=0.6,
            molecular_weight=250.0,
            logp=2.5,
            n_filters_passed=4,
        )
        assert isinstance(cand, ScoredCandidate)
        assert 0.0 < cand.composite_score <= 1.0
        assert cand.smiles == "c1ccccc1"

    def test_score_breakdown_sums_to_composite(self):
        cand = score_candidate(
            smiles="c1ccccc1",
            target_protein_id="gene:X",
            target_protein_name="X",
            causal_confidence=0.8,
            binding_affinity=-7.0,
            qed_score=0.6,
            sa_score=4.0,
        )
        breakdown_sum = sum(cand.score_breakdown.values())
        assert breakdown_sum == pytest.approx(cand.composite_score, abs=0.01)

    def test_higher_causal_confidence_gives_higher_score(self):
        """A candidate with higher causal confidence should score higher."""
        cand_high = score_candidate(
            smiles="c1ccccc1", target_protein_id="A",
            target_protein_name="A", causal_confidence=0.9,
            binding_affinity=-6.0, qed_score=0.5,
        )
        cand_low = score_candidate(
            smiles="c1ccccc1", target_protein_id="B",
            target_protein_name="B", causal_confidence=0.2,
            binding_affinity=-6.0, qed_score=0.5,
        )
        assert cand_high.composite_score > cand_low.composite_score

    def test_causal_beats_binding(self):
        """A moderate binder with high causal confidence should rank
        above a strong binder with low causal confidence.

        This is the KEY design decision of NeoRx.
        """
        # Moderate binder, high causal
        cand_causal = score_candidate(
            smiles="c1ccccc1", target_protein_id="A",
            target_protein_name="A", causal_confidence=0.95,
            binding_affinity=-6.0,  # Moderate
        )
        # Strong binder, low causal
        cand_binding = score_candidate(
            smiles="c1ccccc1", target_protein_id="B",
            target_protein_name="B", causal_confidence=0.2,
            binding_affinity=-11.0,  # Excellent
        )
        assert cand_causal.composite_score > cand_binding.composite_score

    def test_missing_scores_use_defaults(self):
        """When optional scores are None, defaults should be used."""
        cand = score_candidate(
            smiles="C", target_protein_id="X",
            target_protein_name="X", causal_confidence=0.5,
        )
        assert cand.composite_score > 0.0

    def test_drug_likeness_flag(self):
        """Drug-like flag requires QED, MW, and filter criteria."""
        cand = score_candidate(
            smiles="c1ccccc1", target_protein_id="X",
            target_protein_name="X", causal_confidence=0.5,
            qed_score=0.7, molecular_weight=300.0, n_filters_passed=4,
        )
        assert cand.is_drug_like is True

    def test_not_drug_like_when_mw_too_high(self):
        cand = score_candidate(
            smiles="c1ccccc1", target_protein_id="X",
            target_protein_name="X", causal_confidence=0.5,
            qed_score=0.7, molecular_weight=800.0, n_filters_passed=4,
        )
        assert cand.is_drug_like is False

    def test_novelty_flag(self):
        cand = score_candidate(
            smiles="c1ccccc1", target_protein_id="X",
            target_protein_name="X", causal_confidence=0.5,
            novelty_score=0.8,
        )
        assert cand.is_novel is True


class TestRankCandidates:
    """Test candidate ranking."""

    def test_assigns_ranks(self):
        candidates = [
            score_candidate("A", "X", "X", 0.3),
            score_candidate("B", "X", "X", 0.9),
            score_candidate("C", "X", "X", 0.6),
        ]
        ranked = rank_candidates(candidates)
        assert ranked[0].rank == 1
        assert ranked[1].rank == 2
        assert ranked[2].rank == 3

    def test_sorted_descending(self):
        candidates = [
            score_candidate("A", "X", "X", 0.3),
            score_candidate("B", "X", "X", 0.9),
            score_candidate("C", "X", "X", 0.6),
        ]
        ranked = rank_candidates(candidates)
        scores = [c.composite_score for c in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_empty_list(self):
        ranked = rank_candidates([])
        assert ranked == []
