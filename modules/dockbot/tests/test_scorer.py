"""
DockBot Tests — Composite Scorer
===================================

Tests for the scoring normalisation functions and composite scoring.
"""

from __future__ import annotations

import pytest

from modules.dockbot.scorer import (
    normalise_affinity,
    normalise_sa,
    CompositeScore,
    ScoringWeights,
    score_docking_result,
)
from modules.dockbot.models import DockingPose, DockingResult


class TestNormaliseAffinity:
    """Test sigmoid normalisation of Vina affinities."""

    def test_strong_binder(self):
        # -10 kcal/mol should score high
        score = normalise_affinity(-10.0)
        assert score > 0.9

    def test_weak_binder(self):
        # -3 kcal/mol should score low
        score = normalise_affinity(-3.0)
        assert score < 0.1

    def test_midpoint(self):
        # At the midpoint (-7) score should be ~0.5
        score = normalise_affinity(-7.0)
        assert 0.45 < score < 0.55

    def test_zero_affinity(self):
        score = normalise_affinity(0.0)
        assert 0.0 <= score <= 1.0


class TestNormaliseSA:
    """Test SA score normalisation."""

    def test_easy_synthesis(self):
        # SA = 1 (easiest) -> 1.0
        assert normalise_sa(1.0) == 1.0

    def test_hard_synthesis(self):
        # SA = 10 (hardest) -> 0.0
        assert normalise_sa(10.0) == 0.0

    def test_mid_range(self):
        score = normalise_sa(5.0)
        assert 0.4 < score < 0.6


class TestCompositeScoring:
    """Test end-to-end composite scoring."""

    def test_scores_a_result(self):
        result = DockingResult(
            ligand_name="aspirin",
            ligand_smiles="CC(=O)Oc1ccccc1C(O)=O",
            poses=[
                DockingPose(rank=1, affinity_kcal_mol=-7.5),
                DockingPose(rank=2, affinity_kcal_mol=-6.8),
            ],
        )

        score = score_docking_result(result)
        assert isinstance(score, CompositeScore)
        assert score.composite > 0.0
        assert score.ligand_name == "aspirin"

    def test_custom_weights(self):
        result = DockingResult(
            ligand_name="test",
            ligand_smiles="CC(=O)Oc1ccccc1C(O)=O",
            poses=[DockingPose(rank=1, affinity_kcal_mol=-8.0)],
        )

        w1 = ScoringWeights(affinity=1.0, qed=0.0, sa=0.0, filters=0.0)
        s1 = score_docking_result(result, weights=w1)

        w2 = ScoringWeights(affinity=0.0, qed=1.0, sa=0.0, filters=0.0)
        s2 = score_docking_result(result, weights=w2)

        # Pure affinity weight vs pure QED weight should give different scores
        assert s1.composite != s2.composite

    def test_no_poses(self):
        result = DockingResult(
            ligand_name="empty",
            ligand_smiles="CC",
            poses=[],
        )
        score = score_docking_result(result)
        assert isinstance(score, CompositeScore)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
