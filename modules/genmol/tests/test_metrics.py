"""Tests for generation quality metrics."""

import pytest

from modules.genmol.evaluation.metrics import (
    validity,
    uniqueness,
    novelty,
    internal_diversity,
    compute_all_metrics,
)


class TestValidity:
    """Test SMILES validity metric."""

    def test_all_valid(self):
        assert validity(["CCO", "c1ccccc1", "CC(=O)O"]) == 1.0

    def test_none_valid(self):
        assert validity(["INVALID", "XXX", "!!!"]) == 0.0

    def test_partial_valid(self):
        val = validity(["CCO", "INVALID", "c1ccccc1"])
        assert abs(val - 2 / 3) < 1e-6

    def test_empty_list(self):
        assert validity([]) == 0.0


class TestUniqueness:
    """Test uniqueness metric."""

    def test_all_unique(self):
        assert uniqueness(["CCO", "c1ccccc1", "CC(=O)O"]) == 1.0

    def test_duplicates(self):
        # CCO and OCC are the same molecule
        uniq = uniqueness(["CCO", "OCC", "c1ccccc1"])
        assert abs(uniq - 2 / 3) < 1e-6

    def test_empty(self):
        assert uniqueness([]) == 0.0


class TestNovelty:
    """Test novelty metric."""

    def test_all_novel(self):
        training = {"CCO", "c1ccccc1"}
        generated = ["CC(=O)O", "CCCC"]
        assert novelty(generated, training) == 1.0

    def test_none_novel(self):
        training = {"CCO", "c1ccccc1"}
        assert novelty(["CCO", "c1ccccc1"], training) == 0.0

    def test_partial_novel(self):
        training = {"CCO"}
        nov = novelty(["CCO", "c1ccccc1"], training)
        assert abs(nov - 0.5) < 1e-6


class TestDiversity:
    """Test internal diversity metric."""

    def test_identical_molecules(self):
        # All same molecule → diversity near 0
        div = internal_diversity(["CCO", "CCO", "CCO"])
        assert div < 0.01

    def test_diverse_molecules(self):
        # Different molecules → diversity > 0
        smiles = [
            "CCO",
            "c1ccccc1",
            "CC(=O)NC1=CC=C(O)C=C1",
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
        ]
        div = internal_diversity(smiles)
        assert div > 0.1

    def test_single_molecule(self):
        assert internal_diversity(["CCO"]) == 0.0


class TestComputeAll:
    """Test the combined metrics function."""

    def test_returns_all_keys(self):
        metrics = compute_all_metrics(
            ["CCO", "c1ccccc1", "CC(=O)O"],
            ["CCCC", "CCCCC"],
        )
        assert "validity" in metrics
        assert "uniqueness" in metrics
        assert "novelty" in metrics
        assert "diversity" in metrics
        assert "n_generated" in metrics
        assert "n_valid" in metrics
        assert "n_unique" in metrics

    def test_values_in_range(self):
        metrics = compute_all_metrics(
            ["CCO", "c1ccccc1", "CC(=O)O"]
        )
        assert 0.0 <= metrics["validity"] <= 1.0
        assert 0.0 <= metrics["uniqueness"] <= 1.0
        assert 0.0 <= metrics["diversity"] <= 1.0
