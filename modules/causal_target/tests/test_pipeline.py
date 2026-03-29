"""
Tests for causal_target.pipeline
==================================

Integration tests for the full pipeline and its sub-components.
Uses the session-scoped ``hiv_graph`` fixture so the graph is
built ONCE rather than once per test.
"""

import pytest
from modules.causal_target.pipeline import (
    run_pipeline,
    _fallback_molecules,
    _estimate_admet,
)
from modules.causal_target.models import (
    PipelineResult,
    JobStatus,
)


class TestRunPipeline:
    """End-to-end pipeline tests.

    All HIV tests pass ``prebuilt_graph`` to avoid redundant API calls.
    """

    def test_pipeline_completes(self, hiv_graph):
        """The pipeline should complete without errors."""
        result = run_pipeline(
            "HIV",
            top_n_targets=3,
            candidates_per_target=10,
            generate_molecules=True,
            run_docking=False,  # Skip docking for speed
            generate_report=False,
            prebuilt_graph=hiv_graph,
        )
        assert isinstance(result, PipelineResult)
        assert result.job.status == JobStatus.COMPLETE

    def test_pipeline_has_graph(self, hiv_graph):
        result = run_pipeline(
            "HIV", top_n_targets=2, candidates_per_target=10,
            generate_molecules=False, run_docking=False,
            generate_report=False, prebuilt_graph=hiv_graph,
        )
        assert result.graph is not None
        assert len(result.graph.nodes) > 0

    def test_pipeline_has_causal_targets(self, hiv_graph):
        result = run_pipeline(
            "HIV", top_n_targets=5, candidates_per_target=10,
            generate_molecules=False, run_docking=False,
            generate_report=False, prebuilt_graph=hiv_graph,
        )
        assert len(result.causal_targets) > 0

    def test_pipeline_with_generation(self, hiv_graph):
        """Pipeline with molecule generation should produce candidates."""
        result = run_pipeline(
            "HIV", top_n_targets=2, candidates_per_target=10,
            generate_molecules=True, run_docking=False,
            generate_report=False, prebuilt_graph=hiv_graph,
        )
        # May have candidates if any targets are causal
        if result.n_causal_targets > 0:
            assert len(result.scored_candidates) > 0

    def test_pipeline_disease_stored(self, hiv_graph):
        result = run_pipeline(
            "HIV", top_n_targets=2, candidates_per_target=10,
            generate_molecules=False, run_docking=False,
            generate_report=False, prebuilt_graph=hiv_graph,
        )
        assert result.disease == "HIV"

    def test_pipeline_job_id_set(self, hiv_graph):
        result = run_pipeline(
            "HIV", top_n_targets=2, candidates_per_target=10,
            generate_molecules=False, run_docking=False,
            generate_report=False, prebuilt_graph=hiv_graph,
        )
        assert result.job.job_id is not None
        assert len(result.job.job_id) > 0

    def test_pipeline_different_disease(self):
        """Separate disease — requires its own API calls."""
        result = run_pipeline(
            "Type 2 Diabetes", top_n_targets=2, candidates_per_target=10,
            generate_molecules=False, run_docking=False,
            generate_report=False,
        )
        assert result.disease == "Type 2 Diabetes"
        assert result.graph is not None


class TestFallbackMolecules:
    """Test the fallback molecule generator."""

    def test_returns_molecules(self):
        mols = _fallback_molecules("CCR5")
        assert len(mols) > 0
        assert all(isinstance(s, str) for s in mols)

    def test_ccr5_has_specific_scaffolds(self):
        mols = _fallback_molecules("CCR5")
        assert len(mols) > 10  # General + CCR5-specific

    def test_unknown_gene_returns_general(self):
        mols = _fallback_molecules("UNKNOWN_GENE")
        assert len(mols) > 0  # At least general scaffolds


class TestEstimateAdmet:
    """Test ADMET estimation heuristic."""

    def test_good_properties(self):
        score = _estimate_admet(mw=300.0, logp=2.5, qed=0.7)
        assert score >= 0.7

    def test_bad_mw(self):
        score = _estimate_admet(mw=800.0, logp=2.5, qed=0.7)
        assert score < 0.9  # Penalised

    def test_bad_logp(self):
        score = _estimate_admet(mw=300.0, logp=8.0, qed=0.7)
        assert score < 0.9  # Penalised

    def test_none_values(self):
        score = _estimate_admet(mw=None, logp=None, qed=None)
        assert 0.0 <= score <= 1.0

    def test_clamped(self):
        score = _estimate_admet(mw=300.0, logp=2.5, qed=0.9)
        assert 0.0 <= score <= 1.0
