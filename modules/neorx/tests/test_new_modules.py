"""
Tests for new NeoRx modules
=====================================

Covers cache.py, persistence.py, admet.py, configurable weights,
SMILES canonicalization, confidence intervals, and resolve_disease_id.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest
import networkx as nx

# ── cache ──────────────────────────────────────────────────
from modules.neorx.cache import (
    FileCache,
    get_cache,
    reset_cache_instance,
    _cache_key,
)


class TestCacheKey:
    """Test cache key generation."""

    def test_deterministic(self):
        k1 = _cache_key("hello")
        k2 = _cache_key("hello")
        assert k1 == k2

    def test_different_inputs(self):
        k1 = _cache_key("alpha")
        k2 = _cache_key("beta")
        assert k1 != k2


class TestFileCache:
    """Test file-based cache backend."""

    def test_get_set(self, tmp_path: Path):
        cache = FileCache(cache_dir=tmp_path)
        cache.set("k1", {"value": 42}, ttl=60)
        result = cache.get("k1")
        assert result is not None
        assert result["value"] == 42

    def test_miss(self, tmp_path: Path):
        cache = FileCache(cache_dir=tmp_path)
        assert cache.get("nonexistent") is None

    def test_clear_removes_entries(self, tmp_path: Path):
        cache = FileCache(cache_dir=tmp_path)
        cache.set("k2", "hello", ttl=60)
        assert cache.get("k2") is not None
        cache.clear()
        assert cache.get("k2") is None


class TestGetCache:
    """Test cache singleton factory."""

    def test_returns_file_cache_by_default(self):
        reset_cache_instance()
        old = os.environ.get("NEORX_CACHE_BACKEND")
        os.environ.pop("NEORX_CACHE_BACKEND", None)
        try:
            c = get_cache()
            assert isinstance(c, FileCache)
        finally:
            reset_cache_instance()
            if old is not None:
                os.environ["NEORX_CACHE_BACKEND"] = old


# ── persistence ────────────────────────────────────────────
from modules.neorx.persistence import (
    save_graph,
    load_graph,
    list_saved_graphs,
)
from modules.neorx.models import (
    DiseaseGraph,
    GraphNode,
    GraphEdge,
    NodeType,
    EdgeType,
)


class TestPersistence:
    """Test graph save/load round-trips."""

    @pytest.fixture()
    def sample_graph(self) -> DiseaseGraph:
        nodes = [
            GraphNode(node_id="gene:A", name="A", node_type=NodeType.GENE,
                      source="test", score=0.9),
            GraphNode(node_id="protein:B", name="B", node_type=NodeType.PROTEIN,
                      source="test", score=0.5),
        ]
        edges = [
            GraphEdge(source_id="gene:A", target_id="protein:B",
                      edge_type=EdgeType.INTERACTS_WITH, weight=0.8,
                      source_db="test"),
        ]
        return DiseaseGraph(
            disease_name="test_disease",
            nodes=nodes,
            edges=edges,
            sources_queried=["test"],
        )

    def test_json_round_trip(self, sample_graph: DiseaseGraph, tmp_path: Path):
        path = save_graph(sample_graph, fmt="json", output_dir=tmp_path)
        loaded = load_graph(path)
        assert isinstance(loaded, DiseaseGraph)
        assert loaded.disease_name == "test_disease"
        assert len(loaded.nodes) == 2

    def test_graphml_round_trip(self, sample_graph: DiseaseGraph, tmp_path: Path):
        path = save_graph(sample_graph, fmt="graphml", output_dir=tmp_path)
        assert Path(path).exists()
        assert path.endswith(".graphml")

    def test_list_saved_graphs(self, sample_graph: DiseaseGraph, tmp_path: Path):
        save_graph(sample_graph, fmt="json", output_dir=tmp_path)
        graphs = list_saved_graphs(directory=tmp_path)
        assert len(graphs) >= 1


# ── admet ──────────────────────────────────────────────────
from modules.neorx.admet import predict_admet, ADMETProfile


class TestADMET:
    """Test multi-rule ADMET predictor."""

    def test_aspirin(self):
        """Aspirin should score reasonably well."""
        profile = predict_admet("CC(=O)Oc1ccccc1C(=O)O")
        assert isinstance(profile, ADMETProfile)
        assert 0.0 <= profile.composite <= 1.0
        assert not profile.flags or isinstance(profile.flags, list)

    def test_invalid_smiles_returns_zero(self):
        profile = predict_admet("NOT_A_SMILES")
        assert isinstance(profile, ADMETProfile)
        assert profile.composite == 0.0
        assert "Invalid SMILES" in profile.flags

    def test_caffeine(self):
        """Caffeine — small, drug-like."""
        profile = predict_admet("Cn1c(=O)c2c(ncn2C)n(C)c1=O")
        assert profile.composite > 0.4

    def test_all_scores_in_range(self):
        profile = predict_admet("c1ccccc1")  # benzene
        for field in ["absorption", "distribution", "metabolism",
                      "excretion", "toxicity", "composite"]:
            val = getattr(profile, field)
            assert 0.0 <= val <= 1.0, f"{field}={val} out of range"


# ── configurable weights ───────────────────────────────────
from modules.neorx.scorer import _get_weights, DEFAULT_WEIGHTS


class TestConfigurableWeights:
    """Test weight resolution from overrides/env/defaults."""

    def test_defaults_returned(self):
        w = _get_weights(None)
        assert abs(sum(w.values()) - 1.0) < 1e-6

    def test_override_normalises(self):
        custom = {"causal_confidence": 2.0, "binding_affinity": 1.0}
        w = _get_weights(custom)
        assert abs(sum(w.values()) - 1.0) < 1e-6
        # causal_confidence should be the largest
        assert w["causal_confidence"] > w["binding_affinity"]

    def test_env_var_override(self):
        old = os.environ.get("NEORX_WEIGHTS")
        os.environ["NEORX_WEIGHTS"] = json.dumps(
            {"causal_confidence": 0.5, "qed": 0.5}
        )
        try:
            w = _get_weights(None)
            assert abs(sum(w.values()) - 1.0) < 1e-6
        finally:
            if old is not None:
                os.environ["NEORX_WEIGHTS"] = old
            else:
                os.environ.pop("NEORX_WEIGHTS", None)


# ── SMILES canonicalization ────────────────────────────────
from modules.neorx.pipeline import _canonicalize_smiles


class TestSMILESCanonicalization:
    """Test SMILES canonicalization utility."""

    def test_canonical_form(self):
        # Different representations of the same molecule
        s1 = _canonicalize_smiles("c1ccccc1")
        s2 = _canonicalize_smiles("C1=CC=CC=C1")
        assert s1 == s2

    def test_invalid_returns_none(self):
        assert _canonicalize_smiles("INVALID") is None

    def test_empty_returns_falsy(self):
        result = _canonicalize_smiles("")
        # Empty string or None are both acceptable falsy results
        assert not result or result is None


# ── confidence intervals ──────────────────────────────────
from modules.neorx.models import NeoRxResult


class TestConfidenceInterval:
    """Test confidence_interval field on NeoRxResult."""

    def test_default_ci(self):
        r = NeoRxResult(
            protein_id="test:P1",
            protein_name="Protein1",
            gene_name="GENE1",
            uniprot_id="P00000",
            pdb_ids=[],
            causal_confidence=0.7,
            is_causal_target=True,
        )
        assert r.confidence_interval == (0.0, 1.0)

    def test_custom_ci(self):
        r = NeoRxResult(
            protein_id="test:P1",
            protein_name="Protein1",
            gene_name="GENE1",
            uniprot_id="P00000",
            pdb_ids=[],
            causal_confidence=0.7,
            is_causal_target=True,
            confidence_interval=(0.55, 0.85),
        )
        assert r.confidence_interval == (0.55, 0.85)


# ── resolve_disease_id ─────────────────────────────────────
from modules.neorx.data_sources.open_targets import resolve_disease_id


class TestResolveDiseaseId:
    """Test disease ontology resolver."""

    def test_returns_str_or_none(self):
        # Should not raise even with bad network
        result = resolve_disease_id("nonexistent_disease_xyz")
        assert result is None or isinstance(result, str)
