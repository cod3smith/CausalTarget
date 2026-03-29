"""
Tests for causal_target.data_sources
=====================================

Tests for all 7 biomedical data source clients.
Each client must:
1. Return (list[GraphNode], list[GraphEdge]) or dict
2. Never crash — fallback to mock data on API failure
3. Return scientifically plausible data
"""

import pytest
from modules.causal_target.models import GraphNode, GraphEdge, NodeType, EdgeType


class TestDisGeNET:
    def test_returns_nodes_and_edges(self):
        from modules.causal_target.data_sources.disgenet import query_disgenet
        nodes, edges = query_disgenet("HIV")
        assert len(nodes) > 0
        assert len(edges) > 0
        assert all(isinstance(n, GraphNode) for n in nodes)
        assert all(isinstance(e, GraphEdge) for e in edges)

    def test_hiv_includes_ccr5(self):
        from modules.causal_target.data_sources.disgenet import query_disgenet
        nodes, _ = query_disgenet("HIV")
        gene_names = {n.name for n in nodes}
        assert "CCR5" in gene_names

    def test_scores_in_range(self):
        from modules.causal_target.data_sources.disgenet import query_disgenet
        nodes, _ = query_disgenet("HIV")
        for n in nodes:
            assert 0.0 <= n.score <= 1.0


class TestOpenTargets:
    def test_returns_nodes_and_edges(self):
        from modules.causal_target.data_sources.open_targets import query_open_targets
        nodes, edges = query_open_targets("HIV")
        assert len(nodes) > 0
        assert all(isinstance(n, GraphNode) for n in nodes)

    def test_hiv_includes_ccr5(self):
        from modules.causal_target.data_sources.open_targets import query_open_targets
        nodes, _ = query_open_targets("HIV")
        gene_names = {n.name for n in nodes}
        assert "CCR5" in gene_names


class TestKEGG:
    def test_returns_pathways(self):
        from modules.causal_target.data_sources.kegg import query_kegg_pathways
        nodes, edges = query_kegg_pathways(["CCR5", "CD4", "TNF"])
        assert len(nodes) > 0
        pathway_nodes = [n for n in nodes if n.node_type == NodeType.PATHWAY]
        assert len(pathway_nodes) > 0

    def test_edges_are_participates_in(self):
        from modules.causal_target.data_sources.kegg import query_kegg_pathways
        _, edges = query_kegg_pathways(["CCR5"])
        for e in edges:
            assert e.edge_type == EdgeType.PARTICIPATES_IN


class TestReactome:
    def test_returns_pathways(self):
        from modules.causal_target.data_sources.reactome import query_reactome_pathways
        nodes, edges = query_reactome_pathways(["TP53", "CCR5"])
        assert len(nodes) > 0

    def test_edges_exist(self):
        from modules.causal_target.data_sources.reactome import query_reactome_pathways
        _, edges = query_reactome_pathways(["TP53"])
        assert len(edges) > 0


class TestSTRING:
    def test_returns_interactions(self):
        from modules.causal_target.data_sources.string_db import query_string_interactions
        nodes, edges = query_string_interactions(["CCR5", "CD4", "TNF"])
        assert len(edges) > 0

    def test_edge_type_is_interacts(self):
        from modules.causal_target.data_sources.string_db import query_string_interactions
        _, edges = query_string_interactions(["CCR5"])
        for e in edges:
            assert e.edge_type == EdgeType.INTERACTS_WITH


class TestUniProt:
    def test_returns_protein_info(self):
        from modules.causal_target.data_sources.uniprot import query_uniprot
        results = query_uniprot(["CCR5", "CD4"])
        assert len(results) > 0
        assert "CCR5" in results

    def test_ccr5_has_pdb_ids(self):
        from modules.causal_target.data_sources.uniprot import query_uniprot
        results = query_uniprot(["CCR5"])
        assert len(results["CCR5"]["pdb_ids"]) > 0

    def test_ccr5_is_druggable(self):
        from modules.causal_target.data_sources.uniprot import query_uniprot
        results = query_uniprot(["CCR5"])
        assert results["CCR5"]["is_druggable"] is True

    def test_ccr5_has_uniprot_id(self):
        from modules.causal_target.data_sources.uniprot import query_uniprot
        results = query_uniprot(["CCR5"])
        assert results["CCR5"]["uniprot_id"] == "P51681"


class TestPDB:
    def test_returns_structures(self):
        from modules.causal_target.data_sources.pdb import query_pdb_structures
        results = query_pdb_structures({"CCR5": "P51681"})
        assert len(results) > 0
        assert "CCR5" in results

    def test_structures_have_pdb_id(self):
        from modules.causal_target.data_sources.pdb import query_pdb_structures
        results = query_pdb_structures({"CCR5": "P51681"})
        for s in results["CCR5"]:
            assert "pdb_id" in s
            assert len(s["pdb_id"]) == 4

    def test_ccr5_has_ligand_structures(self):
        from modules.causal_target.data_sources.pdb import query_pdb_structures
        results = query_pdb_structures({"CCR5": "P51681"})
        has_ligand = any(s["has_ligand"] for s in results["CCR5"])
        assert has_ligand
