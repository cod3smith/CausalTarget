"""
Tests for neorx.data_sources
=====================================

Tests for all 7 biomedical data source clients.
Each client must:
1. Return (list[GraphNode], list[GraphEdge]) or dict
2. Never crash — fallback to mock data on API failure
3. Return scientifically plausible data
"""

import pytest
from modules.neorx.models import GraphNode, GraphEdge, NodeType, EdgeType


class TestMonarch:
    def test_returns_nodes_and_edges(self):
        from modules.neorx.data_sources.monarch import query_monarch
        nodes, edges = query_monarch("HIV", allow_mocks=True)
        assert len(nodes) > 0
        assert len(edges) > 0
        assert all(isinstance(n, GraphNode) for n in nodes)
        assert all(isinstance(e, GraphEdge) for e in edges)

    def test_gene_nodes_returned(self):
        from modules.neorx.data_sources.monarch import query_monarch
        nodes, _ = query_monarch("HIV", allow_mocks=True)
        gene_names = {n.name for n in nodes}
        # Should return at least one gene (mock is disease-agnostic)
        assert len(gene_names) >= 1

    def test_scores_in_range(self):
        from modules.neorx.data_sources.monarch import query_monarch
        nodes, _ = query_monarch("HIV", allow_mocks=True)
        for n in nodes:
            assert 0.0 <= n.score <= 1.0

    def test_works_for_any_disease(self):
        """Mock data should work for arbitrary diseases, not just curated ones."""
        from modules.neorx.data_sources.monarch import query_monarch
        for disease in ["HIV", "scurvy", "tularemia", "fibromyalgia"]:
            nodes, edges = query_monarch(disease, allow_mocks=True)
            assert len(nodes) > 0, f"No mock data for '{disease}'"


class TestOpenTargets:
    def test_returns_nodes_and_edges(self):
        from modules.neorx.data_sources.open_targets import query_open_targets
        nodes, edges = query_open_targets("HIV", allow_mocks=True)
        assert len(nodes) > 0
        assert all(isinstance(n, GraphNode) for n in nodes)

    def test_gene_nodes_returned(self):
        from modules.neorx.data_sources.open_targets import query_open_targets
        nodes, _ = query_open_targets("HIV", allow_mocks=True)
        gene_names = {n.name for n in nodes}
        # Should return at least one gene (mock is disease-agnostic)
        assert len(gene_names) >= 1

    def test_works_for_any_disease(self):
        """Mock data should work for arbitrary diseases."""
        from modules.neorx.data_sources.open_targets import query_open_targets
        for disease in ["HIV", "scurvy", "tularemia"]:
            nodes, edges = query_open_targets(disease, allow_mocks=True)
            assert len(nodes) > 0, f"No mock data for '{disease}'"


class TestKEGG:
    def test_returns_pathways(self):
        from modules.neorx.data_sources.kegg import query_kegg_pathways
        nodes, edges = query_kegg_pathways(["CCR5", "CD4", "TNF"], allow_mocks=True)
        assert len(nodes) > 0
        pathway_nodes = [n for n in nodes if n.node_type == NodeType.PATHWAY]
        assert len(pathway_nodes) > 0

    def test_edges_are_participates_in(self):
        from modules.neorx.data_sources.kegg import query_kegg_pathways
        _, edges = query_kegg_pathways(["CCR5"], allow_mocks=True)
        for e in edges:
            assert e.edge_type == EdgeType.PARTICIPATES_IN


class TestReactome:
    def test_returns_pathways(self):
        from modules.neorx.data_sources.reactome import query_reactome_pathways
        nodes, edges = query_reactome_pathways(["TP53", "CCR5"], allow_mocks=True)
        assert len(nodes) > 0

    def test_edges_exist(self):
        from modules.neorx.data_sources.reactome import query_reactome_pathways
        _, edges = query_reactome_pathways(["TP53"], allow_mocks=True)
        assert len(edges) > 0


class TestSTRING:
    def test_returns_interactions(self):
        from modules.neorx.data_sources.string_db import query_string_interactions
        nodes, edges = query_string_interactions(["CCR5", "CD4", "TNF"], allow_mocks=True)
        assert len(edges) > 0

    def test_edge_type_is_interacts(self):
        from modules.neorx.data_sources.string_db import query_string_interactions
        _, edges = query_string_interactions(["CCR5"], allow_mocks=True)
        for e in edges:
            assert e.edge_type == EdgeType.INTERACTS_WITH


class TestUniProt:
    def test_returns_protein_info(self):
        from modules.neorx.data_sources.uniprot import query_uniprot
        results = query_uniprot(["CCR5", "CD4"], allow_mocks=True)
        assert len(results) > 0
        assert "CCR5" in results

    def test_has_pdb_ids(self):
        from modules.neorx.data_sources.uniprot import query_uniprot
        results = query_uniprot(["CCR5"], allow_mocks=True)
        assert len(results["CCR5"]["pdb_ids"]) > 0

    def test_has_druggable_field(self):
        from modules.neorx.data_sources.uniprot import query_uniprot
        results = query_uniprot(["CCR5"], allow_mocks=True)
        assert "is_druggable" in results["CCR5"]

    def test_has_uniprot_id(self):
        from modules.neorx.data_sources.uniprot import query_uniprot
        results = query_uniprot(["CCR5"], allow_mocks=True)
        assert results["CCR5"]["uniprot_id"]  # non-empty string

    def test_works_for_any_gene(self):
        """Mock data should work for arbitrary genes."""
        from modules.neorx.data_sources.uniprot import query_uniprot
        results = query_uniprot(["ABCXYZ", "FOOBR1"], allow_mocks=True)
        assert len(results) == 2


class TestPDB:
    def test_returns_structures(self):
        from modules.neorx.data_sources.pdb import query_pdb_structures
        results = query_pdb_structures({"CCR5": "P51681"}, allow_mocks=True)
        assert len(results) > 0
        assert "CCR5" in results

    def test_structures_have_pdb_id(self):
        from modules.neorx.data_sources.pdb import query_pdb_structures
        results = query_pdb_structures({"CCR5": "P51681"}, allow_mocks=True)
        for s in results["CCR5"]:
            assert "pdb_id" in s
            assert len(s["pdb_id"]) == 4

    def test_structures_have_ligand_field(self):
        from modules.neorx.data_sources.pdb import query_pdb_structures
        results = query_pdb_structures({"CCR5": "P51681"}, allow_mocks=True)
        for s in results["CCR5"]:
            assert "has_ligand" in s

    def test_works_for_any_gene(self):
        """Mock data should work for arbitrary genes."""
        from modules.neorx.data_sources.pdb import query_pdb_structures
        for gene, uid in [("ABCXYZ", "Q99999"), ("FOOBR1", "P12345")]:
            results = query_pdb_structures({gene: uid}, allow_mocks=True)
            assert len(results.get(gene, [])) > 0, f"No mock data for '{gene}'"
