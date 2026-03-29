"""
Tests for causal_target.graph_builder
======================================
"""

import pytest
from modules.causal_target.graph_builder import (
    build_disease_graph,
    disease_graph_to_networkx,
    _extract_gene_symbols,
    _merge_nodes,
)
from modules.causal_target.models import (
    DiseaseGraph,
    GraphNode,
    GraphEdge,
    NodeType,
    EdgeType,
)


class TestBuildDiseaseGraph:
    """Test the full graph construction pipeline.

    Uses the session-scoped ``hiv_graph`` fixture from conftest.py
    so the graph is built ONCE across the entire test session.
    """

    def test_build_hiv_graph(self, hiv_graph):
        """Build a graph for HIV — our validation case."""
        assert isinstance(hiv_graph, DiseaseGraph)
        assert hiv_graph.disease_name == "HIV"
        assert len(hiv_graph.nodes) > 0
        assert len(hiv_graph.edges) > 0
        assert hiv_graph.n_genes > 0 or hiv_graph.n_proteins > 0

    def test_graph_has_disease_node(self, hiv_graph):
        """The graph must contain a disease outcome node."""
        disease_nodes = [
            n for n in hiv_graph.nodes if n.node_type == NodeType.DISEASE
        ]
        assert len(disease_nodes) >= 1
        assert "hiv" in disease_nodes[0].node_id.lower()

    def test_graph_sources_queried(self, hiv_graph):
        """All expected data sources should be queried."""
        assert "DisGeNET" in hiv_graph.sources_queried
        assert "OpenTargets" in hiv_graph.sources_queried
        assert "KEGG" in hiv_graph.sources_queried
        assert "Reactome" in hiv_graph.sources_queried

    def test_graph_has_pathways(self, hiv_graph):
        """The graph should include pathway nodes."""
        assert hiv_graph.n_pathways > 0

    def test_graph_edges_reference_existing_nodes(self, hiv_graph):
        """All edge endpoints must exist as nodes."""
        node_ids = {n.node_id for n in hiv_graph.nodes}
        for edge in hiv_graph.edges:
            assert edge.source_id in node_ids, f"Missing source: {edge.source_id}"
            assert edge.target_id in node_ids, f"Missing target: {edge.target_id}"

    def test_build_diabetes_graph(self):
        """Ensure it works for multiple diseases (separate API calls)."""
        graph = build_disease_graph("Type 2 Diabetes")
        assert graph.disease_name == "Type 2 Diabetes"
        assert len(graph.nodes) > 0


class TestDiseaseGraphToNetworkx:
    """Test NetworkX conversion."""

    def test_conversion_preserves_nodes(self, hiv_graph):
        G = disease_graph_to_networkx(hiv_graph)
        assert len(G.nodes) == len(hiv_graph.nodes)

    def test_conversion_preserves_edges(self, hiv_graph):
        G = disease_graph_to_networkx(hiv_graph)
        assert len(G.edges) == len(hiv_graph.edges)

    def test_node_attributes(self, hiv_graph):
        G = disease_graph_to_networkx(hiv_graph)
        for node_id, data in G.nodes(data=True):
            assert "name" in data
            assert "node_type" in data
            assert "score" in data


class TestExtractGeneSymbols:
    """Test gene symbol extraction."""

    def test_extracts_genes(self):
        nodes = [
            GraphNode(node_id="gene:TP53", name="TP53", node_type=NodeType.GENE),
            GraphNode(node_id="gene:BRCA1", name="BRCA1", node_type=NodeType.GENE),
            GraphNode(node_id="pathway:P1", name="PI3K", node_type=NodeType.PATHWAY),
        ]
        symbols = _extract_gene_symbols(nodes)
        assert "TP53" in symbols
        assert "BRCA1" in symbols
        assert len(symbols) == 2  # Pathway excluded

    def test_deduplication(self):
        nodes = [
            GraphNode(node_id="gene:TP53", name="TP53", node_type=NodeType.GENE),
            GraphNode(node_id="gene:TP53_2", name="tp53", node_type=NodeType.GENE),
        ]
        symbols = _extract_gene_symbols(nodes)
        assert len(symbols) == 1


class TestMergeNodes:
    """Test node merging logic."""

    def test_merge_by_id(self):
        nodes = [
            GraphNode(node_id="gene:TP53", name="TP53", node_type=NodeType.GENE,
                      source="DisGeNET", score=0.8),
            GraphNode(node_id="gene:TP53", name="TP53", node_type=NodeType.GENE,
                      source="OpenTargets", score=0.9),
        ]
        edges = []
        merged_n, merged_e = _merge_nodes(nodes, edges)
        assert len(merged_n) == 1
        assert merged_n[0].score == 0.9  # Kept highest

    def test_merge_preserves_uniprot(self):
        nodes = [
            GraphNode(node_id="gene:TP53", name="TP53", node_type=NodeType.GENE,
                      uniprot_id=None),
            GraphNode(node_id="gene:TP53", name="TP53", node_type=NodeType.GENE,
                      uniprot_id="P04637"),
        ]
        merged_n, _ = _merge_nodes(nodes, [])
        assert merged_n[0].uniprot_id == "P04637"

    def test_edge_deduplication(self):
        edges = [
            GraphEdge(source_id="gene:A", target_id="gene:B",
                      edge_type=EdgeType.INTERACTS_WITH),
            GraphEdge(source_id="gene:A", target_id="gene:B",
                      edge_type=EdgeType.INTERACTS_WITH),
        ]
        _, merged_e = _merge_nodes([], edges)
        assert len(merged_e) == 1

    def test_different_edge_types_preserved(self):
        edges = [
            GraphEdge(source_id="gene:A", target_id="gene:B",
                      edge_type=EdgeType.INTERACTS_WITH),
            GraphEdge(source_id="gene:A", target_id="gene:B",
                      edge_type=EdgeType.ACTIVATES),
        ]
        _, merged_e = _merge_nodes([], edges)
        assert len(merged_e) == 2
