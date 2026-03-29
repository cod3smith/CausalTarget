"""
STRING Protein–Protein Interaction Client
==========================================

STRING (https://string-db.org/) is the most comprehensive
database of protein–protein interactions (PPIs).  It integrates:
- Experimental evidence (co-immunoprecipitation, yeast two-hybrid)
- Computational prediction (gene neighbourhood, gene fusion)
- Text-mining of scientific literature
- Co-expression across conditions

Each interaction has a **combined confidence score** (0–1000):
- > 900: highest confidence
- > 700: high confidence
- > 400: medium confidence

For causal graph construction, PPIs reveal:
1. Which proteins form complexes with our candidate targets
2. Which proteins can propagate a drug's effect through the network
3. Whether a target is a hub (many interactions) or peripheral

A target protein with many high-confidence interactions is more
likely to be a key driver of the disease mechanism.

API: https://string-db.org/api/
Free, no key required, species ID for human = 9606.
"""

from __future__ import annotations

import logging

import requests

from ..models import GraphNode, GraphEdge, NodeType, EdgeType

logger = logging.getLogger(__name__)

STRING_BASE = "https://string-db.org/api"
HUMAN_TAXID = 9606
TIMEOUT = 30


def query_string_interactions(
    gene_symbols: list[str],
    min_score: int = 400,
    max_interactions: int = 50,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Query STRING for protein–protein interactions.

    Parameters
    ----------
    gene_symbols : list[str]
        Gene/protein symbols to query.
    min_score : int
        Minimum combined score (0–1000).
    max_interactions : int
        Maximum total interactions to return.

    Returns
    -------
    tuple[list[GraphNode], list[GraphEdge]]
        Interaction nodes and edges.
    """
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    seen_nodes: set[str] = set()

    if not gene_symbols:
        return nodes, edges

    try:
        # Use network endpoint for batch query
        proteins = "%0d".join(gene_symbols[:20])  # STRING limit
        resp = requests.get(
            f"{STRING_BASE}/json/network",
            params={
                "identifiers": proteins,
                "species": HUMAN_TAXID,
                "required_score": min_score,
                "caller_identity": "CausalTarget",
            },
            timeout=TIMEOUT,
        )

        if resp.status_code != 200:
            logger.warning("STRING returned %d. Using mock.", resp.status_code)
            return _mock_string(gene_symbols, max_interactions)

        interactions = resp.json()
        if not interactions:
            return _mock_string(gene_symbols, max_interactions)

        for ix in interactions[:max_interactions]:
            prot_a = ix.get("preferredName_A", ix.get("stringId_A", ""))
            prot_b = ix.get("preferredName_B", ix.get("stringId_B", ""))
            score = ix.get("score", 0.0)

            if not prot_a or not prot_b:
                continue

            # Add nodes for any new proteins
            for prot in (prot_a, prot_b):
                node_id = f"gene:{prot}"
                if node_id not in seen_nodes:
                    seen_nodes.add(node_id)
                    nodes.append(GraphNode(
                        node_id=node_id,
                        name=prot,
                        node_type=NodeType.PROTEIN,
                        source="STRING",
                        score=score,
                    ))

            edges.append(GraphEdge(
                source_id=f"gene:{prot_a}",
                target_id=f"gene:{prot_b}",
                edge_type=EdgeType.INTERACTS_WITH,
                weight=min(score, 1.0),
                source_db="STRING",
                evidence=f"STRING combined score: {score:.3f}",
            ))

        logger.info("STRING: %d interactions among %d proteins.", len(edges), len(seen_nodes))

    except requests.RequestException as e:
        logger.warning("STRING request failed: %s. Using mock.", e)
        return _mock_string(gene_symbols, max_interactions)

    if not edges:
        return _mock_string(gene_symbols, max_interactions)

    return nodes, edges


def _mock_string(
    gene_symbols: list[str],
    max_interactions: int = 50,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Mock STRING PPI data based on well-known interactions."""
    known_ppis: dict[str, list[tuple[str, float]]] = {
        "CCR5": [("CD4", 0.7), ("CXCR4", 0.6)],
        "CD4": [("CCR5", 0.7), ("CXCR4", 0.65)],
        "CXCR4": [("CD4", 0.65), ("CCR5", 0.6)],
        "TNF": [("NFKB1", 0.9), ("IFNG", 0.5)],
        "NFKB1": [("TNF", 0.9), ("IFNG", 0.6)],
        "TP53": [("BRCA1", 0.85), ("MDM2", 0.95)],
        "BRCA1": [("TP53", 0.85), ("BRCA2", 0.9)],
        "EGFR": [("ERBB2", 0.9), ("PIK3CA", 0.8)],
        "ERBB2": [("EGFR", 0.9), ("PIK3CA", 0.75)],
        "INS": [("INSR", 0.95), ("IRS1", 0.85)],
        "INSR": [("INS", 0.95), ("IRS1", 0.9), ("AKT2", 0.7)],
        "PPARG": [("RXRA", 0.85)],
        "APP": [("BACE1", 0.9), ("PSEN1", 0.85)],
        "BACE1": [("APP", 0.9), ("PSEN1", 0.8)],
        "PSEN1": [("APP", 0.85), ("BACE1", 0.8)],
    }

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    seen_nodes: set[str] = set()
    seen_edges: set[tuple[str, str]] = set()

    for gene in gene_symbols:
        partners = known_ppis.get(gene, [])
        for partner, score in partners:
            if len(edges) >= max_interactions:
                break

            pair = tuple(sorted([gene, partner]))
            if pair in seen_edges:
                continue
            seen_edges.add(pair)

            for prot in (gene, partner):
                nid = f"gene:{prot}"
                if nid not in seen_nodes:
                    seen_nodes.add(nid)
                    nodes.append(GraphNode(
                        node_id=nid, name=prot,
                        node_type=NodeType.PROTEIN, source="STRING (mock)",
                        score=score, metadata={"mock": True},
                    ))

            edges.append(GraphEdge(
                source_id=f"gene:{gene}", target_id=f"gene:{partner}",
                edge_type=EdgeType.INTERACTS_WITH, weight=score,
                source_db="STRING (mock)",
            ))

    logger.info("STRING (mock): %d interactions.", len(edges))
    return nodes, edges
