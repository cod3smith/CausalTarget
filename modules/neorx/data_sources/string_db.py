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
    *,
    allow_mocks: bool = False,
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
    allow_mocks : bool
        If *True*, fall back to curated mock data on API failure.
        If *False* (default), return empty results.

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

    _empty: tuple[list[GraphNode], list[GraphEdge]] = ([], [])

    try:
        # Use network endpoint for batch query
        proteins = "%0d".join(gene_symbols[:20])  # STRING limit
        resp = requests.get(
            f"{STRING_BASE}/json/network",
            params={
                "identifiers": proteins,
                "species": HUMAN_TAXID,
                "required_score": min_score,
                "caller_identity": "NeoRx",
            },
            timeout=TIMEOUT,
        )

        if resp.status_code != 200:
            logger.warning("STRING returned %d.", resp.status_code)
            if allow_mocks:
                return _mock_string(gene_symbols, max_interactions)
            return _empty

        interactions = resp.json()
        if not interactions:
            if allow_mocks:
                return _mock_string(gene_symbols, max_interactions)
            return _empty

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
        logger.warning("STRING request failed: %s.", e)
        if allow_mocks:
            return _mock_string(gene_symbols, max_interactions)
        return _empty

    if not edges:
        if allow_mocks:
            return _mock_string(gene_symbols, max_interactions)
        return _empty

    return nodes, edges


def _mock_string(
    gene_symbols: list[str],
    max_interactions: int = 50,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Generate generic mock PPI data for any set of genes.

    Instead of a hardcoded lookup table, creates pairwise
    interactions between the supplied genes with plausible
    confidence scores.  This ensures the mock works for
    ANY gene list, not just a curated subset.
    """
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    seen_nodes: set[str] = set()
    seen_edges: set[tuple[str, str]] = set()

    # Create pairwise interactions between supplied genes
    for i, gene_a in enumerate(gene_symbols):
        for gene_b in gene_symbols[i + 1:]:
            if len(edges) >= max_interactions:
                break

            pair = tuple(sorted([gene_a, gene_b]))
            if pair in seen_edges:
                continue
            seen_edges.add(pair)

            # Deterministic mock score based on gene names
            score = 0.5 + (hash(pair) % 40) / 100.0

            for prot in (gene_a, gene_b):
                nid = f"gene:{prot}"
                if nid not in seen_nodes:
                    seen_nodes.add(nid)
                    nodes.append(GraphNode(
                        node_id=nid, name=prot,
                        node_type=NodeType.PROTEIN, source="STRING (mock)",
                        score=score, metadata={"mock": True},
                    ))

            edges.append(GraphEdge(
                source_id=f"gene:{gene_a}", target_id=f"gene:{gene_b}",
                edge_type=EdgeType.INTERACTS_WITH, weight=score,
                source_db="STRING (mock)",
            ))

    logger.info("STRING (mock): %d interactions.", len(edges))
    return nodes, edges
