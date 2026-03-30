"""
Reactome Pathway API Client
==============================

Reactome (https://reactome.org/) is a curated database of
biological pathways and reactions.  Unlike KEGG, Reactome
pathways are deeply hierarchical — a pathway can contain
sub-pathways, which contain reactions, which involve specific
molecular events.

Content Service: https://reactome.org/ContentService/
Analysis Service: https://reactome.org/AnalysisService/
Free, no API key required.

What we extract:
- Pathways that contain our genes of interest
- Pathway hierarchy (which pathways are part of larger pathways)
- Reaction details (what molecular events occur)
"""

from __future__ import annotations

import logging
import re

import requests

from ..models import GraphNode, GraphEdge, NodeType, EdgeType

logger = logging.getLogger(__name__)

REACTOME_BASE = "https://reactome.org/ContentService"
TIMEOUT = 20


def query_reactome_pathways(
    gene_symbols: list[str],
    max_pathways: int = 15,
    *,
    allow_mocks: bool = False,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Query Reactome for pathways involving the given genes.

    Parameters
    ----------
    gene_symbols : list[str]
        Gene symbols to query.
    max_pathways : int
        Maximum total pathways to return.
    allow_mocks : bool
        If *True*, fall back to curated mock data when the API
        yields no results.  *False* by default.

    Returns
    -------
    tuple[list[GraphNode], list[GraphEdge]]
        Pathway nodes and gene-pathway edges.
    """
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    seen_pathways: set[str] = set()

    for gene in gene_symbols:
        if len(seen_pathways) >= max_pathways:
            break

        try:
            # Search for entities matching the gene symbol in Homo sapiens
            resp = requests.get(
                f"{REACTOME_BASE}/search/query",
                params={
                    "query": gene,
                    "species": "Homo sapiens",
                    "types": "Pathway",
                    "cluster": "true",
                },
                headers={"accept": "application/json"},
                timeout=TIMEOUT,
            )

            if resp.status_code != 200:
                continue

            data = resp.json()
            results = data.get("results", [])

            for group in results:
                entries = group.get("entries", [])
                for entry in entries[:3]:  # top 3 pathways per gene
                    if len(seen_pathways) >= max_pathways:
                        break

                    st_id = entry.get("stId", "")
                    name = re.sub(r"<[^>]+>", "", entry.get("name", ""))

                    if not st_id or st_id in seen_pathways:
                        # Just add edge if pathway already exists
                        if st_id in seen_pathways:
                            edges.append(GraphEdge(
                                source_id=f"gene:{gene}",
                                target_id=f"pathway:{st_id}",
                                edge_type=EdgeType.PARTICIPATES_IN,
                                weight=0.75,
                                source_db="Reactome",
                            ))
                        continue

                    seen_pathways.add(st_id)

                    nodes.append(GraphNode(
                        node_id=f"pathway:{st_id}",
                        name=name,
                        node_type=NodeType.PATHWAY,
                        source="Reactome",
                        score=0.75,
                        metadata={"reactome_id": st_id},
                    ))
                    edges.append(GraphEdge(
                        source_id=f"gene:{gene}",
                        target_id=f"pathway:{st_id}",
                        edge_type=EdgeType.PARTICIPATES_IN,
                        weight=0.75,
                        source_db="Reactome",
                    ))

        except requests.RequestException as e:
            logger.debug("Reactome query failed for %s: %s", gene, e)
            continue

    if not nodes:
        if allow_mocks:
            return _mock_reactome(gene_symbols, max_pathways)
        logger.info("Reactome: no pathways found for any queried gene.")
        return [], []

    logger.info("Reactome: found %d pathways.", len(nodes))
    return nodes, edges


def _mock_reactome(
    gene_symbols: list[str],
    max_pathways: int = 15,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Generate generic mock Reactome pathway data for any gene list.

    Creates a synthetic pathway node for each supplied gene so
    downstream pipeline steps (pathway enrichment, causal graph
    weighting) can run without API access.  No gene-specific
    pathway lookup — real pathway data comes from the live API.
    """
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    seen: set[str] = set()

    for gene in gene_symbols:
        if len(seen) >= max_pathways:
            break
        # Deterministic mock pathway ID from gene name
        pw_hash = abs(hash(gene)) % 9_999_999
        pw_id = f"R-HSA-{pw_hash:07d}"
        pw_name = f"Signalling by {gene}"

        if pw_id in seen:
            continue
        seen.add(pw_id)

        node_id = f"pathway:{pw_id}"
        nodes.append(GraphNode(
            node_id=node_id, name=pw_name,
            node_type=NodeType.PATHWAY, source="Reactome (mock)",
            score=0.75, metadata={"reactome_id": pw_id, "mock": True},
        ))
        edges.append(GraphEdge(
            source_id=f"gene:{gene}", target_id=node_id,
            edge_type=EdgeType.PARTICIPATES_IN, weight=0.75,
            source_db="Reactome (mock)",
        ))

    logger.info("Reactome (mock): %d pathways.", len(nodes))
    return nodes, edges
