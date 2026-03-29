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

import requests

from ..models import GraphNode, GraphEdge, NodeType, EdgeType

logger = logging.getLogger(__name__)

REACTOME_BASE = "https://reactome.org/ContentService"
TIMEOUT = 20


def query_reactome_pathways(
    gene_symbols: list[str],
    max_pathways: int = 15,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Query Reactome for pathways involving the given genes.

    Parameters
    ----------
    gene_symbols : list[str]
        Gene symbols to query.
    max_pathways : int
        Maximum total pathways to return.

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
                    name = entry.get("name", "")

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
        return _mock_reactome(gene_symbols, max_pathways)

    logger.info("Reactome: found %d pathways.", len(nodes))
    return nodes, edges


def _mock_reactome(
    gene_symbols: list[str],
    max_pathways: int = 15,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Mock Reactome pathway data."""
    gene_pathways: dict[str, list[tuple[str, str]]] = {
        "TP53": [("R-HSA-3700989", "Transcriptional regulation by TP53")],
        "CCR5": [("R-HSA-380108", "Chemokine receptors bind chemokines")],
        "CD4": [("R-HSA-202424", "Downstream TCR signalling")],
        "TNF": [("R-HSA-75893", "TNF signalling"), ("R-HSA-5357956", "TNFR1 signalling")],
        "NFKB1": [("R-HSA-9020702", "Interleukin-1 family signalling")],
        "EGFR": [("R-HSA-177929", "Signalling by EGFR")],
        "INS": [("R-HSA-74752", "Signalling by Insulin receptor")],
        "APP": [("R-HSA-6900026", "Amyloid fibre formation")],
        "BRCA1": [("R-HSA-5685942", "HDR through Homologous Recombination")],
        "ERBB2": [("R-HSA-1227986", "Signalling by ERBB2")],
        "PPARG": [("R-HSA-1368082", "RORA activates gene expression")],
        "BACE1": [("R-HSA-6900026", "Amyloid fibre formation")],
    }

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    seen: set[str] = set()

    for gene in gene_symbols:
        for pw_id, pw_name in gene_pathways.get(gene, []):
            if len(seen) >= max_pathways:
                break
            node_id = f"pathway:{pw_id}"
            if pw_id not in seen:
                seen.add(pw_id)
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
