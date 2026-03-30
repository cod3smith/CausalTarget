"""
KEGG Pathway API Client
=========================

KEGG (Kyoto Encyclopedia of Genes and Genomes) maps genes to
metabolic and signalling pathways.  Pathways provide the
mechanistic context for why a gene matters in a disease —
a gene that sits in a critical signalling pathway is more
likely to be a causal driver than an isolated gene.

REST API: https://rest.kegg.jp/
Free, no API key, rate limit ~10 req/s.

Key operations:
- ``/find/disease/{query}`` — search for KEGG disease entries
- ``/link/pathway/hsa:{gene_id}`` — find pathways for a gene
- ``/get/hsa:{pathway_id}`` — get pathway details
- ``/link/hsa/{pathway_id}`` — get genes in a pathway
"""

from __future__ import annotations

import logging

import requests

from ..models import GraphNode, GraphEdge, NodeType, EdgeType

logger = logging.getLogger(__name__)

KEGG_BASE = "https://rest.kegg.jp"
TIMEOUT = 20


def query_kegg_pathways(
    gene_symbols: list[str],
    max_pathways: int = 20,
    *,
    allow_mocks: bool = False,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Find KEGG pathways for a set of genes.

    For each gene, queries KEGG for associated pathways and
    creates pathway nodes + PARTICIPATES_IN edges.

    Parameters
    ----------
    gene_symbols : list[str]
        Gene symbols to query (e.g. ["TP53", "BRCA1"]).
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
            # KEGG uses NCBI gene IDs; we try the gene symbol via /find
            resp = requests.get(
                f"{KEGG_BASE}/find/genes/{gene}+homo+sapiens",
                timeout=TIMEOUT,
            )
            if resp.status_code != 200 or not resp.text.strip():
                continue

            # Parse results to find the correct human gene ID.
            # Each result line:  hsa:1234\tSYMBOL, ALT; Description (gene)
            # We must verify the gene symbol matches our query to
            # avoid picking up unrelated genes that contain the
            # query string in their description.
            kegg_gene_id = ""
            for line in resp.text.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                candidate_id = parts[0].strip()
                # Must be a human gene (hsa:XXXXX)
                if not candidate_id.startswith("hsa:"):
                    continue
                # Extract gene symbol from description:
                # "CCR5, CD195; C-C chemokine receptor ..." → CCR5
                desc = parts[1]
                first_symbol = desc.split(",")[0].split(";")[0].strip()
                if first_symbol.upper() == gene.upper():
                    kegg_gene_id = candidate_id
                    break

            if not kegg_gene_id:
                logger.debug("KEGG: no exact match for gene '%s'.", gene)
                continue

            # Get pathways for this gene
            path_resp = requests.get(
                f"{KEGG_BASE}/link/pathway/{kegg_gene_id}",
                timeout=TIMEOUT,
            )
            if path_resp.status_code != 200 or not path_resp.text.strip():
                continue

            for line in path_resp.text.strip().split("\n"):
                parts = line.split("\t")
                if len(parts) < 2:
                    continue

                pathway_id = parts[1].strip()
                # Only human pathways (hsa)
                if not pathway_id.startswith("path:hsa"):
                    continue

                if pathway_id in seen_pathways:
                    # Just add the edge
                    edges.append(GraphEdge(
                        source_id=f"gene:{gene}",
                        target_id=f"pathway:{pathway_id}",
                        edge_type=EdgeType.PARTICIPATES_IN,
                        weight=0.8,
                        source_db="KEGG",
                    ))
                    continue

                seen_pathways.add(pathway_id)

                # Get pathway name
                pw_name = _get_pathway_name(pathway_id)

                nodes.append(GraphNode(
                    node_id=f"pathway:{pathway_id}",
                    name=pw_name or pathway_id,
                    node_type=NodeType.PATHWAY,
                    source="KEGG",
                    score=0.8,
                    metadata={"kegg_id": pathway_id},
                ))
                edges.append(GraphEdge(
                    source_id=f"gene:{gene}",
                    target_id=f"pathway:{pathway_id}",
                    edge_type=EdgeType.PARTICIPATES_IN,
                    weight=0.8,
                    source_db="KEGG",
                ))

        except requests.RequestException as e:
            logger.debug("KEGG query failed for %s: %s", gene, e)
            continue

    if not nodes:
        if allow_mocks:
            return _mock_kegg_pathways(gene_symbols, max_pathways)
        logger.info("KEGG: no pathways found for any queried gene.")
        return [], []

    logger.info("KEGG: found %d pathways for %d genes.", len(nodes), len(gene_symbols))
    return nodes, edges


def _get_pathway_name(pathway_id: str) -> str:
    """Fetch the human-readable name of a KEGG pathway."""
    try:
        clean_id = pathway_id.replace("path:", "")
        resp = requests.get(f"{KEGG_BASE}/get/{clean_id}", timeout=10)
        if resp.status_code == 200:
            for line in resp.text.split("\n"):
                if line.startswith("NAME"):
                    name = line[12:].strip()
                    # Remove " - Homo sapiens (human)" suffix
                    return name.split(" - ")[0].strip()
    except requests.RequestException:
        pass
    return ""


def _mock_kegg_pathways(
    gene_symbols: list[str],
    max_pathways: int = 20,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Generate generic mock KEGG pathway data for any gene list.

    Creates a synthetic pathway node for each supplied gene so
    downstream steps can run without API access.  No gene-specific
    pathway lookup — real pathway data comes from the live KEGG API.
    """
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    seen: set[str] = set()

    for gene in gene_symbols:
        if len(seen) >= max_pathways:
            break
        # Deterministic mock pathway ID from gene name
        pw_hash = abs(hash(gene)) % 99999
        pw_id = f"hsa{pw_hash:05d}"
        pw_name = f"{gene}-associated pathway"

        if pw_id in seen:
            continue
        seen.add(pw_id)

        node_id = f"pathway:{pw_id}"
        nodes.append(GraphNode(
            node_id=node_id,
            name=pw_name,
            node_type=NodeType.PATHWAY,
            source="KEGG (mock)",
            score=0.8,
            metadata={"kegg_id": pw_id, "mock": True},
        ))
        edges.append(GraphEdge(
            source_id=f"gene:{gene}",
            target_id=node_id,
            edge_type=EdgeType.PARTICIPATES_IN,
            weight=0.8,
            source_db="KEGG (mock)",
        ))

    logger.info("KEGG (mock): %d pathways for %d genes.", len(nodes), len(gene_symbols))
    return nodes, edges
