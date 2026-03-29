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

            # Parse first result to get KEGG gene ID
            first_line = resp.text.strip().split("\n")[0]
            kegg_gene_id = first_line.split("\t")[0] if "\t" in first_line else ""
            if not kegg_gene_id:
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
        return _mock_kegg_pathways(gene_symbols, max_pathways)

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
    """Mock KEGG pathway data."""
    # Well-known pathways that common drug-target genes participate in
    gene_pathways: dict[str, list[tuple[str, str]]] = {
        "TP53": [("hsa04115", "p53 signalling pathway"), ("hsa05200", "Pathways in cancer")],
        "EGFR": [("hsa04012", "ErbB signalling pathway"), ("hsa04014", "Ras signalling pathway")],
        "TNF": [("hsa04668", "TNF signalling pathway"), ("hsa04064", "NF-kappa B signalling")],
        "NFKB1": [("hsa04064", "NF-kappa B signalling"), ("hsa04668", "TNF signalling pathway")],
        "PIK3CA": [("hsa04151", "PI3K-Akt signalling"), ("hsa04012", "ErbB signalling pathway")],
        "CCR5": [("hsa04062", "Chemokine signalling"), ("hsa04060", "Cytokine-cytokine receptor interaction")],
        "CXCR4": [("hsa04062", "Chemokine signalling"), ("hsa04360", "Axon guidance")],
        "CD4": [("hsa04514", "Cell adhesion molecules"), ("hsa04660", "T cell receptor signalling")],
        "INS": [("hsa04910", "Insulin signalling"), ("hsa04930", "Type II diabetes mellitus")],
        "INSR": [("hsa04910", "Insulin signalling"), ("hsa04930", "Type II diabetes mellitus")],
        "PPARG": [("hsa03320", "PPAR signalling pathway"), ("hsa04932", "Non-alcoholic fatty liver")],
        "APP": [("hsa05010", "Alzheimer disease"), ("hsa04726", "Serotonergic synapse")],
        "BACE1": [("hsa05010", "Alzheimer disease")],
        "MAPT": [("hsa05010", "Alzheimer disease"), ("hsa04024", "cAMP signalling")],
        "BRCA1": [("hsa03440", "Homologous recombination"), ("hsa04120", "Ubiquitin mediated proteolysis")],
        "ERBB2": [("hsa04012", "ErbB signalling pathway"), ("hsa05200", "Pathways in cancer")],
        "IFNG": [("hsa04630", "JAK-STAT signalling"), ("hsa04060", "Cytokine-cytokine receptor interaction")],
        "APOBEC3G": [("hsa05170", "HIV-1 infection")],
        "BST2": [("hsa05170", "HIV-1 infection")],
        "GCK": [("hsa04910", "Insulin signalling"), ("hsa04930", "Type II diabetes mellitus")],
    }

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    seen: set[str] = set()

    for gene in gene_symbols:
        pathways = gene_pathways.get(gene, [])
        for pw_id, pw_name in pathways:
            if len(seen) >= max_pathways:
                break
            node_id = f"pathway:{pw_id}"
            if pw_id not in seen:
                seen.add(pw_id)
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
