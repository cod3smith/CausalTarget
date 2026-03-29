"""
Disease Causal Graph Builder
=============================

The graph builder is the first major step in the CausalTarget
pipeline.  It queries all 7 biomedical data sources and assembles
a unified **causal knowledge graph** for a specific disease.

Architecture
------------
1. Query gene–disease associations (DisGeNET, Open Targets)
2. Query pathway memberships (KEGG, Reactome)
3. Query protein–protein interactions (STRING)
4. Enrich proteins with UniProt metadata (function, PDB IDs)
5. Query PDB structures for docking targets
6. Merge duplicate nodes by gene symbol
7. Build a NetworkX DiGraph ready for DoWhy causal inference

Node merging
------------
DisGeNET might call it "CCR5" while Open Targets uses
"ENSG00000160791".  We normalise to gene symbols and merge,
keeping the highest confidence score and union of all metadata.

Edge semantics
--------------
- ``ASSOCIATED_WITH`` → gene ↔ disease (from DisGeNET, OT)
- ``PARTICIPATES_IN`` → gene → pathway (from KEGG, Reactome)
- ``INTERACTS_WITH`` → protein ↔ protein (from STRING)
- ``CAUSES`` → the disease node → disease outcome

The causal identifier (next stage) then tests whether each
gene's path to the disease outcome is *causal* or merely
*correlational* using do-calculus.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx

from .models import (
    DiseaseGraph,
    GraphNode,
    GraphEdge,
    NodeType,
    EdgeType,
)
from .data_sources import (
    query_disgenet,
    query_open_targets,
    query_kegg_pathways,
    query_reactome_pathways,
    query_string_interactions,
    query_uniprot,
    query_pdb_structures,
)

logger = logging.getLogger(__name__)


def build_disease_graph(
    disease: str,
    *,
    disgenet_api_key: str | None = None,
    min_disgenet_score: float = 0.1,
    max_genes: int = 20,
    string_min_score: int = 400,
) -> DiseaseGraph:
    """Build a comprehensive causal knowledge graph for a disease.

    Parameters
    ----------
    disease : str
        Disease name (e.g. "HIV", "Type 2 Diabetes").
    disgenet_api_key : str, optional
        DisGeNET API key for live queries.
    min_disgenet_score : float
        Minimum gene–disease association score.
    max_genes : int
        Cap on number of genes to include.
    string_min_score : int
        STRING combined score threshold (0–1000).

    Returns
    -------
    DiseaseGraph
        Assembled graph with merged nodes and unified edges.
    """
    all_nodes: list[GraphNode] = []
    all_edges: list[GraphEdge] = []
    sources_queried: list[str] = []

    # ── Step 1: Gene–Disease Associations ───────────────────────

    logger.info("Querying DisGeNET for '%s'…", disease)
    dg_nodes, dg_edges = query_disgenet(
        disease, min_score=min_disgenet_score,
        max_results=max_genes, api_key=disgenet_api_key,
    )
    all_nodes.extend(dg_nodes)
    all_edges.extend(dg_edges)
    sources_queried.append("DisGeNET")
    logger.info("  DisGeNET: %d nodes, %d edges.", len(dg_nodes), len(dg_edges))

    logger.info("Querying Open Targets for '%s'…", disease)
    ot_nodes, ot_edges = query_open_targets(disease, max_results=max_genes)
    all_nodes.extend(ot_nodes)
    all_edges.extend(ot_edges)
    sources_queried.append("OpenTargets")
    logger.info("  Open Targets: %d nodes, %d edges.", len(ot_nodes), len(ot_edges))

    # ── Step 2: Pathway Memberships ─────────────────────────────

    gene_symbols = _extract_gene_symbols(all_nodes, max_genes=max_genes)
    logger.info("Found %d unique gene symbols (capped at %d).",
                len(gene_symbols), max_genes)

    logger.info("Querying KEGG pathways…")
    kegg_nodes, kegg_edges = query_kegg_pathways(gene_symbols)
    all_nodes.extend(kegg_nodes)
    all_edges.extend(kegg_edges)
    sources_queried.append("KEGG")
    logger.info("  KEGG: %d nodes, %d edges.", len(kegg_nodes), len(kegg_edges))

    logger.info("Querying Reactome pathways…")
    react_nodes, react_edges = query_reactome_pathways(gene_symbols)
    all_nodes.extend(react_nodes)
    all_edges.extend(react_edges)
    sources_queried.append("Reactome")
    logger.info("  Reactome: %d nodes, %d edges.", len(react_nodes), len(react_edges))

    # ── Step 3: Protein–Protein Interactions ────────────────────

    logger.info("Querying STRING interactions…")
    string_nodes, string_edges = query_string_interactions(
        gene_symbols, min_score=string_min_score,
    )
    all_nodes.extend(string_nodes)
    all_edges.extend(string_edges)
    sources_queried.append("STRING")
    logger.info("  STRING: %d nodes, %d edges.", len(string_nodes), len(string_edges))

    # ── Step 4: UniProt Enrichment ──────────────────────────────

    logger.info("Enriching with UniProt metadata…")
    uniprot_data = query_uniprot(gene_symbols)
    _enrich_nodes_with_uniprot(all_nodes, uniprot_data)
    sources_queried.append("UniProt")
    logger.info("  UniProt: enriched %d/%d proteins.", len(uniprot_data), len(gene_symbols))

    # ── Step 5: PDB Structures ──────────────────────────────────

    uniprot_map = {
        gene: info["uniprot_id"]
        for gene, info in uniprot_data.items()
        if info.get("uniprot_id")
    }
    if uniprot_map:
        logger.info("Querying PDB structures for %d proteins…", len(uniprot_map))
        pdb_data = query_pdb_structures(uniprot_map)
        _enrich_nodes_with_pdb(all_nodes, pdb_data)
        sources_queried.append("PDB")
        logger.info("  PDB: structures for %d proteins.", len(pdb_data))

    # ── Step 6: Merge & Build ───────────────────────────────────

    merged_nodes, merged_edges = _merge_nodes(all_nodes, all_edges)

    # Add disease outcome node
    disease_node = GraphNode(
        node_id=f"disease:{disease.lower().replace(' ', '_')}",
        name=disease,
        node_type=NodeType.DISEASE,
        source="CausalTarget",
        score=1.0,
    )
    merged_nodes.append(disease_node)

    # Connect genes directly to disease via ASSOCIATED_WITH if
    # they don't already have that edge
    existing_disease_edges = {
        (e.source_id, e.target_id) for e in merged_edges
        if e.edge_type == EdgeType.ASSOCIATED_WITH
    }
    for node in merged_nodes:
        if node.node_type in (NodeType.GENE, NodeType.PROTEIN):
            pair = (node.node_id, disease_node.node_id)
            if pair not in existing_disease_edges:
                merged_edges.append(GraphEdge(
                    source_id=node.node_id,
                    target_id=disease_node.node_id,
                    edge_type=EdgeType.ASSOCIATED_WITH,
                    weight=node.score,
                    source_db="CausalTarget",
                ))

    graph = DiseaseGraph(
        disease_name=disease,
        nodes=merged_nodes,
        edges=merged_edges,
        sources_queried=sources_queried,
    )

    logger.info(
        "Graph built: %d nodes (%d genes, %d proteins, %d pathways), %d edges.",
        len(graph.nodes), graph.n_genes, graph.n_proteins,
        graph.n_pathways, len(graph.edges),
    )

    return graph


def disease_graph_to_networkx(graph: DiseaseGraph) -> nx.DiGraph:
    """Convert a DiseaseGraph to a NetworkX directed graph.

    This is needed by DoWhy for causal inference.  Node attributes
    include ``node_type``, ``score``, ``uniprot_id``, etc.  Edge
    attributes include ``edge_type``, ``weight``, ``source_db``.
    """
    G = nx.DiGraph()

    for node in graph.nodes:
        G.add_node(
            node.node_id,
            name=node.name,
            node_type=node.node_type.value,
            score=node.score,
            uniprot_id=node.uniprot_id or "",
            pdb_ids=node.pdb_ids,
            description=node.description or "",
        )

    for edge in graph.edges:
        G.add_edge(
            edge.source_id,
            edge.target_id,
            edge_type=edge.edge_type.value,
            weight=edge.weight,
            source_db=edge.source_db,
            evidence=edge.evidence or "",
        )

    return G


# ── Internal Helpers ────────────────────────────────────────────────

def _extract_gene_symbols(
    nodes: list[GraphNode],
    max_genes: int = 0,
) -> list[str]:
    """Extract unique gene symbols from node names.

    Parameters
    ----------
    nodes : list[GraphNode]
        All nodes collected so far.
    max_genes : int
        If > 0, keep only the top-scoring genes (by their node
        score) to avoid sending hundreds of genes to per-gene APIs.
    """
    # Collect unique genes, remembering the best score for each
    best_score: dict[str, float] = {}
    for node in nodes:
        if node.node_type in (NodeType.GENE, NodeType.PROTEIN):
            sym = node.name.upper()
            if sym not in best_score or node.score > best_score[sym]:
                best_score[sym] = node.score

    # Sort by score descending, then cap
    ranked = sorted(best_score.items(), key=lambda kv: kv[1], reverse=True)
    if max_genes > 0:
        ranked = ranked[:max_genes]
    return [sym for sym, _ in ranked]


def _enrich_nodes_with_uniprot(
    nodes: list[GraphNode],
    uniprot_data: dict[str, dict[str, Any]],
) -> None:
    """In-place enrichment of nodes with UniProt metadata."""
    for node in nodes:
        gene = node.name.upper()
        info = uniprot_data.get(gene)
        if not info:
            continue

        if not node.uniprot_id and info.get("uniprot_id"):
            node.uniprot_id = info["uniprot_id"]
        if not node.pdb_ids and info.get("pdb_ids"):
            node.pdb_ids = info["pdb_ids"]
        if not node.description and info.get("function"):
            node.description = info["function"]

        # Store druggability in metadata
        node.metadata["is_druggable"] = info.get("is_druggable", False)
        node.metadata["subcellular_location"] = info.get("subcellular_location", "")
        node.metadata["go_terms"] = info.get("go_terms", [])


def _enrich_nodes_with_pdb(
    nodes: list[GraphNode],
    pdb_data: dict[str, list[dict[str, Any]]],
) -> None:
    """In-place enrichment of nodes with PDB structure IDs."""
    for node in nodes:
        gene = node.name.upper()
        structs = pdb_data.get(gene)
        if not structs:
            continue
        # Prefer structures with ligands (defines binding pocket)
        sorted_structs = sorted(structs, key=lambda s: (s.get("has_ligand", False), -(s.get("resolution") or 99)))
        pdb_ids = [s["pdb_id"] for s in sorted_structs]
        if not node.pdb_ids:
            node.pdb_ids = pdb_ids
        else:
            # Union
            existing = set(node.pdb_ids)
            for pid in pdb_ids:
                if pid not in existing:
                    node.pdb_ids.append(pid)


def _merge_nodes(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Merge duplicate nodes by gene/pathway name.

    When the same gene appears from multiple sources (DisGeNET
    and Open Targets both report CCR5), we keep the entry with
    the highest score and merge metadata.
    """
    merged: dict[str, GraphNode] = {}

    for node in nodes:
        key = node.node_id
        if key in merged:
            existing = merged[key]
            # Keep highest score
            if node.score > existing.score:
                existing.score = node.score
            # Merge UniProt
            if node.uniprot_id and not existing.uniprot_id:
                existing.uniprot_id = node.uniprot_id
            # Merge PDB IDs
            existing_pdb = set(existing.pdb_ids)
            for pid in node.pdb_ids:
                if pid not in existing_pdb:
                    existing.pdb_ids.append(pid)
            # Merge metadata
            existing.metadata.update(node.metadata)
            # Record multiple sources
            if node.source and node.source not in existing.source:
                existing.source = f"{existing.source}, {node.source}"
        else:
            merged[key] = node.model_copy()

    # Deduplicate edges
    seen_edges: set[tuple[str, str, str]] = set()
    unique_edges: list[GraphEdge] = []
    for edge in edges:
        key = (edge.source_id, edge.target_id, edge.edge_type.value)
        if key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(edge)

    return list(merged.values()), unique_edges
