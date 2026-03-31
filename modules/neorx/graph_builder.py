"""
Disease Causal Graph Builder
=============================

The graph builder is the first major step in the NeoRx
pipeline.  It queries all 8 biomedical data sources and assembles
a unified **causal knowledge graph** for a specific disease.

Architecture
------------
1. Query gene–disease associations (Monarch Initiative, Open Targets)
1b. Query validated drug targets (ChEMBL — human + pathogen)
2. Query pathway memberships (KEGG, Reactome)
3. Query protein–protein interactions (STRING)
4. Enrich proteins with UniProt metadata (function, PDB IDs)
5. Query PDB structures for docking targets
6. Merge duplicate nodes by gene symbol
7. Build a NetworkX DiGraph ready for DoWhy causal inference

Node merging
------------
Monarch might call it "CCR5" while Open Targets uses
"ENSG00000160791".  We normalise to gene symbols and merge,
keeping the highest confidence score and union of all metadata.

Edge semantics
--------------
- ``ASSOCIATED_WITH`` → gene ↔ disease (from Monarch, OT)
- ``PARTICIPATES_IN`` → gene → pathway (from KEGG, Reactome)
- ``INTERACTS_WITH`` → protein ↔ protein (from STRING)
- ``CAUSES`` → the disease node → disease outcome

The causal identifier (next stage) then tests whether each
gene's path to the disease outcome is *causal* or merely
*correlational* using do-calculus.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
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
    query_monarch,
    query_open_targets,
    query_kegg_pathways,
    query_reactome_pathways,
    query_string_interactions,
    query_uniprot,
    query_pdb_structures,
    query_chembl,
)
from .cache import get_cache, _cache_key, GRAPH_TTL

logger = logging.getLogger(__name__)


def build_disease_graph(
    disease: str,
    *,
    max_genes: int = 20,
    string_min_score: int = 400,
    use_cache: bool = True,
    allow_mocks: bool = False,
) -> DiseaseGraph:
    """Build a comprehensive causal knowledge graph for a disease.

    Parameters
    ----------
    disease : str
        Disease name (e.g. "HIV", "Type 2 Diabetes").
    max_genes : int
        Cap on number of genes to include.
    string_min_score : int
        STRING combined score threshold (0–1000).
    use_cache : bool
        Check/store results in the cache layer.
    allow_mocks : bool
        If *True*, data source clients may fall back to curated
        mock data when a live API call fails.  If *False*
        (default), failed API calls produce empty results.

    Returns
    -------
    DiseaseGraph
        Assembled graph with merged nodes and unified edges.
    """
    # ── Check cache first ─────────────────────────────────────
    if use_cache:
        cache = get_cache()
        cache_key = _cache_key(
            "graph", disease=disease.lower(),
            max_genes=max_genes, string_min_score=string_min_score,
        )
        cached = cache.get(cache_key)
        if cached is not None:
            try:
                graph = DiseaseGraph.model_validate(cached)
                logger.info("Graph loaded from cache (%d nodes, %d edges).",
                            len(graph.nodes), len(graph.edges))
                return graph
            except Exception:
                pass  # Invalid cache entry — rebuild

    all_nodes: list[GraphNode] = []
    all_edges: list[GraphEdge] = []
    sources_queried: list[str] = []
    disease_id: str | None = None

    # ── Step 1: Gene–Disease Associations (parallel) ──────────

    logger.info("Querying Monarch + Open Targets for '%s' (parallel)…", disease)
    with ThreadPoolExecutor(max_workers=2) as pool:
        mn_future = pool.submit(
            query_monarch, disease,
            max_results=max_genes,
            allow_mocks=allow_mocks,
        )
        ot_future = pool.submit(
            query_open_targets, disease, max_results=max_genes,
            allow_mocks=allow_mocks,
        )
        mn_nodes, mn_edges = mn_future.result()
        ot_nodes, ot_edges = ot_future.result()

    all_nodes.extend(mn_nodes)
    all_edges.extend(mn_edges)
    sources_queried.append("Monarch")
    logger.info("  Monarch: %d nodes, %d edges.", len(mn_nodes), len(mn_edges))

    all_nodes.extend(ot_nodes)
    all_edges.extend(ot_edges)
    sources_queried.append("OpenTargets")
    logger.info("  Open Targets: %d nodes, %d edges.", len(ot_nodes), len(ot_edges))

    # ── Step 1b: ChEMBL Drug Targets (local SQLite) ────────────

    logger.info("Querying ChEMBL for '%s'…", disease)
    chembl_nodes, chembl_edges = query_chembl(
        disease, max_results=max_genes * 3,
        allow_mocks=allow_mocks,
    )
    all_nodes.extend(chembl_nodes)
    all_edges.extend(chembl_edges)
    if chembl_nodes:
        sources_queried.append("ChEMBL")
    n_pathogen = sum(
        1 for n in chembl_nodes
        if n.metadata.get("is_pathogen_target")
    )
    logger.info(
        "  ChEMBL: %d nodes (%d pathogen), %d edges.",
        len(chembl_nodes), n_pathogen, len(chembl_edges),
    )

    # Resolve disease ontology ID from Open Targets
    try:
        from .data_sources.open_targets import resolve_disease_id
        disease_id = resolve_disease_id(disease)
    except Exception:
        pass

    # ── Step 2: Pathway Memberships ─────────────────────────────

    gene_symbols = _extract_gene_symbols(all_nodes, max_genes=max_genes)
    logger.info("Found %d unique gene symbols (capped at %d).",
                len(gene_symbols), max_genes)

    logger.info("Querying KEGG pathways…")
    kegg_nodes, kegg_edges = query_kegg_pathways(gene_symbols, allow_mocks=allow_mocks)
    all_nodes.extend(kegg_nodes)
    all_edges.extend(kegg_edges)
    sources_queried.append("KEGG")
    logger.info("  KEGG: %d nodes, %d edges.", len(kegg_nodes), len(kegg_edges))

    logger.info("Querying Reactome pathways…")
    react_nodes, react_edges = query_reactome_pathways(gene_symbols, allow_mocks=allow_mocks)
    all_nodes.extend(react_nodes)
    all_edges.extend(react_edges)
    sources_queried.append("Reactome")
    logger.info("  Reactome: %d nodes, %d edges.", len(react_nodes), len(react_edges))

    # ── Step 3: Protein–Protein Interactions ────────────────────

    logger.info("Querying STRING interactions…")
    string_nodes, string_edges = query_string_interactions(
        gene_symbols, min_score=string_min_score,
        allow_mocks=allow_mocks,
    )
    all_nodes.extend(string_nodes)
    all_edges.extend(string_edges)
    sources_queried.append("STRING")
    logger.info("  STRING: %d nodes, %d edges.", len(string_nodes), len(string_edges))

    # ── Step 4: UniProt Enrichment ──────────────────────────────

    logger.info("Enriching with UniProt metadata…")
    uniprot_data = query_uniprot(gene_symbols, allow_mocks=allow_mocks)
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
        pdb_data = query_pdb_structures(uniprot_map, allow_mocks=allow_mocks)
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
        source="NeoRx",
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
        if node.node_type in (NodeType.GENE, NodeType.PROTEIN, NodeType.PATHOGEN_GENE):
            pair = (node.node_id, disease_node.node_id)
            if pair not in existing_disease_edges:
                merged_edges.append(GraphEdge(
                    source_id=node.node_id,
                    target_id=disease_node.node_id,
                    edge_type=EdgeType.ASSOCIATED_WITH,
                    weight=node.score,
                    source_db="NeoRx",
                ))

    graph = DiseaseGraph(
        disease_name=disease,
        disease_id=disease_id,
        nodes=merged_nodes,
        edges=merged_edges,
        sources_queried=sources_queried,
    )

    logger.info(
        "Graph built: %d nodes (%d genes, %d proteins, %d pathways), %d edges.",
        len(graph.nodes), graph.n_genes, graph.n_proteins,
        graph.n_pathways, len(graph.edges),
    )

    # ── Cache & persist ───────────────────────────────────────
    if use_cache:
        try:
            cache.set(cache_key, graph.model_dump(mode="json"), ttl=GRAPH_TTL)
        except Exception:
            pass

    try:
        from .persistence import save_graph_to_db
        save_graph_to_db(graph, params={"max_genes": max_genes})
    except Exception:
        pass

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
            source=node.source or "",
            metadata=dict(node.metadata),
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

    When the same gene appears from multiple sources (Monarch
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
