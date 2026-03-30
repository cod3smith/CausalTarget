"""
Monarch Initiative API Client
===============================

The Monarch Initiative (https://monarchinitiative.org/) is a fully
open-source, open-access platform that integrates gene-disease
associations from OMIM, ClinGen, Orphanet, and other curated sources.

Advantages over DisGeNET:
- **No API key required** — completely open access
- **CC BY 4.0 license** — free for any use
- Distinguishes **causal** vs **correlated** gene-disease associations
- Weekly updates from OMIM, ClinGen, Orphanet, HPO

API: https://api-v3.monarchinitiative.org/v3/api

What we extract:
- Gene symbols (from HGNC identifiers)
- Association category (causal vs correlated)
- Source provenance (OMIM, ClinGen, Orphanet)
- Disease ontology links (MONDO IDs)
"""

from __future__ import annotations

import logging
from typing import Optional

import requests

from ..models import GraphNode, GraphEdge, NodeType, EdgeType

logger = logging.getLogger(__name__)

MONARCH_BASE = "https://api-v3.monarchinitiative.org/v3/api"
TIMEOUT = 30


def query_monarch(
    disease_name: str,
    max_results: int = 50,
    *,
    allow_mocks: bool = False,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Query Monarch Initiative for gene-disease associations.

    Parameters
    ----------
    disease_name : str
        Disease name to search for (e.g. "lung cancer", "HIV").
    max_results : int
        Maximum number of gene associations to return.
    allow_mocks : bool
        If *True*, fall back to curated mock data on API failure.
        If *False* (default), return empty results — no fake data.

    Returns
    -------
    tuple[list[GraphNode], list[GraphEdge]]
        Discovered nodes and edges.
    """
    _empty: tuple[list[GraphNode], list[GraphEdge]] = ([], [])
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    try:
        # Step 1: Resolve disease name → MONDO ID
        mondo_id = _resolve_disease_id(disease_name)
        if not mondo_id:
            logger.warning(
                "Monarch: could not resolve '%s' to a MONDO ID.",
                disease_name,
            )
            if allow_mocks:
                return _mock_monarch(disease_name, max_results)
            return _empty

        disease_node_id = f"disease:{disease_name.lower().replace(' ', '_')}"

        # Step 2: Query causal gene-disease associations
        causal_genes = _query_associations(
            mondo_id,
            category="biolink:CausalGeneToDiseaseAssociation",
            limit=max_results,
        )

        # Step 3: Query correlated gene-disease associations
        correlated_genes = _query_associations(
            mondo_id,
            category="biolink:CorrelatedGeneToDiseaseAssociation",
            limit=max_results,
        )

        # Step 4: If both are empty, try the broader catch-all category.
        # Monarch's curated data (OMIM, ClinGen, Orphanet) is richest for
        # Mendelian / genetic diseases and may have zero entries for
        # infectious diseases like HIV or typhoid.
        if not causal_genes and not correlated_genes:
            broad_genes = _query_associations(
                mondo_id,
                category="biolink:GeneToDiseaseAssociation",
                limit=max_results,
            )
            if broad_genes:
                correlated_genes = broad_genes
            else:
                logger.info(
                    "Monarch: 0 gene-disease associations for '%s' (%s). "
                    "This is expected for infectious diseases — Monarch's "
                    "data is strongest for Mendelian/genetic conditions. "
                    "Other sources (Open Targets, STRING) will compensate.",
                    disease_name, mondo_id,
                )

        # Merge: causal genes get higher base score
        seen: set[str] = set()

        for assoc in causal_genes:
            gene_symbol = assoc.get("subject_label", "")
            if not gene_symbol or gene_symbol in seen:
                continue
            seen.add(gene_symbol)

            source = assoc.get("provided_by", "Monarch")
            # Causal associations get a high base score
            score = 0.85

            node_id = f"gene:{gene_symbol}"
            nodes.append(GraphNode(
                node_id=node_id,
                name=gene_symbol,
                node_type=NodeType.GENE,
                source="Monarch",
                score=score,
                metadata={
                    "mondo_id": mondo_id,
                    "hgnc_id": assoc.get("subject", ""),
                    "association_type": "causal",
                    "provenance": source,
                },
            ))
            edges.append(GraphEdge(
                source_id=node_id,
                target_id=disease_node_id,
                edge_type=EdgeType.ASSOCIATED_WITH,
                weight=score,
                source_db="Monarch",
                evidence=f"Causal association (source: {source})",
            ))

        for assoc in correlated_genes:
            gene_symbol = assoc.get("subject_label", "")
            if not gene_symbol or gene_symbol in seen:
                continue
            seen.add(gene_symbol)

            source = assoc.get("provided_by", "Monarch")
            # Correlated associations get a moderate score
            score = 0.60

            node_id = f"gene:{gene_symbol}"
            nodes.append(GraphNode(
                node_id=node_id,
                name=gene_symbol,
                node_type=NodeType.GENE,
                source="Monarch",
                score=score,
                metadata={
                    "mondo_id": mondo_id,
                    "hgnc_id": assoc.get("subject", ""),
                    "association_type": "correlated",
                    "provenance": source,
                },
            ))
            edges.append(GraphEdge(
                source_id=node_id,
                target_id=disease_node_id,
                edge_type=EdgeType.ASSOCIATED_WITH,
                weight=score,
                source_db="Monarch",
                evidence=f"Correlated association (source: {source})",
            ))

        # Cap at max_results
        if len(nodes) > max_results:
            nodes = nodes[:max_results]
            edges = edges[:max_results]

        logger.info(
            "Monarch: found %d genes for '%s' (%s) — %d causal, %d correlated.",
            len(nodes), disease_name, mondo_id,
            len(causal_genes), len(correlated_genes),
        )

    except requests.RequestException as e:
        logger.warning("Monarch request failed: %s.", e)
        if allow_mocks:
            return _mock_monarch(disease_name, max_results)
        return _empty
    except Exception as e:
        logger.warning("Monarch query error: %s.", e)
        if allow_mocks:
            return _mock_monarch(disease_name, max_results)
        return _empty

    if not nodes:
        if allow_mocks:
            return _mock_monarch(disease_name, max_results)
        return _empty

    return nodes, edges


# ── Internal helpers ────────────────────────────────────────────────────


def _resolve_disease_id(disease_name: str) -> str | None:
    """Resolve a disease name to a MONDO ID via Monarch search API.

    Prefers exact name matches.  When none match exactly, ranks
    candidates by descendant count (broader diseases rank higher),
    which prevents "HIV" from resolving to "HIV enteropathy" instead
    of "HIV infectious disease".
    """
    try:
        resp = requests.get(
            f"{MONARCH_BASE}/search",
            params={
                "q": disease_name,
                "category": "biolink:Disease",
                "limit": 10,
            },
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            return None

        data = resp.json()
        items = data.get("items", [])
        if not items:
            return None

        name_lower = disease_name.lower().strip()

        # 1. Exact match
        for item in items:
            if item.get("name", "").lower() == name_lower:
                return item["id"]

        # 2. Among items whose name contains the query, prefer the
        #    *broadest* concept (most descendants).  A broad disease
        #    like "HIV infectious disease" has many sub-types, while
        #    "HIV enteropathy" has few.
        candidates = [
            item for item in items
            if name_lower in item.get("name", "").lower()
        ]
        if candidates:
            candidates.sort(
                key=lambda it: it.get("has_descendant_count", 0),
                reverse=True,
            )
            return candidates[0]["id"]

        # 3. Last resort: first result
        return items[0]["id"]

    except Exception:
        return None


def _query_associations(
    mondo_id: str,
    category: str,
    limit: int = 50,
) -> list[dict]:
    """Query Monarch for gene-disease associations of a specific type."""
    try:
        resp = requests.get(
            f"{MONARCH_BASE}/association",
            params={
                "object": mondo_id,
                "category": category,
                "limit": limit,
            },
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            return []

        data = resp.json()
        return data.get("items", [])

    except Exception:
        return []


# ── Mock data for tests ─────────────────────────────────────────────────


def _mock_monarch(
    disease_name: str,
    max_results: int = 50,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Generate generic mock Monarch-style data for any disease.

    Returns a small set of well-known, broadly disease-relevant
    human genes.  These are NOT disease-specific — they represent
    commonly studied drug-target families so the pipeline can
    exercise downstream steps (scoring, generation, docking)
    even without API access.

    Real disease–gene associations come exclusively from the
    live Monarch Initiative API.
    """
    disease_key = disease_name.lower().strip()
    disease_node_id = f"disease:{disease_key.replace(' ', '_')}"

    # Generic broadly-studied human genes — no disease bias
    # (descending score so downstream ranking has variance)
    generic_genes: list[tuple[str, float, str]] = [
        ("EGFR",  0.70, "causal"),
        ("TP53",  0.68, "causal"),
        ("TNF",   0.55, "correlated"),
        ("IL6",   0.50, "correlated"),
        ("VEGFA", 0.48, "correlated"),
        ("AKT1",  0.45, "correlated"),
        ("MTOR",  0.42, "correlated"),
        ("PIK3CA", 0.40, "correlated"),
    ]

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    for gene_symbol, score, assoc_type in generic_genes[:max_results]:
        node_id = f"gene:{gene_symbol}"
        nodes.append(GraphNode(
            node_id=node_id,
            name=gene_symbol,
            node_type=NodeType.GENE,
            source="Monarch (mock)",
            score=score,
            metadata={"mock": True, "association_type": assoc_type},
        ))
        edges.append(GraphEdge(
            source_id=node_id,
            target_id=disease_node_id,
            edge_type=EdgeType.ASSOCIATED_WITH,
            weight=score,
            source_db="Monarch (mock)",
            evidence=f"Mock {assoc_type} association: {score:.2f}",
        ))

    logger.info("Monarch (mock): %d generic genes for '%s'.", len(nodes), disease_name)
    return nodes, edges
