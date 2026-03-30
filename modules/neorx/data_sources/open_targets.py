"""
Open Targets Platform API Client
==================================

Open Targets (https://platform.opentargets.org/) integrates
evidence from genomics, transcriptomics, drugs, animal models,
and literature to link targets to diseases.

Uses the **GraphQL API** (https://api.platform.opentargets.org/api/v4/graphql).

Key concepts:
- **Association score** — how strongly a target is linked to a disease
  (0–1), computed from multiple evidence types
- **Data types** — genetic_association, somatic_mutation, known_drug,
  affected_pathway, literature, animal_model, rna_expression
- **Tractability** — can this target be drugged? (small molecule,
  antibody, other modalities)

Open Targets is free, no API key required, rate limit ~10 req/s.
"""

from __future__ import annotations

import logging
from typing import Optional

import requests

from ..models import GraphNode, GraphEdge, NodeType, EdgeType

logger = logging.getLogger(__name__)

OT_GRAPHQL = "https://api.platform.opentargets.org/api/v4/graphql"
TIMEOUT = 30


def query_open_targets(
    disease_name: str,
    max_results: int = 25,
    *,
    allow_mocks: bool = False,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Query Open Targets for target-disease associations.

    Uses the GraphQL API to search for a disease and retrieve
    associated targets ranked by overall association score.

    Parameters
    ----------
    disease_name : str
        Disease name (e.g. "HIV infection", "Alzheimer disease").
    max_results : int
        Maximum targets to return.
    allow_mocks : bool
        If *True*, fall back to curated mock data on API failure.
        If *False* (default), return empty results.

    Returns
    -------
    tuple[list[GraphNode], list[GraphEdge]]
        Discovered nodes and edges.
    """
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    _empty: tuple[list[GraphNode], list[GraphEdge]] = ([], [])

    try:
        # Step 1: Search for disease EFO ID
        disease_id = _search_disease_id(disease_name)
        if not disease_id:
            logger.info("Open Targets: no disease ID for '%s'.", disease_name)
            if allow_mocks:
                return _mock_open_targets(disease_name, max_results)
            return _empty

        # Step 2: Query associated targets
        query = """
        query AssociatedTargets($diseaseId: String!, $size: Int!) {
          disease(efoId: $diseaseId) {
            name
            associatedTargets(page: {size: $size, index: 0}) {
              rows {
                target {
                  id
                  approvedSymbol
                  approvedName
                  proteinIds {
                    id
                    source
                  }
                }
                score
                datatypeScores {
                  id
                  score
                }
              }
            }
          }
        }
        """

        resp = requests.post(
            OT_GRAPHQL,
            json={
                "query": query,
                "variables": {"diseaseId": disease_id, "size": max_results},
            },
            timeout=TIMEOUT,
        )

        if resp.status_code != 200:
            logger.warning("Open Targets returned %d.", resp.status_code)
            if allow_mocks:
                return _mock_open_targets(disease_name, max_results)
            return _empty

        data = resp.json().get("data", {}).get("disease", {})
        if not data:
            if allow_mocks:
                return _mock_open_targets(disease_name, max_results)
            return _empty

        disease_node_id = f"disease:{disease_name.lower().replace(' ', '_')}"
        rows = data.get("associatedTargets", {}).get("rows", [])

        for row in rows:
            target = row.get("target", {})
            score = row.get("score", 0.0)

            gene_symbol = target.get("approvedSymbol", "")
            ensembl_id = target.get("id", "")
            description = target.get("approvedName", "")

            # Extract UniProt ID
            uniprot_id = ""
            for pid in target.get("proteinIds", []):
                if pid.get("source") == "uniprot_swissprot":
                    uniprot_id = pid.get("id", "")
                    break

            if not gene_symbol:
                continue

            node_id = f"gene:{gene_symbol}"

            # Datatype breakdown
            dt_scores = {dt["id"]: dt["score"] for dt in row.get("datatypeScores", [])}

            node = GraphNode(
                node_id=node_id,
                name=gene_symbol,
                node_type=NodeType.GENE,
                source="Open Targets",
                score=min(score, 1.0),
                uniprot_id=uniprot_id or None,
                description=description,
                metadata={
                    "ensembl_id": ensembl_id,
                    "datatype_scores": dt_scores,
                    "has_known_drug": dt_scores.get("known_drug", 0) > 0,
                },
            )
            nodes.append(node)

            edges.append(GraphEdge(
                source_id=node_id,
                target_id=disease_node_id,
                edge_type=EdgeType.ASSOCIATED_WITH,
                weight=min(score, 1.0),
                source_db="Open Targets",
                evidence=f"OT score: {score:.3f}",
            ))

        logger.info("Open Targets: %d targets for '%s'.", len(nodes), disease_name)

    except requests.RequestException as e:
        logger.warning("Open Targets request failed: %s.", e)
        if allow_mocks:
            return _mock_open_targets(disease_name, max_results)
        return _empty

    if not nodes:
        if allow_mocks:
            return _mock_open_targets(disease_name, max_results)
        return _empty

    return nodes, edges


def _search_disease_id(disease_name: str) -> Optional[str]:
    """Search Open Targets for a disease EFO ID."""
    query = """
    query SearchDisease($name: String!) {
      search(queryString: $name, entityNames: ["disease"], page: {size: 1, index: 0}) {
        hits {
          id
          name
          entity
        }
      }
    }
    """
    try:
        resp = requests.post(
            OT_GRAPHQL,
            json={"query": query, "variables": {"name": disease_name}},
            timeout=TIMEOUT,
        )
        if resp.status_code != 200:
            return None

        hits = resp.json().get("data", {}).get("search", {}).get("hits", [])
        for hit in hits:
            if hit.get("entity") == "disease":
                return hit.get("id")
    except requests.RequestException:
        pass

    return None


def resolve_disease_id(disease_name: str) -> str | None:
    """Resolve a free-text disease name to an EFO/MONDO ontology ID.

    Delegates to the Open Targets search endpoint and returns
    the best-matching disease identifier (e.g. ``EFO_0000764``
    for *HIV infection*).  Returns ``None`` when no match is
    found or the API is unreachable.

    Parameters
    ----------
    disease_name : str
        Human-readable disease name.

    Returns
    -------
    str or None
        Ontology identifier or *None*.
    """
    return _search_disease_id(disease_name)


def _mock_open_targets(
    disease_name: str,
    max_results: int = 25,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Generate generic mock Open Targets data for any disease.

    Returns a small set of broadly-studied drug targets with
    realistic-looking scores and UniProt IDs.  No disease-specific
    gene lists — real associations come from the live API.
    """
    disease_node_id = f"disease:{disease_name.lower().replace(' ', '_')}"

    # Generic well-known drug targets (disease-agnostic)
    generic_targets: list[tuple[str, float, str, bool]] = [
        ("EGFR",   0.65, "P00533", True),
        ("TP53",   0.60, "P04637", False),
        ("TNF",    0.55, "P01375", True),
        ("PIK3CA", 0.50, "P42336", True),
        ("AKT1",   0.48, "P31749", True),
        ("MTOR",   0.45, "P42345", True),
    ]

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    for gene, score, uniprot, has_drug in generic_targets[:max_results]:
        node_id = f"gene:{gene}"
        nodes.append(GraphNode(
            node_id=node_id,
            name=gene,
            node_type=NodeType.GENE,
            source="Open Targets (mock)",
            score=score,
            uniprot_id=uniprot,
            metadata={"has_known_drug": has_drug, "mock": True},
        ))
        edges.append(GraphEdge(
            source_id=node_id,
            target_id=disease_node_id,
            edge_type=EdgeType.ASSOCIATED_WITH,
            weight=score,
            source_db="Open Targets (mock)",
        ))

    logger.info("Open Targets (mock): %d generic targets for '%s'.", len(nodes), disease_name)
    return nodes, edges
