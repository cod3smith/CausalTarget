"""
DisGeNET API Client
====================

DisGeNET (https://www.disgenet.org/) is the largest public
repository of gene-disease associations.  It integrates data
from expert-curated databases (UniProt, ClinGen, CTD, CGI),
animal models, and text-mining of the scientific literature.

Each association has a **GDA score** (Gene-Disease Association
score) from 0 to 1, reflecting the volume and quality of
evidence.  A score > 0.3 is generally considered reliable.

API: https://www.disgenet.org/api
Free tier: 10 requests/minute (no API key required for basic use)

What we extract:
- Gene symbols, Ensembl IDs, UniProt accessions
- GDA scores (used as edge weights in our causal graph)
- Disease specificity index (DSI) — how specific is this gene
  to the disease vs. being associated with many diseases?

A gene with high GDA score AND high DSI is a strong candidate
for causal involvement.
"""

from __future__ import annotations

import logging
from typing import Optional

import requests

from ..models import GraphNode, GraphEdge, NodeType, EdgeType

logger = logging.getLogger(__name__)

DISGENET_BASE = "https://www.disgenet.org/api"
TIMEOUT = 30


def query_disgenet(
    disease_name: str,
    min_score: float = 0.1,
    max_results: int = 50,
    api_key: Optional[str] = None,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Query DisGeNET for gene-disease associations.

    Parameters
    ----------
    disease_name : str
        Disease name to search for (e.g. "HIV", "Type 2 Diabetes").
    min_score : float
        Minimum GDA score filter (0–1).
    max_results : int
        Maximum number of associations to return.
    api_key : str, optional
        DisGeNET API key for higher rate limits.

    Returns
    -------
    tuple[list[GraphNode], list[GraphEdge]]
        Discovered nodes and edges.
    """
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    # Step 1: Search for the disease to get its CUI/ID
    try:
        headers = {"accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Search diseases
        resp = requests.get(
            f"{DISGENET_BASE}/gda/disease/{disease_name}",
            params={"source": "ALL", "format": "json"},
            headers=headers,
            timeout=TIMEOUT,
        )

        if resp.status_code == 429:
            logger.warning("DisGeNET rate limit hit.  Using cached/mock data.")
            return _mock_disgenet(disease_name, max_results)

        if resp.status_code != 200:
            logger.warning(
                "DisGeNET returned %d for '%s'.  Falling back to mock.",
                resp.status_code, disease_name,
            )
            return _mock_disgenet(disease_name, max_results)

        data = resp.json()
        if not data:
            logger.info("No DisGeNET results for '%s'.", disease_name)
            return _mock_disgenet(disease_name, max_results)

        # Parse associations
        disease_node_id = f"disease:{disease_name.lower().replace(' ', '_')}"

        for assoc in data[:max_results]:
            score = assoc.get("score", 0.0)
            if score < min_score:
                continue

            gene_symbol = assoc.get("gene_symbol", "")
            gene_id = assoc.get("geneid", "")
            uniprot = assoc.get("uniprotid", "")

            if not gene_symbol:
                continue

            node_id = f"gene:{gene_symbol}"
            node = GraphNode(
                node_id=node_id,
                name=gene_symbol,
                node_type=NodeType.GENE,
                source="DisGeNET",
                score=min(score, 1.0),
                uniprot_id=uniprot if uniprot else None,
                description=assoc.get("disease_name", ""),
                metadata={
                    "geneid": str(gene_id),
                    "dsi": assoc.get("dsi", 0.0),
                    "dpi": assoc.get("dpi", 0.0),
                    "ei": assoc.get("ei", 0.0),
                },
            )
            nodes.append(node)

            edge = GraphEdge(
                source_id=node_id,
                target_id=disease_node_id,
                edge_type=EdgeType.ASSOCIATED_WITH,
                weight=min(score, 1.0),
                source_db="DisGeNET",
                evidence=f"GDA score: {score:.3f}",
                pmids=[str(p) for p in assoc.get("pmid", "").split(";")[:5] if p],
            )
            edges.append(edge)

        logger.info("DisGeNET: found %d genes for '%s'.", len(nodes), disease_name)

    except requests.RequestException as e:
        logger.warning("DisGeNET request failed: %s.  Using mock data.", e)
        return _mock_disgenet(disease_name, max_results)

    if not nodes:
        return _mock_disgenet(disease_name, max_results)

    return nodes, edges


def _mock_disgenet(
    disease_name: str,
    max_results: int = 50,
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Generate mock DisGeNET-style data for common diseases.

    This fallback ensures the pipeline works for demos and tests
    even without API access.  The mock data is curated from
    well-known disease biology.
    """
    disease_key = disease_name.lower().strip()
    disease_node_id = f"disease:{disease_key.replace(' ', '_')}"

    # Curated mock data for validation diseases
    mock_db: dict[str, list[tuple[str, float, str]]] = {
        "hiv": [
            ("GAG", 0.95, "P04585"),       # HIV Gag polyprotein
            ("POL", 0.95, "P04587"),       # HIV reverse transcriptase/protease
            ("ENV", 0.90, "P04578"),       # HIV envelope gp160 → gp120+gp41
            ("CCR5", 0.85, "P51681"),      # Co-receptor (maraviroc target)
            ("CXCR4", 0.80, "P61073"),     # Co-receptor
            ("CD4", 0.85, "P01730"),       # Primary receptor
            ("TNF", 0.60, "P01375"),       # Correlational bystander
            ("NFKB1", 0.55, "P19838"),     # NF-κB pathway
            ("IFNG", 0.50, "P01579"),      # Interferon-gamma
            ("APOBEC3G", 0.70, "Q9HC16"), # Host restriction factor
            ("TRIM5", 0.65, "Q9C035"),     # Host restriction factor
            ("BST2", 0.60, "Q10589"),      # Tetherin
        ],
        "type 2 diabetes": [
            ("INS", 0.95, "P01308"),       # Insulin
            ("INSR", 0.90, "P06213"),      # Insulin receptor
            ("GCK", 0.85, "P35557"),       # Glucokinase (MODY2)
            ("PPARG", 0.85, "P37231"),     # PPARγ (glitazone target)
            ("SLC2A4", 0.80, "P14672"),    # GLUT4 transporter
            ("TCF7L2", 0.80, "Q9NQB0"),    # WNT signalling
            ("KCNJ11", 0.75, "Q14654"),    # K+ channel (sulfonylurea target)
            ("ABCC8", 0.75, "Q09428"),     # SUR1 (sulfonylurea receptor)
            ("IRS1", 0.70, "P35568"),      # Insulin receptor substrate 1
            ("AKT2", 0.65, "Q9Y243"),      # PI3K-AKT pathway
        ],
        "alzheimer": [
            ("APP", 0.95, "P05067"),       # Amyloid precursor protein
            ("PSEN1", 0.90, "P49768"),     # Presenilin-1 (γ-secretase)
            ("PSEN2", 0.85, "P49810"),     # Presenilin-2
            ("MAPT", 0.85, "P10636"),      # Tau protein
            ("APOE", 0.80, "P02649"),      # ApoE (risk factor)
            ("BACE1", 0.80, "P56817"),     # β-secretase
            ("TREM2", 0.70, "Q9NZC2"),     # Microglial receptor
            ("CLU", 0.65, "P10909"),       # Clusterin
        ],
        "breast cancer": [
            ("BRCA1", 0.95, "P38398"),
            ("BRCA2", 0.90, "P51587"),
            ("TP53", 0.85, "P04637"),
            ("ERBB2", 0.85, "P04626"),     # HER2
            ("ESR1", 0.80, "P03372"),      # Estrogen receptor
            ("PIK3CA", 0.75, "P42336"),
            ("CDH1", 0.70, "P12830"),
            ("PTEN", 0.70, "P60484"),
        ],
    }

    # Find best matching disease
    genes: list[tuple[str, float, str]] = []
    for key, gene_list in mock_db.items():
        if key in disease_key or disease_key in key:
            genes = gene_list
            break

    if not genes:
        # Generic fallback — common cancer/inflammation genes
        genes = [
            ("TP53", 0.70, "P04637"),
            ("EGFR", 0.65, "P00533"),
            ("TNF", 0.55, "P01375"),
            ("IL6", 0.50, "P05231"),
            ("VEGFA", 0.50, "P15692"),
        ]

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []

    for gene_symbol, score, uniprot in genes[:max_results]:
        node_id = f"gene:{gene_symbol}"
        nodes.append(GraphNode(
            node_id=node_id,
            name=gene_symbol,
            node_type=NodeType.GENE,
            source="DisGeNET (mock)",
            score=score,
            uniprot_id=uniprot,
            metadata={"mock": True},
        ))
        edges.append(GraphEdge(
            source_id=node_id,
            target_id=disease_node_id,
            edge_type=EdgeType.ASSOCIATED_WITH,
            weight=score,
            source_db="DisGeNET (mock)",
            evidence=f"Mock GDA score: {score:.2f}",
        ))

    logger.info("DisGeNET (mock): %d genes for '%s'.", len(nodes), disease_name)
    return nodes, edges
