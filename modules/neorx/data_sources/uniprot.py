"""
UniProt Protein Knowledge-Base Client
======================================

UniProt (https://www.uniprot.org/) is the most comprehensive
curated protein database.  For drug target identification we
extract:

- **Function annotation**: what the protein actually does
- **PDB cross-references**: available 3D structures for docking
- **Subcellular location**: druggable targets are often
  extracellular or membrane-bound
- **Disease associations**: confirms causal linkage
- **GO terms**: gene ontology for mechanistic context

The 2022 REST API (`rest.uniprot.org`) replaced the legacy one.
We query by gene symbol with taxon filter for *Homo sapiens*
(taxon:9606).

Why this matters for causal inference:
--------------------------------------
A gene that is merely *associated* with a disease in GWAS may
have no druggable protein product.  UniProt tells us whether
the protein product is:
1. A receptor/kinase/enzyme (druggable classes)
2. Has known 3D structures (essential for DockBot)
3. Is located at the cell surface (accessible to small molecules)
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from ..models import GraphNode, NodeType

logger = logging.getLogger(__name__)

UNIPROT_BASE = "https://rest.uniprot.org"
TIMEOUT = 30


def query_uniprot(
    gene_symbols: list[str],
    *,
    allow_mocks: bool = False,
) -> dict[str, dict[str, Any]]:
    """Fetch protein metadata from UniProt for each gene symbol.

    Parameters
    ----------
    gene_symbols : list[str]
        Gene symbols to look up (e.g. ["CCR5", "CD4"]).
    allow_mocks : bool
        If *True*, fall back to curated mock data per gene.
        If *False* (default), skip genes whose API call fails.

    Returns
    -------
    dict[str, dict]
        Keyed by gene symbol.  Each value contains:
        - uniprot_id : str
        - protein_name : str
        - function : str
        - pdb_ids : list[str]
        - subcellular_location : str
        - is_druggable : bool
        - go_terms : list[str]
    """
    results: dict[str, dict[str, Any]] = {}

    for gene in gene_symbols:
        info = _query_single(gene, allow_mocks=allow_mocks)
        if info:
            results[gene] = info

    return results


def _query_single(gene: str, *, allow_mocks: bool = False) -> dict[str, Any] | None:
    """Query UniProt for a single gene symbol."""
    try:
        resp = requests.get(
            f"{UNIPROT_BASE}/uniprotkb/search",
            params={
                "query": f"gene_exact:{gene} AND organism_id:9606 AND reviewed:true",
                "format": "json",
                "fields": (
                    "accession,gene_names,protein_name,"
                    "cc_function,xref_pdb,cc_subcellular_location,"
                    "go_id,go"
                ),
                "size": 1,
            },
            timeout=TIMEOUT,
        )

        if resp.status_code != 200:
            logger.warning("UniProt returned %d for %s.", resp.status_code, gene)
            if allow_mocks:
                return _mock_uniprot_single(gene)
            return None

        data = resp.json()
        results = data.get("results", [])
        if not results:
            if allow_mocks:
                return _mock_uniprot_single(gene)
            return None

        entry = results[0]
        accession = entry.get("primaryAccession", "")

        # Protein name
        prot_desc = entry.get("proteinDescription", {})
        rec_name = prot_desc.get("recommendedName", {})
        full_name = rec_name.get("fullName", {}).get("value", gene)

        # Function
        comments = entry.get("comments", [])
        function_text = ""
        location_text = ""
        for c in comments:
            if c.get("commentType") == "FUNCTION":
                texts = c.get("texts", [])
                if texts:
                    function_text = texts[0].get("value", "")
            if c.get("commentType") == "SUBCELLULAR LOCATION":
                locs = c.get("subcellularLocations", [])
                parts = []
                for loc in locs:
                    loc_val = loc.get("location", {}).get("value", "")
                    if loc_val:
                        parts.append(loc_val)
                location_text = "; ".join(parts)

        # PDB cross-references
        xrefs = entry.get("uniProtKBCrossReferences", [])
        pdb_ids = [x["id"] for x in xrefs if x.get("database") == "PDB"]

        # GO terms
        go_terms = []
        for x in xrefs:
            if x.get("database") == "GO":
                props = x.get("properties", [])
                for p in props:
                    if p.get("key") == "GoTerm":
                        go_terms.append(p.get("value", ""))

        # Heuristic druggability
        druggable_keywords = [
            "receptor", "kinase", "protease", "enzyme",
            "channel", "transporter", "gpcr", "nuclear receptor",
        ]
        is_druggable = any(
            kw in full_name.lower() or kw in function_text.lower()
            for kw in druggable_keywords
        )

        return {
            "uniprot_id": accession,
            "protein_name": full_name,
            "function": function_text,
            "pdb_ids": pdb_ids[:10],
            "subcellular_location": location_text,
            "is_druggable": is_druggable or len(pdb_ids) > 0,
            "go_terms": go_terms[:20],
        }

    except requests.RequestException as e:
        logger.warning("UniProt request failed for %s: %s.", gene, e)
        if allow_mocks:
            return _mock_uniprot_single(gene)
        return None


def _mock_uniprot_single(gene: str) -> dict[str, Any] | None:
    """Generate generic mock UniProt data for any gene.

    Returns a synthetic protein record with plausible metadata
    so downstream steps can run without API access.  No gene-specific
    lookup — real protein data comes from the live UniProt API.
    """
    # Deterministic mock UniProt accession from gene name
    h = abs(hash(gene))
    uid = f"P{h % 99999:05d}"

    # Generate deterministic mock PDB IDs
    pdb1 = f"{h % 10}{chr(65 + h % 26)}{chr(65 + (h >> 4) % 26)}{chr(65 + (h >> 8) % 26)}".upper()
    pdb2 = f"{(h >> 12) % 10}{chr(65 + (h >> 16) % 26)}{chr(65 + (h >> 20) % 26)}{chr(65 + (h >> 24) % 26)}".upper()

    info: dict[str, Any] = {
        "uniprot_id": uid,
        "protein_name": f"{gene} protein",
        "function": f"Protein encoded by {gene}.",
        "pdb_ids": [pdb1[:4], pdb2[:4]],
        "subcellular_location": "Cell membrane",
        "is_druggable": True,
        "go_terms": [
            "C:integral component of membrane",
            "F:protein binding",
            "P:signal transduction",
        ],
    }
    logger.info("UniProt (mock): generic data for %s.", gene)
    return info
