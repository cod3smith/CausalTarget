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
) -> dict[str, dict[str, Any]]:
    """Fetch protein metadata from UniProt for each gene symbol.

    Parameters
    ----------
    gene_symbols : list[str]
        Gene symbols to look up (e.g. ["CCR5", "CD4"]).

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
        info = _query_single(gene)
        if info:
            results[gene] = info

    return results


def _query_single(gene: str) -> dict[str, Any] | None:
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
            logger.warning("UniProt returned %d for %s. Using mock.", resp.status_code, gene)
            return _mock_uniprot_single(gene)

        data = resp.json()
        results = data.get("results", [])
        if not results:
            return _mock_uniprot_single(gene)

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
        return _mock_uniprot_single(gene)


def _mock_uniprot_single(gene: str) -> dict[str, Any] | None:
    """Return curated mock data for common drug-target genes."""
    _mock: dict[str, dict[str, Any]] = {
        "CCR5": {
            "uniprot_id": "P51681",
            "protein_name": "C-C chemokine receptor type 5",
            "function": (
                "Receptor for chemokines CCL3, CCL4, and CCL5. "
                "Acts as a coreceptor for HIV-1 R5 strains."
            ),
            "pdb_ids": ["4MBS", "5UIW", "6AKX", "6MEO", "7F1R"],
            "subcellular_location": "Cell membrane; Multi-pass membrane protein",
            "is_druggable": True,
            "go_terms": [
                "C:integral component of membrane",
                "F:C-C chemokine receptor activity",
                "P:chemokine-mediated signaling pathway",
                "P:viral entry into host cell",
            ],
        },
        "CXCR4": {
            "uniprot_id": "P61073",
            "protein_name": "C-X-C chemokine receptor type 4",
            "function": (
                "Receptor for CXCL12/SDF-1. "
                "Acts as a coreceptor for X4-tropic HIV-1."
            ),
            "pdb_ids": ["3ODU", "3OE0", "3OE6", "3OE8", "3OE9"],
            "subcellular_location": "Cell membrane; Multi-pass membrane protein",
            "is_druggable": True,
            "go_terms": [
                "C:integral component of membrane",
                "F:CXCR chemokine receptor activity",
                "P:cell chemotaxis",
            ],
        },
        "CD4": {
            "uniprot_id": "P01730",
            "protein_name": "T-cell surface glycoprotein CD4",
            "function": (
                "Co-receptor for MHC class II molecules. "
                "Primary receptor for HIV-1 gp120."
            ),
            "pdb_ids": ["1CDH", "1CDJ", "1GC1", "1WIO", "3JWD"],
            "subcellular_location": "Cell membrane; Single-pass type I membrane protein",
            "is_druggable": True,
            "go_terms": [
                "C:external side of plasma membrane",
                "F:MHC class II protein binding",
                "P:T cell activation",
            ],
        },
        "TNF": {
            "uniprot_id": "P01375",
            "protein_name": "Tumor necrosis factor",
            "function": (
                "Pro-inflammatory cytokine involved in systemic "
                "inflammation. Binds TNFRSF1A/TNFR1 and TNFRSF1B/TNFR2."
            ),
            "pdb_ids": ["1TNF", "2AZ5", "2ZJC", "2ZPX", "5TSW"],
            "subcellular_location": "Secreted; Cell membrane",
            "is_druggable": True,
            "go_terms": [
                "F:tumor necrosis factor receptor binding",
                "P:inflammatory response",
                "P:apoptotic process",
            ],
        },
        "NFKB1": {
            "uniprot_id": "P19838",
            "protein_name": "Nuclear factor NF-kappa-B p105 subunit",
            "function": (
                "Key transcription factor in immunity and inflammation."
            ),
            "pdb_ids": ["1SVC", "1VKX"],
            "subcellular_location": "Nucleus; Cytoplasm",
            "is_druggable": False,
            "go_terms": [
                "F:DNA-binding transcription factor activity",
                "P:NF-kappaB signaling",
            ],
        },
        "TP53": {
            "uniprot_id": "P04637",
            "protein_name": "Cellular tumor antigen p53",
            "function": (
                "Tumor suppressor that responds to DNA damage by "
                "regulating target genes involved in cell cycle arrest."
            ),
            "pdb_ids": ["1TSR", "2AC0", "2AHI", "3KMD", "5HOU"],
            "subcellular_location": "Nucleus",
            "is_druggable": False,
            "go_terms": [
                "F:DNA-binding transcription factor activity",
                "P:apoptotic process",
                "P:cell cycle arrest",
            ],
        },
        "BRCA1": {
            "uniprot_id": "P38398",
            "protein_name": "Breast cancer type 1 susceptibility protein",
            "function": "E3 ubiquitin-protein ligase in DNA repair.",
            "pdb_ids": ["1JM7", "1T15", "1T29", "3K0H", "4OFB"],
            "subcellular_location": "Nucleus",
            "is_druggable": False,
            "go_terms": [
                "P:DNA repair",
                "P:double-strand break repair",
            ],
        },
        "EGFR": {
            "uniprot_id": "P00533",
            "protein_name": "Epidermal growth factor receptor",
            "function": (
                "Receptor tyrosine kinase binding EGF family ligands."
            ),
            "pdb_ids": ["1NQL", "2GS6", "3POZ", "4HJO", "5UG9"],
            "subcellular_location": "Cell membrane; Single-pass type I membrane protein",
            "is_druggable": True,
            "go_terms": [
                "F:receptor tyrosine kinase activity",
                "P:MAPK cascade",
            ],
        },
        "INS": {
            "uniprot_id": "P01308",
            "protein_name": "Insulin",
            "function": "Hormone that regulates glucose homeostasis.",
            "pdb_ids": ["4INS", "1MSO", "1ZNI"],
            "subcellular_location": "Secreted",
            "is_druggable": False,
            "go_terms": [
                "P:glucose homeostasis",
                "F:insulin receptor binding",
            ],
        },
        "APOBEC3G": {
            "uniprot_id": "Q9HC16",
            "protein_name": "DNA dC->dU-editing enzyme APOBEC-3G",
            "function": (
                "Innate anti-retroviral immunity factor. "
                "Deaminates HIV-1 cDNA causing G-to-A hypermutation."
            ),
            "pdb_ids": ["3IR2", "3IQS", "6BUX"],
            "subcellular_location": "Cytoplasm",
            "is_druggable": False,
            "go_terms": [
                "F:cytidine deaminase activity",
                "P:defense response to virus",
            ],
        },
        "BST2": {
            "uniprot_id": "Q10589",
            "protein_name": "Bone marrow stromal antigen 2 (Tetherin)",
            "function": (
                "IFN-induced antiviral host factor that tethers "
                "budding HIV-1 virions to the cell surface."
            ),
            "pdb_ids": ["3MQB", "3MQ7"],
            "subcellular_location": "Cell membrane",
            "is_druggable": False,
            "go_terms": [
                "P:defense response to virus",
                "C:cell surface",
            ],
        },
        "TRIM5": {
            "uniprot_id": "Q9C035",
            "protein_name": "Tripartite motif-containing protein 5",
            "function": (
                "Restriction factor recognising retroviral capsid."
            ),
            "pdb_ids": ["2LM3", "4TN3"],
            "subcellular_location": "Cytoplasm",
            "is_druggable": False,
            "go_terms": [
                "P:defense response to virus",
            ],
        },
        "IFNG": {
            "uniprot_id": "P01579",
            "protein_name": "Interferon gamma",
            "function": "Key cytokine for innate and adaptive immunity.",
            "pdb_ids": ["1FG9", "1HIG"],
            "subcellular_location": "Secreted",
            "is_druggable": False,
            "go_terms": [
                "P:defense response to virus",
                "F:cytokine activity",
            ],
        },
    }

    info = _mock.get(gene)
    if info:
        logger.info("UniProt (mock): data for %s.", gene)
    return info
