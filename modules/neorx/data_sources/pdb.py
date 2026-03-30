"""
RCSB PDB Structure Client
==========================

The Protein Data Bank (https://www.rcsb.org/) stores every
experimentally determined 3D structure of biological macro-
molecules.  For our drug-discovery pipeline:

1. **DockBot** needs a PDB ID to download and prepare a
   protein receptor for molecular docking.
2. **MirrorFold** can compare predicted folds to known
   crystal structures.
3. Structures with bound ligands reveal the binding site
   and inform pharmacophore models.

We query the RCSB Search API v2 to find structures for a
given UniProt accession, then retrieve summary metadata
(resolution, method, ligands) via the Data API.

Priority ordering
-----------------
- X-ray < 2.5 Å resolution preferred
- Cryo-EM acceptable when no crystal structures exist
- Structures with co-crystallised ligands ranked higher
  (they define the binding pocket)
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from ..models import GraphNode, NodeType

logger = logging.getLogger(__name__)

PDB_SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
PDB_DATA = "https://data.rcsb.org/rest/v1/core/entry"
TIMEOUT = 30


def query_pdb_structures(
    uniprot_ids: dict[str, str],
    max_per_protein: int = 3,
    *,
    allow_mocks: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Find PDB structures for proteins by UniProt accession.

    Parameters
    ----------
    uniprot_ids : dict[str, str]
        Mapping gene_symbol → UniProt accession.
    max_per_protein : int
        Maximum structures to return per protein.
    allow_mocks : bool
        If *True*, fall back to curated mock data per protein.
        If *False* (default), skip proteins whose lookup fails.

    Returns
    -------
    dict[str, list[dict]]
        Keyed by gene symbol.  Each value is a list of dicts:
        - pdb_id : str
        - resolution : float | None
        - method : str
        - title : str
        - has_ligand : bool
    """
    results: dict[str, list[dict[str, Any]]] = {}

    for gene, uid in uniprot_ids.items():
        structs = _query_single(gene, uid, max_per_protein, allow_mocks=allow_mocks)
        if structs:
            results[gene] = structs

    return results


def _query_single(
    gene: str, uniprot_id: str, max_results: int,
    *, allow_mocks: bool = False,
) -> list[dict[str, Any]]:
    """Search RCSB for structures mapping to a UniProt ID."""
    try:
        query = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_polymer_entity_container_identifiers"
                                 ".reference_sequence_identifiers"
                                 ".database_accession",
                    "operator": "exact_match",
                    "value": uniprot_id,
                },
            },
            "return_type": "entry",
            "request_options": {
                "results_content_type": ["experimental"],
                "sort": [
                    {"sort_by": "rcsb_entry_info.resolution_combined",
                     "direction": "asc"},
                ],
                "paginate": {"start": 0, "rows": max_results},
            },
        }

        resp = requests.post(PDB_SEARCH, json=query, timeout=TIMEOUT)
        if resp.status_code != 200:
            logger.warning("PDB search returned %d for %s.", resp.status_code, gene)
            if allow_mocks:
                return _mock_pdb_single(gene)
            return []

        data = resp.json()
        hits = data.get("result_set", [])
        if not hits:
            if allow_mocks:
                return _mock_pdb_single(gene)
            return []

        structures = []
        for hit in hits[:max_results]:
            pdb_id = hit.get("identifier", "")
            meta = _get_entry_meta(pdb_id)
            structures.append({
                "pdb_id": pdb_id,
                "resolution": meta.get("resolution"),
                "method": meta.get("method", "unknown"),
                "title": meta.get("title", ""),
                "has_ligand": meta.get("has_ligand", False),
            })

        return structures

    except requests.RequestException as e:
        logger.warning("PDB request failed for %s: %s.", gene, e)
        if allow_mocks:
            return _mock_pdb_single(gene)
        return []


def _get_entry_meta(pdb_id: str) -> dict[str, Any]:
    """Retrieve resolution, method, title, ligand info for one PDB entry."""
    try:
        resp = requests.get(f"{PDB_DATA}/{pdb_id}", timeout=TIMEOUT)
        if resp.status_code != 200:
            return {}
        d = resp.json()
        info = d.get("rcsb_entry_info", {})

        # Detect ligands: check for non-polymer entity count (small
        # molecules, ions, etc.) OR explicit binding affinity data.
        # ``nonpolymer_entity_count`` > 0 reliably indicates a
        # co-crystallised ligand or small-molecule in the structure.
        has_ligand = (
            bool(d.get("rcsb_binding_affinity"))
            or info.get("nonpolymer_entity_count", 0) > 0
        )

        return {
            "resolution": info.get("resolution_combined", [None])[0]
            if info.get("resolution_combined") else None,
            "method": info.get("experimental_method", "unknown"),
            "title": d.get("struct", {}).get("title", ""),
            "has_ligand": has_ligand,
        }
    except Exception:
        return {}


def _mock_pdb_single(gene: str) -> list[dict[str, Any]]:
    """Generate generic mock PDB structure data for any gene.

    Returns synthetic structure entries with plausible metadata
    so downstream docking steps can run without API access.
    No gene-specific lookup — real PDB data comes from the
    live RCSB API.
    """
    # Generate deterministic mock PDB IDs from gene name
    h = abs(hash(gene))
    pdb_id_1 = f"{h % 10}{chr(65 + h % 26)}{chr(65 + (h >> 4) % 26)}{chr(65 + (h >> 8) % 26)}"
    pdb_id_2 = f"{(h >> 12) % 10}{chr(65 + (h >> 16) % 26)}{chr(65 + (h >> 20) % 26)}{chr(65 + (h >> 24) % 26)}"

    structs = [
        {"pdb_id": pdb_id_1[:4].upper(), "resolution": 2.50, "method": "X-ray",
         "title": f"{gene} crystal structure", "has_ligand": True},
        {"pdb_id": pdb_id_2[:4].upper(), "resolution": 2.80, "method": "X-ray",
         "title": f"{gene} apo structure", "has_ligand": False},
    ]

    logger.info("PDB (mock): %d structures for %s.", len(structs), gene)
    return structs
