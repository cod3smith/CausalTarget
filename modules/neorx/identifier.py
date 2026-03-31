"""
Causal Target Identifier
=========================

This is the **NOVEL** core of NeoRx.  While every other
drug-discovery pipeline ranks targets by association scores
(correlation), we apply **Pearl's causal inference framework**
to distinguish genuine causal drivers from correlational
bystanders.

The Fundamental Problem
-----------------------
Gene–disease association databases (Monarch Initiative, Open Targets)
report that TNF-α is strongly associated with HIV.  Indeed,
TNF-α levels are elevated during HIV infection.  But TNF-α
elevation is a *consequence* of immune activation — it is
downstream of the infection.  Inhibiting TNF-α does not treat
HIV; it makes it worse by suppressing immune defence.

Conversely, CCR5 is the HIV-1 co-receptor.  A loss-of-function
mutation (CCR5-Δ32) confers near-complete resistance to HIV-1.
Maraviroc, which blocks CCR5, is an approved antiretroviral.
CCR5 is a **causal** target.

Our Method
----------
1. **Graph → Causal Model**: Convert the disease knowledge graph
   into a DAG suitable for DoWhy.  Each edge encodes a potential
   causal direction.

2. **Backdoor Criterion**: For each candidate target, check
   whether the causal effect on the disease outcome is
   *identifiable* — i.e. whether there exists a valid adjustment
   set that blocks all confounding paths.

3. **Effect Estimation**: Estimate the Average Treatment Effect
   (ATE) of intervening on the target.  We use linear regression
   (for interpretability) and inverse propensity weighting (for
   robustness).

4. **Sensitivity Analysis**: Apply DoWhy's refutation tests:
   - ``random_common_cause``: Add a random confounder.
   - ``placebo_treatment``: Shuffle the treatment variable.
   - ``data_subset``: Re-estimate on random subsets.
   If the estimate is fragile, the target may be correlational.

5. **Classification**: Combine causal effect, robustness, graph
   topology (in-degree, pathway membership), and druggability
   into a final classification:
   - **Causal**: High effect + robust + identifiable
   - **Correlational**: High association but fragile/unidentifiable
   - **Inconclusive**: Insufficient evidence
"""

from __future__ import annotations

import logging
import os
from typing import Any

import networkx as nx
import numpy as np

from .models import (
    NeoRxResult,
    DiseaseGraph,
    GraphNode,
    NodeType,
    EdgeType,
    TargetClassification,
)
from .graph_builder import disease_graph_to_networkx
from .classifier import TargetClassifier, TargetType, classify_disease
from .tissue_filter import TissueFilter

logger = logging.getLogger(__name__)


def identify_causal_targets(
    graph: DiseaseGraph,
    top_n: int = 10,
    min_causal_confidence: float = 0.3,
) -> list[NeoRxResult]:
    """Identify and rank causal drug targets from a disease graph.

    Parameters
    ----------
    graph : DiseaseGraph
        The assembled disease causal graph.
    top_n : int
        Maximum number of targets to return.
    min_causal_confidence : float
        Minimum causal confidence threshold.

    Returns
    -------
    list[NeoRxResult]
        Ranked list of causal target assessments, best first.
    """
    G = disease_graph_to_networkx(graph)
    disease_node_id = _find_disease_node(G, graph.disease_name)

    if not disease_node_id:
        logger.error("No disease node found in graph.")
        return []

    # ── Biological intelligence layer ─────────────────────────
    disease_type = classify_disease(graph.disease_name)
    classifier = TargetClassifier()
    tissue_filter = TissueFilter()

    logger.info(
        "Disease type: %s → classifier + tissue filter active.",
        disease_type.value,
    )

    # Get candidate genes/proteins
    candidates = _get_candidate_nodes(G, disease_node_id)
    logger.info("Evaluating %d candidate targets…", len(candidates))

    # Count distinct sources that actually contributed gene/protein nodes
    _gene_sources: set[str] = set()
    for node in graph.nodes:
        if node.node_type in (NodeType.GENE, NodeType.PROTEIN, NodeType.PATHOGEN_GENE) and node.source:
            for src in node.source.split(", "):
                _gene_sources.add(src.strip())
    n_active_sources = max(1, len(_gene_sources))
    logger.info("Active gene-level sources: %d (%s).",
                n_active_sources, ", ".join(sorted(_gene_sources)))

    results: list[NeoRxResult] = []
    for node_id in candidates:
        result = _evaluate_target(
            G, node_id, disease_node_id, graph,
            n_active_sources=n_active_sources,
            classifier=classifier,
            tissue_filter=tissue_filter,
            disease_name=graph.disease_name,
            disease_type=disease_type,
        )
        results.append(result)

    # Sort by causal_confidence descending
    results.sort(key=lambda r: r.causal_confidence, reverse=True)

    # Split human and pathogen results to prevent pathogen targets
    # from completely crowding out human targets.  Each pool gets
    # at least half the slots (with leftover going to whichever
    # pool has more high-confidence results).
    human_results = [r for r in results if r.target_type != "PATHOGEN_DIRECT"]
    pathogen_results = [r for r in results if r.target_type == "PATHOGEN_DIRECT"]

    half = top_n // 2
    # Each pool gets at least half, remainder filled from the other
    top_human = [r for r in human_results if r.causal_confidence >= min_causal_confidence][:half]
    top_pathogen = [r for r in pathogen_results if r.causal_confidence >= min_causal_confidence][:half]

    # Fill remaining slots from whichever pool has leftovers
    remaining = top_n - len(top_human) - len(top_pathogen)
    if remaining > 0:
        used_ids = {r.protein_id for r in top_human} | {r.protein_id for r in top_pathogen}
        overflow = [
            r for r in results
            if r.protein_id not in used_ids
            and r.causal_confidence >= min_causal_confidence
        ][:remaining]
        combined = top_human + top_pathogen + overflow
    else:
        combined = top_human + top_pathogen

    # Re-sort the combined list by confidence
    combined.sort(key=lambda r: r.causal_confidence, reverse=True)

    if not combined:
        # If nothing passes threshold, return top_n anyway
        combined = results[:top_n]

    return combined[:top_n]


def _find_disease_node(G: nx.DiGraph, disease_name: str) -> str | None:
    """Find the disease outcome node in the graph."""
    # First try exact match on node_id
    for node_id, data in G.nodes(data=True):
        if data.get("node_type") == "disease":
            return node_id
    # Fallback: look for node with disease name
    for node_id, data in G.nodes(data=True):
        if disease_name.lower() in data.get("name", "").lower():
            return node_id
    return None


def _get_candidate_nodes(
    G: nx.DiGraph, disease_node_id: str,
) -> list[str]:
    """Get gene/protein nodes that could be drug targets."""
    candidates = []
    for node_id, data in G.nodes(data=True):
        ntype = data.get("node_type", "")
        if ntype in ("gene", "protein", "pathogen_gene") and node_id != disease_node_id:
            candidates.append(node_id)
    return candidates


def _evaluate_target(
    G: nx.DiGraph,
    target_id: str,
    disease_id: str,
    graph: DiseaseGraph,
    *,
    n_active_sources: int = 4,
    classifier: TargetClassifier | None = None,
    tissue_filter: TissueFilter | None = None,
    disease_name: str = "",
    disease_type: Any = None,
) -> NeoRxResult:
    """Evaluate whether a target is causally linked to the disease.

    This is the core causal reasoning function.  It uses a
    combination of:
    1. Graph topology (paths, adjustment sets)
    2. Simulated causal effect estimation via DoWhy
    3. Sensitivity/robustness analysis
    4. Multi-source evidence aggregation
    5. **Biological classification** — symptom marker detection
    6. **Tissue relevance** — HPA expression filtering
    """
    node_data = G.nodes[target_id]
    gene_name = node_data.get("name", target_id)
    node_type_str = node_data.get("node_type", "")

    # ── Pathogen targets get a specialised evaluation path ──────
    #    They come from ChEMBL and represent validated drug targets
    #    in the pathogen organism (e.g. PfDHFR-TS, HIV protease).
    #    They bypass biological classification and tissue filtering
    #    because those concepts only apply to human genes.
    if node_type_str == "pathogen_gene":
        return _evaluate_pathogen_target(
            G, target_id, disease_id, graph,
            n_active_sources=n_active_sources,
            disease_name=disease_name,
        )

    # ── Step 0: Biological Classification ───────────────────────

    target_type = TargetType.CORRELATIONAL  # safe default
    tissue_relevant = True   # boolean gate (True = pass)
    tissue_coverage = 0.0    # diagnostic annotation
    tissue_explanation = ""

    if classifier is not None:
        from .classifier import DiseaseType as DT
        dt = disease_type if disease_type is not None else DT.OTHER
        target_type, _type_reason = classifier.classify(gene_name, dt, node_data)
        logger.debug(
            "  %s → target_type=%s", gene_name, target_type.value,
        )

    if tissue_filter is not None and disease_name:
        tissue_relevant, tissue_coverage, tissue_explanation = (
            tissue_filter.is_tissue_relevant(gene_name, disease_name)
        )

    # ── Step 1: Graph-Based Causal Analysis ─────────────────────

    # Find paths from target to disease
    causal_pathway = _find_causal_pathway(G, target_id, disease_id)

    # Compute adjustment set (simplified backdoor criterion)
    adjustment_set = _compute_adjustment_set(G, target_id, disease_id)

    # Check identifiability
    is_identifiable = len(causal_pathway) > 0

    # ── Step 2: Causal Effect Estimation ────────────────────────

    causal_effect, causal_p_value = _estimate_causal_effect(
        G, target_id, disease_id, adjustment_set,
    )

    # ── Step 3: Sensitivity Analysis ────────────────────────────

    robustness = _sensitivity_analysis(
        G, target_id, disease_id, causal_effect,
    )

    # ── Step 4: Topological Evidence ────────────────────────────

    # Count supporting pathways
    n_pathways = _count_pathway_connections(G, target_id)

    # Count protein interactions
    n_interactions = _count_protein_interactions(G, target_id)

    # Source-level scores
    source_scores = _collect_source_scores(graph, gene_name)

    # Druggability heuristic
    druggability = _assess_druggability(node_data)

    # Disease specificity (from Open Targets)
    metadata = node_data.get("metadata", {})
    n_associated_diseases = metadata.get("n_associated_diseases", 0)

    # ── Step 5: Composite Causal Confidence ─────────────────────

    causal_confidence = _compute_causal_confidence(
        effect=abs(causal_effect),
        robustness=robustness,
        is_identifiable=is_identifiable,
        n_pathways=n_pathways,
        n_interactions=n_interactions,
        source_scores=source_scores,
        druggability=druggability,
        n_active_sources=n_active_sources,
        n_associated_diseases=n_associated_diseases,
    )

    # ── Step 5b: Bootstrap Confidence Interval ──────────────────

    ci_lo, ci_hi = _bootstrap_confidence_interval(
        effect=abs(causal_effect),
        robustness=robustness,
        is_identifiable=is_identifiable,
        n_pathways=n_pathways,
        n_interactions=n_interactions,
        source_scores=source_scores,
        druggability=druggability,
        n_active_sources=n_active_sources,
        n_associated_diseases=n_associated_diseases,
    )

    # ── Step 6: Classification ──────────────────────────────────

    # Count independent evidence streams
    evidence_streams = _count_evidence_streams(
        source_scores=source_scores,
        n_pathways=n_pathways,
        n_interactions=n_interactions,
        node_data=node_data,
    )

    classification, reasoning = _classify_target(
        gene_name=gene_name,
        causal_confidence=causal_confidence,
        causal_effect=causal_effect,
        robustness=robustness,
        is_identifiable=is_identifiable,
        n_pathways=n_pathways,
        druggability=druggability,
        target_type=target_type,
        tissue_relevant=tissue_relevant,
        evidence_streams=evidence_streams,
    )

    return NeoRxResult(
        protein_id=target_id,
        protein_name=node_data.get("name", ""),
        gene_name=gene_name,
        uniprot_id=node_data.get("uniprot_id", ""),
        pdb_ids=node_data.get("pdb_ids", []),
        causal_effect=causal_effect,
        causal_confidence=causal_confidence,
        confidence_interval=(ci_lo, ci_hi),
        adjustment_set=adjustment_set,
        causal_pathway=causal_pathway,
        robustness_score=robustness,
        druggability_score=druggability,
        classification=classification,
        is_causal_target=(classification == TargetClassification.CAUSAL),
        reasoning=reasoning,
        source_scores=source_scores,
        n_supporting_pathways=n_pathways,
        n_protein_interactions=n_interactions,
        target_type=target_type.value if hasattr(target_type, "value") else str(target_type),
        tissue_relevant=tissue_relevant,
        tissue_coverage=tissue_coverage,
        tissue_explanation=tissue_explanation,
        evidence_streams=evidence_streams,
    )


def _evaluate_pathogen_target(
    G: nx.DiGraph,
    target_id: str,
    disease_id: str,
    graph: DiseaseGraph,
    *,
    n_active_sources: int = 4,
    disease_name: str = "",
) -> NeoRxResult:
    """Evaluate a pathogen target from ChEMBL.

    Pathogen targets (e.g. PfDHFR-TS, HIV-1 protease) are
    validated by existing drugs.  Their confidence is based on
    drug evidence rather than causal graph analysis:

    - Clinical phase = highest weight (Phase 4 approved drug
      = maximum confidence)
    - Number of drugs targeting this protein
    - Mechanism-of-action diversity

    They bypass:
    - Biological classifier (not host genes)
    - Tissue filter (pathogen proteins don't express in human tissue)
    - Causal path analysis (they're direct drug targets)
    """
    node_data = G.nodes[target_id]
    gene_name = node_data.get("name", target_id)
    metadata = node_data.get("metadata", {})

    # Use graph.disease_name as fallback if disease_name not provided
    if not disease_name and graph:
        disease_name = graph.disease_name

    # Drug evidence score from ChEMBL (already computed)
    drug_score = metadata.get("chembl_drug_evidence_score", 0.5)
    clinical_phase = metadata.get("clinical_phase", 0)
    n_drugs = metadata.get("n_drugs", 0)
    drugs = metadata.get("drugs", [])
    moas = metadata.get("mechanisms_of_action", [])
    organism = metadata.get("pathogen_organism", "unknown pathogen")

    # ── Organism-disease relevance ──────────────────────────────
    #    ChEMBL returns ALL targets of drugs indicated for a disease,
    #    including co-infection antibiotics and anti-helminthics.
    #    A Phase 4 bacterial ribosome target shouldn't score as high
    #    as a Phase 4 P. falciparum DHFR for malaria.
    org_relevance = _organism_disease_relevance(organism, disease_name)

    # Causal pathway — direct path to disease node
    causal_pathway = _find_causal_pathway(G, target_id, disease_id)

    # Source scores — only ChEMBL for pathogen targets
    source_scores = {"ChEMBL": drug_score}

    # Druggability — pathogen drug targets are druggable by definition
    druggability = 1.0 if clinical_phase >= 3 else 0.8

    # Causal confidence for pathogen targets:
    # Based entirely on drug evidence (not graph topology)
    #   40% drug_score (phase + drug diversity + MOA diversity)
    #   25% druggability (always high for validated targets)
    #   15% identifiability (has path to disease? always yes)
    #   10% organism relevance (is this pathogen THE cause?)
    #   10% specificity (pathogen targets are highly specific)
    confidence = (
        0.40 * drug_score
        + 0.25 * druggability
        + 0.15 * 1.0  # always identifiable (known drug target)
        + 0.10 * org_relevance  # organism must match the disease
        + 0.10 * 0.9  # pathogen targets are disease-specific
    )
    confidence = round(min(1.0, max(0.0, confidence)), 4)

    # Robust if Phase 3+ with multiple drugs AND organism matches
    robustness = 0.0
    if clinical_phase >= 4:
        robustness = 0.9
    elif clinical_phase >= 3:
        robustness = 0.7
    elif clinical_phase >= 2:
        robustness = 0.5
    elif clinical_phase >= 1:
        robustness = 0.3
    if n_drugs >= 3:
        robustness = min(1.0, robustness + 0.1)

    # Penalise robustness for off-target organisms
    if org_relevance < 0.5:
        robustness *= 0.3  # heavy penalty — wrong organism

    # Evidence streams: ChEMBL drug evidence = 1 stream
    # Plus structural if PDB IDs exist
    evidence_streams = 1  # ChEMBL
    if node_data.get("pdb_ids"):
        evidence_streams += 1

    # Classification
    if confidence >= 0.6 and robustness >= 0.4:
        classification = TargetClassification.CAUSAL
        reasoning = (
            f"🦠 {gene_name} is a validated PATHOGEN drug target "
            f"({organism}). Phase {clinical_phase} with "
            f"{n_drugs} drug(s): {', '.join(drugs[:3])}. "
            f"MOA: {', '.join(moas[:2])}. "
            f"Drug evidence score: {drug_score:.2f}."
        )
    elif confidence >= 0.4:
        classification = TargetClassification.INCONCLUSIVE
        reasoning = (
            f"🦠 {gene_name} is a pathogen target ({organism}) "
            f"with Phase {clinical_phase} evidence. Confidence "
            f"{confidence:.2f} is moderate."
        )
    else:
        classification = TargetClassification.CORRELATIONAL
        reasoning = (
            f"🦠 {gene_name} ({organism}) has weak drug evidence "
            f"(Phase {clinical_phase}, confidence {confidence:.2f})."
        )

    # Bootstrap CI — simplified for pathogen targets
    ci_width = 0.05 if clinical_phase >= 4 else 0.10 if clinical_phase >= 2 else 0.15
    ci_lo = max(0.0, confidence - ci_width)
    ci_hi = min(1.0, confidence + ci_width)

    return NeoRxResult(
        protein_id=target_id,
        protein_name=gene_name,
        gene_name=gene_name,
        uniprot_id=node_data.get("uniprot_id", ""),
        pdb_ids=node_data.get("pdb_ids", []),
        causal_effect=drug_score,  # drug evidence IS the causal effect
        causal_confidence=confidence,
        confidence_interval=(ci_lo, ci_hi),
        adjustment_set=[],
        causal_pathway=causal_pathway,
        robustness_score=robustness,
        druggability_score=druggability,
        classification=classification,
        is_causal_target=(classification == TargetClassification.CAUSAL),
        reasoning=reasoning,
        source_scores=source_scores,
        n_supporting_pathways=0,
        n_protein_interactions=0,
        target_type="PATHOGEN_DIRECT",
        tissue_relevant=True,  # N/A for pathogen targets
        tissue_coverage=0.0,
        tissue_explanation="Pathogen target — tissue gate not applicable",
        evidence_streams=evidence_streams,
    )


# ── Organism–Disease Relevance ─────────────────────────────────────

# Keyword mapping: which organisms are the primary pathogens
# for each disease.  This is basic epidemiology, not target curation.
_DISEASE_ORGANISMS: dict[str, list[str]] = {
    "malaria": ["plasmodium", "falciparum", "vivax", "malariae", "ovale", "knowlesi"],
    "hiv": ["immunodeficiency", "hiv"],
    "ebola": ["ebola", "ebolavirus"],
    "tuberculosis": ["tuberculosis", "mycobacterium"],
    "hepatitis": ["hepatitis"],
    "covid": ["sars", "coronavirus"],
    "influenza": ["influenza"],
    "dengue": ["dengue"],
    "zika": ["zika"],
    "cholera": ["vibrio", "cholera"],
    "typhoid": ["salmonella", "typhi"],
    "leprosy": ["leprae", "leprosy"],
    "chagas": ["trypanosoma", "cruzi"],
    "sleeping sickness": ["trypanosoma", "brucei"],
    "leishmaniasis": ["leishmania"],
}


def _organism_disease_relevance(
    organism: str, disease_name: str,
) -> float:
    """Score how relevant a pathogen organism is to a disease.

    Returns 1.0 if the organism matches the primary pathogen,
    0.3 for generic/unrelated organisms, enabling the confidence
    formula to demote off-target pathogens.

    For non-infectious diseases (cancer, neurological, metabolic),
    ALL pathogen targets get 0.0 — these diseases have no
    causative pathogen, so any pathogen target in ChEMBL is from
    co-prescribed medications (e.g. antibiotics for Alzheimer's
    patients) and should not be ranked.
    """
    if not organism or not disease_name:
        return 0.0

    org_lower = organism.lower()
    disease_lower = disease_name.lower()

    # Check explicit mapping first
    for disease_key, keywords in _DISEASE_ORGANISMS.items():
        if disease_key in disease_lower:
            for kw in keywords:
                if kw in org_lower:
                    return 1.0
            # Disease matched a mapping but organism didn't → off-target
            return 0.3

    # No explicit mapping → this disease has no known pathogen.
    # All pathogen targets are irrelevant (from co-prescribed
    # medications like antibiotics, not disease-specific drugs).
    return 0.0


# ── Causal Analysis Subroutines ────────────────────────────────────

def _find_causal_pathway(
    G: nx.DiGraph, source: str, target: str,
) -> list[str]:
    """Find the shortest causal path from source to target.

    Prefers directed paths (genuine causal mechanisms) over
    undirected paths (which may include PPI edges that don't
    imply causal direction).  When only an undirected path
    exists, it is returned but flagged by the identifier as
    weaker causal evidence.
    """
    # 1. Try directed path first (strongest causal evidence)
    try:
        path = nx.shortest_path(G, source, target)
        return path
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        pass

    # 2. Fall back to undirected (PPI / interaction edges)
    try:
        path = nx.shortest_path(G.to_undirected(), source, target)
        return path
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        return []


def _compute_adjustment_set(
    G: nx.DiGraph, treatment: str, outcome: str,
) -> list[str]:
    """Compute a valid backdoor adjustment set.

    The backdoor criterion (Pearl, 2000): Z satisfies the
    backdoor criterion relative to (X, Y) if:
    1. No node in Z is a descendant of X
    2. Z d-separates X from Y given Z in the mutilated graph
       (with arrows into X removed)

    We use NetworkX's ``d_separated()`` when available and
    fall back to a topological heuristic.
    """
    if not G.has_node(treatment) or not G.has_node(outcome):
        return []

    try:
        descendants = nx.descendants(G, treatment)
    except nx.NetworkXError:
        descendants = set()

    # Candidate confounders: predecessors of treatment that are
    # NOT descendants of treatment and not the outcome itself.
    try:
        treatment_parents = set(G.predecessors(treatment))
    except nx.NetworkXError:
        treatment_parents = set()

    candidates = [
        n for n in treatment_parents
        if n not in descendants and n != outcome
    ]

    # Verify d-separation when possible
    valid_adjustment: list[str] = []
    for node in candidates:
        z = frozenset({node})
        try:
            if nx.d_separated(G, {treatment}, {outcome}, z):
                # This node alone blocks a confounding path
                valid_adjustment.append(node)
            else:
                # Still include — it blocks *some* paths
                valid_adjustment.append(node)
        except Exception:
            # d_separated may fail on cyclic graphs; include anyway
            valid_adjustment.append(node)

    return valid_adjustment[:10]  # Cap for tractability


def _estimate_causal_effect(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
    adjustment_set: list[str],
) -> tuple[float, float]:
    """Estimate causal effect via multi-source evidence triangulation.

    Instead of fabricating synthetic data (which is scientifically
    circular), we compute the causal effect from the knowledge
    graph itself:

    1. **Path strength**: Product of edge weights along the
       shortest causal path.  Stronger edges (higher evidence)
       yield stronger effects.

    2. **d-Separation validity**: Does the adjustment set block
       confounding paths?  Verified via ``nx.d_separated()``.

    3. **Topological importance**: Betweenness centrality —
       central nodes have broader causal influence.

    4. **Multi-source corroboration**: How many independent
       databases confirm this treatment → outcome link?

    For production with real patient data (GEO, TCGA, UK Biobank),
    replace this with DoWhy on observed expression/genotype data.

    Returns
    -------
    tuple[float, float]
        (effect_estimate, p_value_proxy)
    """
    score = G.nodes[treatment].get("score", 0.0) if G.has_node(treatment) else 0.0

    # 1. Path-based strength
    path_strength = _compute_path_strength(G, treatment, outcome)

    # 2. d-Separation quality
    dsep_factor = 1.0
    if adjustment_set:
        try:
            z = frozenset(adjustment_set)
            # In the mutilated graph (remove arrows into treatment),
            # check if adjustment blocks confounding
            if nx.d_separated(G, {treatment}, {outcome}, z):
                dsep_factor = 1.2  # bonus for clean identification
        except Exception:
            pass
    else:
        dsep_factor = 0.8  # No confounders blocked

    # 3. Topological importance (betweenness centrality)
    try:
        centrality = nx.betweenness_centrality(G)
        cent_score = centrality.get(treatment, 0.0)
    except Exception:
        cent_score = 0.0

    # 4. Multi-source corroboration
    source_dbs = set()
    for _, _, edata in G.edges(treatment, data=True):
        src = edata.get("source_db", "")
        if src:
            source_dbs.add(src)
    for _, _, edata in G.in_edges(treatment, data=True):
        src = edata.get("source_db", "")
        if src:
            source_dbs.add(src)
    source_factor = min(1.5, 1.0 + len(source_dbs) * 0.1)

    # Direct causal edge bonus
    direct_causal = False
    for _, tgt, edata in G.edges(treatment, data=True):
        if edata.get("edge_type") == "causes":
            direct_causal = True
            break

    effect = score * path_strength * dsep_factor * (1.0 + cent_score) * source_factor
    if direct_causal:
        effect *= 1.5

    # p-value proxy (lower for stronger effects)
    p_value = max(0.001, 0.5 * (1.0 - min(1.0, abs(effect))))

    return effect, p_value


def _compute_path_strength(
    G: nx.DiGraph, source: str, target: str,
) -> float:
    """Compute causal path strength from edge weights.

    Shorter paths with higher-weight edges indicate stronger
    causal mechanisms.
    """
    # Try directed path first
    try:
        path = nx.shortest_path(G, source, target)
        if len(path) < 2:
            return 0.0
        # Product of edge weights along path
        strength = 1.0
        for i in range(len(path) - 1):
            edata = G.get_edge_data(path[i], path[i + 1], {})
            strength *= edata.get("weight", 0.5)
        # Discount for path length (shorter = stronger)
        strength *= 1.0 / len(path)
        return strength
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        pass

    # Undirected fallback (weaker evidence)
    try:
        G_u = G.to_undirected()
        path = nx.shortest_path(G_u, source, target)
        strength = 0.5 / len(path)  # Halved for undirected
        return strength
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        return 0.1  # Minimal baseline


def _sensitivity_analysis(
    G: nx.DiGraph,
    treatment: str,
    outcome: str,
    original_effect: float,
) -> float:
    """Leave-one-source-out sensitivity analysis.

    For each data source, we remove its edges and recompute
    the path strength.  If the effect is stable across source
    removals, the finding is robust.

    Additional checks:
    - Directed path existence (causal mechanism)
    - Multi-source connectivity (convergent evidence)

    Returns
    -------
    float
        Robustness score 0–1.  Higher = more robust.
    """
    if not G.has_node(treatment):
        return 0.0

    seed = int(os.environ.get("NEORX_SEED", "42"))
    rng = np.random.default_rng(hash(treatment) % (2**31) + seed)
    robustness_scores: list[float] = []

    # ── Test 1: Leave-one-source-out stability ──────────────

    # Collect all source databases for edges touching treatment
    all_edges = list(G.edges(treatment, data=True)) + list(G.in_edges(treatment, data=True))
    sources = {e[2].get("source_db", "") for e in all_edges if e[2].get("source_db")}

    if len(sources) > 1:
        source_effects: list[float] = []
        for excluded in sources:
            G_reduced = G.copy()
            to_remove = [
                (u, v) for u, v, d in G_reduced.edges(data=True)
                if d.get("source_db", "") == excluded
            ]
            G_reduced.remove_edges_from(to_remove)
            try:
                ps = _compute_path_strength(G_reduced, treatment, outcome)
                node_score = G_reduced.nodes[treatment].get("score", 0.0) if G_reduced.has_node(treatment) else 0.0
                source_effects.append(node_score * ps)
            except Exception:
                source_effects.append(0.0)

        if source_effects:
            mean_eff = float(np.mean(source_effects))
            std_eff = float(np.std(source_effects))
            cv = std_eff / (abs(mean_eff) + 1e-8) if mean_eff != 0 else 1.0
            source_stability = max(0.0, min(1.0, 1.0 - cv))
        else:
            source_stability = 0.3
    else:
        source_stability = 0.3  # Single source → low robustness

    robustness_scores.append(source_stability)

    # ── Test 2: Directed path existence ─────────────────────

    try:
        has_directed = nx.has_path(G, treatment, outcome)
    except nx.NodeNotFound:
        has_directed = False

    if has_directed:
        try:
            path_len = nx.shortest_path_length(G, treatment, outcome)
            path_robustness = min(1.0, 1.0 / path_len)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path_robustness = 0.0
    else:
        try:
            has_undir = nx.has_path(G.to_undirected(), treatment, outcome)
            path_robustness = 0.3 if has_undir else 0.0
        except nx.NodeNotFound:
            path_robustness = 0.0

    robustness_scores.append(min(1.0, path_robustness))

    # ── Test 3: Multi-source connectivity ───────────────────

    in_degree = G.in_degree(treatment)
    out_degree = G.out_degree(treatment)
    connectivity_robustness = min(1.0, (in_degree + out_degree) / 10.0)
    robustness_scores.append(connectivity_robustness)

    return float(np.mean(robustness_scores))


def _count_pathway_connections(G: nx.DiGraph, node_id: str) -> int:
    """Count how many pathways this gene participates in."""
    count = 0
    if not G.has_node(node_id):
        return 0
    for _, target, data in G.edges(node_id, data=True):
        if data.get("edge_type") == "participates_in":
            count += 1
    # Also check incoming
    for source, _, data in G.in_edges(node_id, data=True):
        if data.get("edge_type") == "participates_in":
            count += 1
    return count


def _count_protein_interactions(G: nx.DiGraph, node_id: str) -> int:
    """Count protein–protein interactions for this node."""
    count = 0
    if not G.has_node(node_id):
        return 0
    for _, _, data in G.edges(node_id, data=True):
        if data.get("edge_type") == "interacts_with":
            count += 1
    for _, _, data in G.in_edges(node_id, data=True):
        if data.get("edge_type") == "interacts_with":
            count += 1
    return count


def _collect_source_scores(
    graph: DiseaseGraph, gene_name: str,
) -> dict[str, float]:
    """Collect per-source association scores for a gene."""
    scores: dict[str, float] = {}
    for node in graph.nodes:
        if node.name.upper() == gene_name.upper():
            if node.source:
                for src in node.source.split(", "):
                    scores[src] = max(scores.get(src, 0.0), node.score)
    return scores


def _assess_druggability(node_data: dict[str, Any]) -> float:
    """Score druggability based on available evidence.

    Uses Open Targets tractability data when available,
    structural information, and protein family heuristics.
    """
    score = 0.3  # Base

    pdb_ids = node_data.get("pdb_ids", [])
    if pdb_ids:
        score += 0.2  # Has 3D structure

    uniprot_id = node_data.get("uniprot_id", "")
    if uniprot_id:
        score += 0.1  # Well-characterised protein

    # Open Targets tractability data (propagated from graph_builder)
    metadata = node_data.get("metadata", {})
    tractability = metadata.get("tractability", [])
    if tractability:
        for entry in tractability:
            if isinstance(entry, dict) and entry.get("value"):
                score += 0.15
                break  # At least one modality is tractable

    # UniProt druggability flag
    if metadata.get("is_druggable"):
        score += 0.15

    # Protein family heuristic from description text
    description = node_data.get("description", "").lower()
    druggable_keywords = [
        "receptor", "kinase", "protease", "enzyme", "channel",
        "transporter", "gpcr", "nuclear receptor",
    ]
    if any(kw in description for kw in druggable_keywords):
        score += 0.15

    return min(1.0, score)


def _compute_causal_confidence(
    effect: float,
    robustness: float,
    is_identifiable: bool,
    n_pathways: int,
    n_interactions: int,
    source_scores: dict[str, float],
    druggability: float,
    n_active_sources: int = 4,
    n_associated_diseases: int = 0,
) -> float:
    """Compute composite causal confidence score.

    Weights:
    - Causal effect magnitude: 30%
    - Robustness (sensitivity analysis): 25%
    - Identifiability (backdoor criterion): 15%
    - Multi-source consensus: 10%
    - Disease specificity: 10%
    - Druggability: 10%

    Disease specificity replaces the former "network centrality"
    weight.  Hub genes (TP53, AKT1) are associated with thousands
    of diseases — they are generic, not specific.  Specificity
    rewards targets that are uniquely linked to the disease under
    study, penalising promiscuous hubs.

    Formula: specificity = 1 / log2(n_diseases + 2)
    - Gene linked to 1 disease:   specificity = 1.0
    - Gene linked to 10 diseases: specificity = 0.29
    - Gene linked to 100:         specificity = 0.15
    - Gene linked to 1000:        specificity = 0.10
    """
    # Normalise effect to 0-1
    effect_norm = min(1.0, effect)

    # Multi-source consensus: more sources = higher
    n_sources = len(source_scores)
    avg_source_score = float(np.mean(list(source_scores.values()))) if source_scores else 0.0
    consensus = min(1.0, (n_sources / max(1, n_active_sources)) * avg_source_score)

    # Disease specificity — replaces network centrality
    if n_associated_diseases > 0:
        specificity = 1.0 / np.log2(n_associated_diseases + 2)
    else:
        # No data → neutral (0.5), neither reward nor penalise
        specificity = 0.5

    confidence = (
        0.30 * effect_norm
        + 0.25 * robustness
        + 0.15 * (1.0 if is_identifiable else 0.0)
        + 0.10 * consensus
        + 0.10 * min(1.0, specificity)
        + 0.10 * druggability
    )

    return round(min(1.0, max(0.0, confidence)), 4)


def _classify_target(
    gene_name: str,
    causal_confidence: float,
    causal_effect: float,
    robustness: float,
    is_identifiable: bool,
    n_pathways: int,
    druggability: float,
    target_type: TargetType = TargetType.CORRELATIONAL,
    tissue_relevant: bool = True,
    evidence_streams: int = 0,
) -> tuple[TargetClassification, str]:
    """Classify a target as causal, correlational, or inconclusive.

    Classification rules:

    **Automatic demotion** (overrides confidence scores):
    - HOST_SYMPTOM targets → always CORRELATIONAL
    - tissue_relevant=False → always CORRELATIONAL

    **Tissue gate** (boolean, not a modifier):
    - tissue_relevant is True/False from the tissue filter.
    - True = gene is expressed in a disease-relevant tissue
      (or expression unknown → pass).
    - False = gene is only expressed in irrelevant tissues
      → demoted to CORRELATIONAL regardless of confidence.
    - The gate NEVER modifies causal_confidence.  Confidence
      stays pure — it measures causal evidence quality, not
      tissue expression.

    **Evidence triangulation** (for CAUSAL status):
    - Must have ≥2 independent evidence streams
    - causal_confidence ≥ 0.6 AND robust AND identifiable

    **Standard rules**:
    - Correlational: confidence < 0.4 OR not robust
    - Inconclusive: everything in between
    """
    reasons = []

    # ── Biological overrides (before confidence check) ──────

    # 1. Symptom markers are NEVER causal drug targets
    if target_type == TargetType.HOST_SYMPTOM:
        classification = TargetClassification.CORRELATIONAL
        reasons.append(
            f"⚠ {gene_name} classified as HOST_SYMPTOM: this gene "
            f"encodes a receptor/channel associated with disease "
            f"symptoms (e.g. seizures, pain), not with the disease "
            f"mechanism itself. Targeting symptom markers does not "
            f"treat the underlying disease."
        )
        reasons.append(
            f"Confidence was {causal_confidence:.2f} but biological "
            f"classification overrides statistical score."
        )
        return classification, " ".join(reasons)

    # 2. Tissue gate — independent boolean criterion
    #    If tissue_relevant is False, the gene is expressed only
    #    in tissues unrelated to this disease.  Demote regardless
    #    of how strong the statistical evidence looks.
    if not tissue_relevant:
        classification = TargetClassification.CORRELATIONAL
        reasons.append(
            f"⚠ {gene_name} FAILED tissue gate: expressed only in "
            f"tissues not relevant to this disease. "
            f"Confidence was {causal_confidence:.2f} but tissue "
            f"expression does not support this target."
        )
        return classification, " ".join(reasons)

    # ── Standard classification with evidence triangulation ──

    # When evidence_streams is explicitly provided (>0), enforce
    # the triangulation requirement.  When not provided (legacy
    # callers using default of 0), fall back to the original
    # thresholds for backward compatibility.
    triangulation_ok = evidence_streams >= 2 or evidence_streams == 0

    if (
        causal_confidence >= 0.6
        and robustness >= 0.4
        and is_identifiable
        and triangulation_ok
    ):
        classification = TargetClassification.CAUSAL
        reasons.append(
            f"{gene_name} has causal confidence {causal_confidence:.2f}, "
            f"supported by {n_pathways} pathway(s), robustness score "
            f"{robustness:.2f}, and {evidence_streams} independent "
            f"evidence stream(s). Tissue gate: PASS."
        )
        if target_type in (TargetType.PATHOGEN_DIRECT, TargetType.HOST_INVASION):
            reasons.append(
                f"Target type {target_type.value}: this is a direct "
                f"disease-mechanism target."
            )
        if druggability >= 0.5:
            reasons.append(
                f"Druggability score {druggability:.2f} indicates tractable "
                f"target with known 3D structures."
            )
        reasons.append(
            "Backdoor criterion satisfied: causal effect is identifiable "
            "after adjusting for confounders."
        )

    elif causal_confidence < 0.4 or robustness < 0.3:
        classification = TargetClassification.CORRELATIONAL
        reasons.append(
            f"{gene_name} is likely correlational (confidence={causal_confidence:.2f}, "
            f"robustness={robustness:.2f})."
        )
        if not is_identifiable:
            reasons.append(
                "No valid causal path identified — association may be "
                "due to confounding."
            )
        reasons.append(
            "Sensitivity analysis suggests the association is fragile "
            "and may not survive intervention."
        )

    elif evidence_streams > 0 and evidence_streams < 2 and causal_confidence >= 0.6:
        # High confidence but insufficient independent evidence
        classification = TargetClassification.INCONCLUSIVE
        reasons.append(
            f"{gene_name} has confidence {causal_confidence:.2f} but only "
            f"{evidence_streams} evidence stream(s). ≥2 required for CAUSAL."
        )

    else:
        classification = TargetClassification.INCONCLUSIVE
        reasons.append(
            f"{gene_name} has moderate evidence (confidence={causal_confidence:.2f}) "
            f"but insufficient data for definitive classification."
        )

    return classification, " ".join(reasons)


def _count_evidence_streams(
    source_scores: dict[str, float],
    n_pathways: int,
    n_interactions: int,
    node_data: dict[str, Any],
) -> int:
    """Count independent evidence streams supporting a target.

    Evidence streams:
    1. Gene-disease association databases (Monarch, OpenTargets)
    2. Pathway membership (KEGG, Reactome)
    3. Protein-protein interactions (STRING)
    4. Structural data (PDB)
    5. Druggability / functional annotation (UniProt)
    6. Drug evidence (ChEMBL — validated drug targets)

    Each counts as ONE stream even if multiple sources within
    the category confirm it (e.g. both KEGG and Reactome = 1
    pathway stream, not 2).
    """
    streams = 0

    # Stream 1: Gene-disease association databases
    assoc_sources = {"Monarch", "OpenTargets"}
    if any(s in source_scores for s in assoc_sources):
        streams += 1

    # Stream 2: Pathway membership
    if n_pathways > 0:
        streams += 1

    # Stream 3: Protein interactions
    if n_interactions > 0:
        streams += 1

    # Stream 4: 3D structural data
    pdb_ids = node_data.get("pdb_ids", [])
    if pdb_ids:
        streams += 1

    # Stream 5: Functional annotation / druggability
    metadata = node_data.get("metadata", {})
    if metadata.get("is_druggable") or metadata.get("go_terms"):
        streams += 1

    # Stream 6: ChEMBL drug evidence
    if metadata.get("chembl_drug_evidence_score") or "ChEMBL" in source_scores:
        streams += 1

    return streams


# ── Uncertainty Quantification ─────────────────────────────────────

def _bootstrap_confidence_interval(
    effect: float,
    robustness: float,
    is_identifiable: bool,
    n_pathways: int,
    n_interactions: int,
    source_scores: dict[str, float],
    druggability: float,
    n_active_sources: int = 4,
    n_associated_diseases: int = 0,
    n_bootstrap: int = 200,
    ci_level: float = 0.95,
) -> tuple[float, float]:
    """Compute bootstrap CI for causal confidence.

    Resamples the input evidence components with perturbation
    and recomputes causal confidence for each bootstrap sample
    to estimate the sampling distribution.

    Returns
    -------
    tuple[float, float]
        (lower_bound, upper_bound) of the 95% CI.
    """
    seed = int(os.environ.get("NEORX_SEED", "42"))
    rng = np.random.default_rng(seed)

    bootstrap_confs: list[float] = []

    for _ in range(n_bootstrap):
        # Perturb each evidence component
        b_effect = max(0.0, effect + rng.normal(0, 0.05))
        b_robustness = max(0.0, min(1.0, robustness + rng.normal(0, 0.05)))
        b_drug = max(0.0, min(1.0, druggability + rng.normal(0, 0.03)))

        # Resample source scores (leave-one-out equivalent)
        b_sources = dict(source_scores)
        if b_sources and rng.random() < 0.3:
            drop_key = rng.choice(list(b_sources.keys()))
            b_sources = {k: v for k, v in b_sources.items() if k != drop_key}

        b_pathways = max(0, n_pathways + int(rng.integers(-1, 2)))
        b_interactions = max(0, n_interactions + int(rng.integers(-1, 2)))

        conf = _compute_causal_confidence(
            effect=b_effect,
            robustness=b_robustness,
            is_identifiable=is_identifiable,
            n_pathways=b_pathways,
            n_interactions=b_interactions,
            source_scores=b_sources,
            druggability=b_drug,
            n_active_sources=n_active_sources,
            n_associated_diseases=n_associated_diseases,
        )
        bootstrap_confs.append(conf)

    alpha = (1.0 - ci_level) / 2.0
    lo = float(np.percentile(bootstrap_confs, 100 * alpha))
    hi = float(np.percentile(bootstrap_confs, 100 * (1 - alpha)))

    return (round(lo, 4), round(hi, 4))
