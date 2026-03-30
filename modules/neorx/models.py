"""
NeoRx Data Models
=========================

Pydantic models for the causal drug-discovery pipeline.

The hierarchy:
1. ``GeneNode`` / ``ProteinNode`` / ``PathwayNode`` — graph entities
2. ``DiseaseGraph`` — the assembled causal knowledge graph
3. ``NeoRxResult`` — a protein evaluated as a potential target
4. ``ScoredCandidate`` — a drug candidate with composite score
5. ``PipelineResult`` — the full output of a pipeline run

Why causal reasoning matters
------------------------------
Most drug-discovery pipelines rank targets by **correlation** —
which genes are differentially expressed in disease tissue?  But
correlation ≠ causation:

* TNF-α is highly correlated with HIV progression, but targeting
  it does not treat HIV.  It is a *downstream consequence*, not a
  *cause*.
* HIV protease is causally upstream — inhibiting it blocks viral
  replication.  Protease inhibitors *work*.

NeoRx applies Pearl's do-calculus to distinguish genuine
causal targets from correlational bystanders.  The
``NeoRxResult.is_causal_target`` flag captures this
distinction, and the composite scorer weights causal confidence
above binding affinity.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Enums ───────────────────────────────────────────────────────────

class NodeType(str, Enum):
    """Types of nodes in the disease causal graph."""
    GENE = "gene"
    PROTEIN = "protein"
    PATHWAY = "pathway"
    PHENOTYPE = "phenotype"
    DISEASE = "disease"
    DRUG = "drug"
    METABOLITE = "metabolite"


class EdgeType(str, Enum):
    """Types of biological relationships (edges) in the graph."""
    ACTIVATES = "activates"
    INHIBITS = "inhibits"
    PHOSPHORYLATES = "phosphorylates"
    BINDS = "binds"
    REGULATES = "regulates"
    PARTICIPATES_IN = "participates_in"
    ASSOCIATED_WITH = "associated_with"
    CAUSES = "causes"
    TREATS = "treats"
    INTERACTS_WITH = "interacts_with"
    UPREGULATES = "upregulates"
    DOWNREGULATES = "downregulates"


class JobStatus(str, Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    BUILDING_GRAPH = "building_graph"
    IDENTIFYING_TARGETS = "identifying_targets"
    GENERATING_CANDIDATES = "generating_candidates"
    SCREENING = "screening"
    SCORING = "scoring"
    REPORTING = "reporting"
    COMPLETE = "complete"
    FAILED = "failed"


class TargetClassification(str, Enum):
    """Causal classification of a putative target."""
    CAUSAL = "causal"
    CORRELATIONAL = "correlational"
    INCONCLUSIVE = "inconclusive"


# ── Graph Nodes ─────────────────────────────────────────────────────

class GraphNode(BaseModel):
    """A node in the disease causal graph.

    Every entity (gene, protein, pathway, phenotype) is
    represented as a node with a unique ID, a human-readable
    name, and metadata from the source database.
    """
    node_id: str = Field(..., description="Unique identifier (e.g. ENSG00000145675)")
    name: str = Field(..., description="Human-readable name (e.g. PIK3CA)")
    node_type: NodeType = Field(..., description="Entity type")
    source: str = Field("", description="Database source (Monarch, OpenTargets, etc.)")
    score: float = Field(0.0, description="Association/confidence score from source", ge=0.0, le=1.0)
    uniprot_id: Optional[str] = Field(None, description="UniProt accession")
    pdb_ids: list[str] = Field(default_factory=list, description="Known PDB structures")
    description: Optional[str] = Field(None, description="Functional description")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional source metadata")


class GraphEdge(BaseModel):
    """An edge (relationship) in the disease causal graph.

    Each edge represents a biological relationship with a
    confidence score and provenance.
    """
    source_id: str = Field(..., description="Source node ID")
    target_id: str = Field(..., description="Target node ID")
    edge_type: EdgeType = Field(..., description="Relationship type")
    weight: float = Field(0.5, description="Confidence/evidence score", ge=0.0, le=1.0)
    source_db: str = Field("", description="Database this edge came from")
    evidence: Optional[str] = Field(None, description="Supporting evidence text")
    pmids: list[str] = Field(default_factory=list, description="PubMed IDs supporting this edge")


class DiseaseGraph(BaseModel):
    """A causal knowledge graph for a specific disease.

    Assembled from multiple databases (Monarch Initiative, Open Targets,
    KEGG, Reactome, STRING, UniProt).  The graph captures known
    biological relationships: which genes cause the disease,
    which proteins interact, which pathways are disrupted.

    The graph is stored both as:
    - A networkx ``DiGraph`` (for DoWhy compatibility)
    - A list of nodes and edges (for serialisation)
    """
    disease_name: str = Field(..., description="Disease being analysed")
    disease_id: Optional[str] = Field(None, description="Disease ontology ID (e.g. MONDO:0005109)")
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    sources_queried: list[str] = Field(default_factory=list, description="Which databases were queried")
    build_timestamp: datetime = Field(default_factory=datetime.now)

    @property
    def n_genes(self) -> int:
        return sum(1 for n in self.nodes if n.node_type == NodeType.GENE)

    @property
    def n_proteins(self) -> int:
        return sum(1 for n in self.nodes if n.node_type == NodeType.PROTEIN)

    @property
    def n_pathways(self) -> int:
        return sum(1 for n in self.nodes if n.node_type == NodeType.PATHWAY)


# ── Causal Analysis Results ────────────────────────────────────────

class NeoRxResult(BaseModel):
    """Result of causal analysis for a single protein target.

    This is the core output of the causal reasoning layer.  For
    each protein in the disease graph, we ask: "If we intervene
    on this protein (i.e. inhibit/activate it with a drug), does
    it causally affect the disease outcome?"

    We answer this using Pearl's do-calculus via DoWhy:

    1. **Identifiability** — Can we estimate the causal effect?
       The backdoor criterion checks if there is a valid set of
       variables to condition on that blocks all spurious paths.

    2. **Effect estimation** — What is the magnitude of the
       causal effect?  Larger effects = more promising targets.

    3. **Sensitivity analysis** — Is the estimate robust to
       unmeasured confounders?  If adding a random common cause
       changes the estimate, the original finding was fragile.

    4. **Classification** — Combine all evidence to label the
       protein as a genuine causal target or a correlational
       bystander.
    """
    protein_id: str = Field(..., description="Unique protein identifier")
    protein_name: str = Field(..., description="Human-readable protein name")
    gene_name: str = Field("", description="Gene symbol (e.g. TP53)")
    uniprot_id: str = Field("", description="UniProt accession (e.g. P04637)")
    pdb_ids: list[str] = Field(default_factory=list, description="Known PDB structures")

    # Causal analysis results
    causal_effect: float = Field(0.0, description="Estimated causal effect size")
    causal_confidence: float = Field(0.0, description="Combined confidence 0-1", ge=0.0, le=1.0)
    adjustment_set: list[str] = Field(default_factory=list, description="Variables adjusted for (backdoor)")
    causal_pathway: list[str] = Field(default_factory=list, description="Path from target to disease")
    robustness_score: float = Field(0.0, description="Sensitivity analysis robustness 0-1", ge=0.0, le=1.0)
    druggability_score: float = Field(0.0, description="Predicted druggability 0-1", ge=0.0, le=1.0)

    # Classification
    classification: TargetClassification = Field(TargetClassification.INCONCLUSIVE)
    is_causal_target: bool = Field(False, description="Final: is this a genuine causal target?")
    reasoning: str = Field("", description="Human-readable causal reasoning explanation")

    # Uncertainty quantification
    confidence_interval: tuple[float, float] = Field(
        (0.0, 1.0), description="95% bootstrap CI on causal_confidence",
    )

    # Evidence
    source_scores: dict[str, float] = Field(default_factory=dict, description="Per-database evidence scores")
    n_supporting_pathways: int = Field(0)
    n_protein_interactions: int = Field(0)

    # Biological classification (scientific-rigor layer)
    target_type: str = Field("", description="TargetType from classifier (e.g. HOST_SYMPTOM, PATHOGEN_DIRECT)")
    tissue_relevant: float = Field(1.0, description="Tissue relevance score 0.0–1.0 from HPA expression data")
    tissue_explanation: str = Field("", description="Why tissue is or is not relevant")
    counterfactual_effect: float | None = Field(None, description="Pearl's counterfactual ΔY")
    evidence_streams: int = Field(0, description="Number of independent evidence streams supporting this target")


# ── Drug Candidate Scoring ─────────────────────────────────────────

class ScoredCandidate(BaseModel):
    """A drug candidate scored across multiple dimensions.

    The composite score integrates:
    - **Causal confidence** (weight 0.30) — Is the target genuine?
    - **Binding affinity** (weight 0.25) — How tightly does it bind?
    - **Drug-likeness (QED)** (weight 0.15) — Is it drug-like?
    - **Synthetic accessibility** (weight 0.10) — Can it be made?
    - **ADMET** (weight 0.10) — Predicted pharmacokinetics
    - **Novelty** (weight 0.10) — Structural novelty vs known drugs

    The key design decision: causal confidence has the HIGHEST
    weight.  A moderately-binding molecule against a validated
    causal target ranks above a strongly-binding molecule against
    a correlational bystander.
    """
    smiles: str = Field(..., description="SMILES string")
    candidate_name: str = Field("", description="Generated name")

    # Target info
    target_protein_id: str = Field(..., description="Target this was designed for")
    target_protein_name: str = Field("", description="Target protein name")

    # Individual scores
    causal_confidence: float = Field(0.0, ge=0.0, le=1.0)
    binding_affinity: Optional[float] = Field(None, description="kcal/mol (negative = better)")
    qed_score: Optional[float] = Field(None, description="Quantitative estimate of drug-likeness")
    sa_score: Optional[float] = Field(None, description="Synthetic accessibility 1-10")
    admet_score: float = Field(0.5, description="Predicted ADMET score 0-1", ge=0.0, le=1.0)
    novelty_score: float = Field(0.5, description="Structural novelty vs known drugs 0-1", ge=0.0, le=1.0)

    # Composite
    composite_score: float = Field(0.0, description="Weighted composite score")
    score_breakdown: dict[str, float] = Field(default_factory=dict, description="Per-dimension normalised contributions")

    # Properties
    molecular_weight: Optional[float] = None
    logp: Optional[float] = None
    drug_likeness_class: str = Field("", description="Drug-likeness classification")
    n_filters_passed: int = Field(0)

    # Status flags
    is_drug_like: bool = Field(False)
    is_novel: bool = Field(False)
    rank: int = Field(0, description="Global rank across all targets")


# ── Pipeline Result ────────────────────────────────────────────────

class PipelineJob(BaseModel):
    """Tracks a pipeline execution job."""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    disease: str = Field(...)
    status: JobStatus = Field(JobStatus.PENDING)
    top_n_targets: int = Field(5, ge=1, le=50)
    candidates_per_target: int = Field(100, ge=10)
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    progress_pct: float = Field(0.0, ge=0.0, le=100.0)
    current_step: str = Field("")


class PipelineResult(BaseModel):
    """Complete output of a NeoRx pipeline run.

    Contains everything needed to understand why specific drug
    candidates were recommended: the disease graph, the causal
    reasoning for each target, and the scored candidates.
    """
    job: PipelineJob
    disease: str = Field(...)
    graph: Optional[DiseaseGraph] = None
    causal_targets: list[NeoRxResult] = Field(default_factory=list)
    scored_candidates: list[ScoredCandidate] = Field(default_factory=list)
    report_html: Optional[str] = Field(None, description="Generated HTML report")
    report_path: Optional[str] = Field(None, description="Path to saved report")
    validation: Optional[dict[str, Any]] = Field(None, description="Known-target validation results")

    @property
    def n_causal_targets(self) -> int:
        return sum(1 for t in self.causal_targets if t.is_causal_target)

    @property
    def top_candidates(self) -> list[ScoredCandidate]:
        """Top 10 candidates by composite score."""
        return sorted(self.scored_candidates, key=lambda c: c.composite_score, reverse=True)[:10]


# ── API Request/Response Models ────────────────────────────────────

class RunRequest(BaseModel):
    """Request to start a full pipeline run."""
    disease: str = Field(..., description="Disease name", min_length=1)
    top_n_targets: int = Field(5, ge=1, le=50)
    candidates_per_target: int = Field(1000, ge=10, le=50000)
    allow_mocks: bool = Field(
        False, description="Allow mock-data fallback when APIs fail",
    )


class GraphRequest(BaseModel):
    """Request to build a disease graph."""
    disease: str = Field(..., min_length=1)
    allow_mocks: bool = Field(
        False, description="Allow mock-data fallback when APIs fail",
    )


class IdentifyRequest(BaseModel):
    """Request to identify causal targets."""
    disease: str = Field(..., min_length=1)
    top_n: int = Field(10, ge=1, le=100)
    allow_mocks: bool = Field(
        False, description="Allow mock-data fallback when APIs fail",
    )


class ScreenTargetRequest(BaseModel):
    """Request to screen candidates against a specific target."""
    target_uniprot_id: str = Field(..., description="UniProt accession")
    target_pdb_id: str = Field(..., description="PDB ID for docking")
    n_candidates: int = Field(100, ge=10, le=10000)
    causal_confidence: float = Field(0.5, ge=0.0, le=1.0)


class StatusResponse(BaseModel):
    """Pipeline job status."""
    job_id: str
    status: JobStatus
    progress_pct: float
    current_step: str
    error: Optional[str] = None
