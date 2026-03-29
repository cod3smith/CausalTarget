"""
CausalTarget — AI-Driven Causal Drug Target Discovery
=======================================================

The capstone integration module of the CausalTarget platform.
Given a disease name, CausalTarget:

1. Builds a causal knowledge graph from 7 biomedical databases
2. Identifies genuine causal targets using Pearl's do-calculus
3. Generates novel drug candidates with GenMol (VAE)
4. Screens candidates for drug-likeness (MolScreen) and binding (DockBot)
5. Ranks by composite score with causal confidence weighted highest
6. Generates an interactive HTML report

Quick start
-----------
>>> from modules.causal_target import run_pipeline, build_disease_graph
>>> from modules.causal_target import identify_causal_targets
>>>
>>> # Full pipeline
>>> result = run_pipeline("HIV", top_n_targets=5)
>>> print(result.n_causal_targets)
>>>
>>> # Just the causal graph
>>> graph = build_disease_graph("HIV")
>>> print(graph.n_genes, "genes,", graph.n_pathways, "pathways")
>>>
>>> # Just causal target identification
>>> targets = identify_causal_targets(graph, top_n=10)
>>> for t in targets:
...     print(t.gene_name, t.classification.value, t.causal_confidence)
"""

from .models import (
    # Enums
    NodeType,
    EdgeType,
    JobStatus,
    TargetClassification,
    # Graph models
    GraphNode,
    GraphEdge,
    DiseaseGraph,
    # Analysis models
    CausalTargetResult,
    ScoredCandidate,
    # Pipeline models
    PipelineJob,
    PipelineResult,
    # API models
    RunRequest,
    GraphRequest,
    IdentifyRequest,
    ScreenTargetRequest,
    StatusResponse,
)
from .graph_builder import build_disease_graph, disease_graph_to_networkx
from .identifier import identify_causal_targets
from .scorer import score_candidate, rank_candidates, normalise_affinity, normalise_sa
from .pipeline import run_pipeline
from .report import generate_report

__all__ = [
    # Enums
    "NodeType",
    "EdgeType",
    "JobStatus",
    "TargetClassification",
    # Graph models
    "GraphNode",
    "GraphEdge",
    "DiseaseGraph",
    # Analysis models
    "CausalTargetResult",
    "ScoredCandidate",
    # Pipeline models
    "PipelineJob",
    "PipelineResult",
    # API models
    "RunRequest",
    "GraphRequest",
    "IdentifyRequest",
    "ScreenTargetRequest",
    "StatusResponse",
    # Core functions
    "build_disease_graph",
    "disease_graph_to_networkx",
    "identify_causal_targets",
    "score_candidate",
    "rank_candidates",
    "normalise_affinity",
    "normalise_sa",
    "run_pipeline",
    "generate_report",
]
