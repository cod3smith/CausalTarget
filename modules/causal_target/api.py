"""
CausalTarget FastAPI Service
==============================

REST API for the causal drug-discovery pipeline.

Endpoints
---------
- ``POST /run``               — Start a full pipeline run
- ``GET  /status/{job_id}``   — Check job status
- ``GET  /report/{job_id}``   — Download HTML report
- ``POST /graph``             — Build disease graph only
- ``POST /identify``          — Identify causal targets only
- ``POST /screen-target``     — Screen candidates for one target
- ``GET  /health``            — Health check
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .models import (
    RunRequest,
    GraphRequest,
    IdentifyRequest,
    ScreenTargetRequest,
    StatusResponse,
    JobStatus,
    PipelineResult,
    DiseaseGraph,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="CausalTarget API",
    description=(
        "AI-driven causal drug target discovery. "
        "Applies Pearl's do-calculus to identify genuine "
        "causal drug targets and rank novel candidates."
    ),
    version="0.1.0",
)

# In-memory job store (production would use Redis/DB)
_jobs: dict[str, PipelineResult] = {}
_graphs: dict[str, DiseaseGraph] = {}


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check."""
    return {"status": "ok", "service": "CausalTarget"}


@app.post("/run", response_model=StatusResponse)
async def run_pipeline_endpoint(request: RunRequest) -> StatusResponse:
    """Start a full CausalTarget pipeline run.

    This runs synchronously for simplicity.  In production,
    use a task queue (Celery, RQ) for async execution.
    """
    from .pipeline import run_pipeline

    logger.info("Pipeline run requested for '%s'.", request.disease)

    result = run_pipeline(
        disease=request.disease,
        top_n_targets=request.top_n_targets,
        candidates_per_target=request.candidates_per_target,
        generate_molecules=True,
        run_docking=False,  # Disable docking in API by default
    )

    _jobs[result.job.job_id] = result

    return StatusResponse(
        job_id=result.job.job_id,
        status=result.job.status,
        progress_pct=result.job.progress_pct,
        current_step=result.job.current_step,
        error=result.job.error,
    )


@app.get("/status/{job_id}", response_model=StatusResponse)
async def get_status(job_id: str) -> StatusResponse:
    """Get pipeline job status."""
    result = _jobs.get(job_id)
    if not result:
        raise HTTPException(404, f"Job {job_id} not found.")

    return StatusResponse(
        job_id=result.job.job_id,
        status=result.job.status,
        progress_pct=result.job.progress_pct,
        current_step=result.job.current_step,
        error=result.job.error,
    )


@app.get("/report/{job_id}", response_class=HTMLResponse)
async def get_report(job_id: str) -> HTMLResponse:
    """Download the HTML report for a completed job."""
    result = _jobs.get(job_id)
    if not result:
        raise HTTPException(404, f"Job {job_id} not found.")
    if result.job.status != JobStatus.COMPLETE:
        raise HTTPException(400, f"Job {job_id} is not complete ({result.job.status.value}).")
    if not result.report_html:
        raise HTTPException(404, "Report not generated for this job.")

    return HTMLResponse(content=result.report_html)


@app.post("/graph")
async def build_graph_endpoint(request: GraphRequest) -> dict[str, Any]:
    """Build a disease causal knowledge graph."""
    from .graph_builder import build_disease_graph

    graph = build_disease_graph(request.disease)
    _graphs[request.disease.lower()] = graph

    return {
        "disease": graph.disease_name,
        "n_nodes": len(graph.nodes),
        "n_edges": len(graph.edges),
        "n_genes": graph.n_genes,
        "n_proteins": graph.n_proteins,
        "n_pathways": graph.n_pathways,
        "sources": graph.sources_queried,
    }


@app.post("/identify")
async def identify_targets_endpoint(request: IdentifyRequest) -> dict[str, Any]:
    """Identify causal drug targets for a disease."""
    from .graph_builder import build_disease_graph
    from .identifier import identify_causal_targets

    graph = build_disease_graph(request.disease)
    targets = identify_causal_targets(graph, top_n=request.top_n)

    return {
        "disease": request.disease,
        "n_evaluated": len(targets),
        "n_causal": sum(1 for t in targets if t.is_causal_target),
        "targets": [
            {
                "gene": t.gene_name,
                "protein": t.protein_name,
                "classification": t.classification.value,
                "causal_confidence": t.causal_confidence,
                "robustness": t.robustness_score,
                "druggability": t.druggability_score,
                "reasoning": t.reasoning,
            }
            for t in targets
        ],
    }


@app.post("/screen-target")
async def screen_target_endpoint(request: ScreenTargetRequest) -> dict[str, Any]:
    """Screen generated candidates against a specific target."""
    from .models import CausalTargetResult
    from .pipeline import _generate_for_target, _screen_candidates
    from .scorer import rank_candidates

    # Build a minimal CausalTargetResult for the target
    target = CausalTargetResult(
        protein_id=request.target_uniprot_id,
        protein_name=request.target_uniprot_id,
        gene_name=request.target_uniprot_id,
        uniprot_id=request.target_uniprot_id,
        pdb_ids=[request.target_pdb_id],
        causal_confidence=request.causal_confidence,
        is_causal_target=True,
    )

    smiles_list = _generate_for_target(target, request.n_candidates)
    scored = _screen_candidates(smiles_list, target, run_docking=False)
    ranked = rank_candidates(scored)

    return {
        "target": request.target_uniprot_id,
        "n_candidates": len(ranked),
        "top_10": [
            {
                "rank": c.rank,
                "smiles": c.smiles,
                "composite_score": c.composite_score,
                "qed": c.qed_score,
                "is_drug_like": c.is_drug_like,
            }
            for c in ranked[:10]
        ],
    }
