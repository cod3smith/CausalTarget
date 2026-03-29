"""
DockBot REST API
==================

FastAPI endpoints for programmatic access to the docking pipeline.

Endpoints
---------
- ``POST /dock``       — dock a single ligand
- ``POST /screen``     — submit a screening job
- ``GET  /protein/{pdb_id}`` — fetch & prepare a protein
- ``GET  /results/{job_id}`` — poll screening status / results
- ``GET  /health``     — service health check

All endpoints return JSON.  Docking and screening are CPU-intensive
so they run synchronously in the request thread (for small jobs) or
should be put behind a task queue (Celery / ARQ) in production.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
except ImportError:
    raise ImportError(
        "DockBot API requires FastAPI.  Install it with: pip install fastapi uvicorn"
    )

from .models import (
    BindingSite,
    BindingSiteMethod,
    DockingResult,
    DockRequest,
    ScreenRequest,
    ScreeningResult,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="DockBot API",
    description="Automated molecular docking pipeline.",
    version="0.1.0",
)

# Simple in-memory job store (swap for Redis/DB in production)
_jobs: dict[str, dict] = {}

OUTPUT_DIR = Path("docking_api_output")


@app.get("/health")
async def health():
    """Service health check."""
    return {"status": "ok", "service": "dockbot"}


# ── Protein preparation ────────────────────────────────────────────

@app.get("/protein/{pdb_id}")
async def prepare_protein_endpoint(pdb_id: str):
    """Fetch and prepare a protein for docking.

    Returns paths to the cleaned PDB and PDBQT files plus metadata.
    """
    from .protein_prep import prepare_protein

    try:
        result = prepare_protein(pdb_id, output_dir=OUTPUT_DIR)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Binding site ────────────────────────────────────────────────────

@app.post("/binding-site")
async def detect_binding_site_endpoint(
    pdb_path: str,
    method: str = "ligand",
    ligand_resname: str | None = None,
    pocket_rank: int = 1,
    center_x: float | None = None,
    center_y: float | None = None,
    center_z: float | None = None,
    size_x: float = 25.0,
    size_y: float = 25.0,
    size_z: float = 25.0,
    padding: float = 10.0,
):
    """Detect or specify a binding site."""
    from .binding_site import detect_binding_site

    center = None
    if center_x is not None and center_y is not None and center_z is not None:
        center = (center_x, center_y, center_z)

    try:
        site = detect_binding_site(
            pdb_path,
            method=method,
            ligand_resname=ligand_resname,
            pocket_rank=pocket_rank,
            center=center,
            size=(size_x, size_y, size_z),
            padding=padding,
        )
        return site.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── Single docking ─────────────────────────────────────────────────

@app.post("/dock")
async def dock_endpoint(request: DockRequest):
    """Dock a single ligand against a protein.

    The receptor must have been prepared first (via ``/protein/{pdb_id}``).
    """
    from .binding_site import detect_binding_site
    from .docker import VinaConfig, dock
    from .ligand_prep import prepare_ligand_pdbqt
    from .protein_prep import prepare_protein

    try:
        # Prepare receptor if needed
        receptor_pdbqt = Path(request.receptor_pdbqt)
        if not receptor_pdbqt.exists():
            prep = prepare_protein(request.receptor_pdbqt, output_dir=OUTPUT_DIR)
            receptor_pdbqt = Path(prep["pdbqt"])
            clean_pdb = Path(prep["clean_pdb"])
        else:
            clean_pdb = receptor_pdbqt.with_suffix(".pdb")

        # Binding site
        if request.binding_site:
            site = request.binding_site
        else:
            site = detect_binding_site(clean_pdb, method="ligand")

        # Prepare ligand
        mol, lig_pdbqt = prepare_ligand_pdbqt(
            request.smiles, name=request.name
        )
        if mol is None:
            raise HTTPException(status_code=400, detail="Ligand preparation failed.")

        # Dock
        config = VinaConfig(
            exhaustiveness=request.exhaustiveness or 32,
            n_poses=request.n_poses or 9,
        )
        result = dock(
            receptor_pdbqt=receptor_pdbqt,
            ligand_pdbqt=lig_pdbqt,
            site=site,
            config=config,
            ligand_name=request.name,
            ligand_smiles=request.smiles,
        )

        return result.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Screening ──────────────────────────────────────────────────────

@app.post("/screen")
async def screen_endpoint(request: ScreenRequest):
    """Submit a virtual screening job.

    Returns a job ID that can be polled via ``/results/{job_id}``.

    Note: in this basic implementation, screening runs synchronously.
    For production, integrate with a task queue.
    """
    from .binding_site import detect_binding_site
    from .docker import VinaConfig
    from .parallel import ScreeningJob, run_screening
    from .protein_prep import prepare_protein

    job_id = str(uuid.uuid4())[:8]

    try:
        # Prepare receptor
        receptor_pdbqt = Path(request.receptor_pdbqt)
        if not receptor_pdbqt.exists():
            prep = prepare_protein(request.receptor_pdbqt, output_dir=OUTPUT_DIR)
            receptor_pdbqt = Path(prep["pdbqt"])
            clean_pdb = Path(prep["clean_pdb"])
        else:
            clean_pdb = receptor_pdbqt.with_suffix(".pdb")

        # Binding site
        if request.binding_site:
            site = request.binding_site
        else:
            site = detect_binding_site(clean_pdb, method="ligand")

        # Build ligand list
        ligands = [(smi, name) for smi, name in request.ligands]

        job = ScreeningJob(
            receptor_pdbqt=receptor_pdbqt,
            binding_site=site,
            ligands=ligands,
            output_dir=OUTPUT_DIR / job_id,
            config=VinaConfig(exhaustiveness=request.exhaustiveness or 16),
        )

        result = run_screening(job)

        _jobs[job_id] = {
            "status": "completed",
            "result": result,
        }

        return {
            "job_id": job_id,
            "status": "completed",
            "total_ligands": result.total_ligands,
            "completed": result.completed,
            "elapsed_seconds": result.elapsed_seconds,
        }

    except Exception as e:
        _jobs[job_id] = {"status": "failed", "error": str(e)}
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/{job_id}")
async def get_results(job_id: str):
    """Get results for a screening job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    job_data = _jobs[job_id]

    if job_data["status"] == "failed":
        raise HTTPException(status_code=500, detail=job_data.get("error", "Unknown error"))

    if job_data["status"] == "completed":
        result: ScreeningResult = job_data["result"]
        return {
            "job_id": job_id,
            "status": "completed",
            "total_ligands": result.total_ligands,
            "completed": result.completed,
            "elapsed_seconds": result.elapsed_seconds,
            "results": [
                {
                    "name": r.ligand_name,
                    "smiles": r.ligand_smiles,
                    "best_affinity": r.poses[0].affinity_kcal_mol if r.poses else None,
                    "n_poses": len(r.poses),
                }
                for r in result.results[:50]
            ],
        }

    return {"job_id": job_id, "status": job_data["status"]}


@app.get("/report/{job_id}", response_class=HTMLResponse)
async def get_report(job_id: str):
    """Get an HTML report for a completed screening job."""
    if job_id not in _jobs or _jobs[job_id]["status"] != "completed":
        raise HTTPException(status_code=404, detail="Report not available.")

    from .report import generate_report

    result = _jobs[job_id]["result"]
    report_path = OUTPUT_DIR / job_id / "report.html"

    generate_report(
        result,
        output_path=report_path,
        protein_name=job_id,
    )

    return report_path.read_text()
