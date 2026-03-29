"""
GenMol FastAPI Service
=======================

REST API for the generative molecular design module.

Endpoints::

    POST /generate       — Generate molecules from the prior
    POST /generate/cond  — Conditional generation (CVAE)
    POST /interpolate    — Latent space interpolation
    POST /evaluate       — Compute generation metrics
    POST /screen         — Screen molecules through MolScreen
    GET  /health         — Service health check
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="GenMol API",
    description="Generative molecular design with VAE — NeoForge CausalTarget",
    version="0.1.0",
)

# ── Lazy-loaded model singleton ─────────────────────────────────────
_model = None
_tokenizer = None
_device = None


def _load_model(
    checkpoint_path: str = "checkpoints/genmol/best_model.pt",
    tokenizer_path: str = "checkpoints/genmol/tokenizer.json",
):
    """Lazily load the VAE model and tokenizer."""
    global _model, _tokenizer, _device

    if _model is not None:
        return

    import torch
    from .data.tokenizer import SmilesTokenizer
    from .models.vae import MolVAE

    _tokenizer = SmilesTokenizer.load(tokenizer_path)

    ckpt = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )

    _model = MolVAE(
        vocab_size=_tokenizer.vocab_size,
        max_length=_tokenizer.max_length,
        pad_idx=_tokenizer.pad_idx,
    )
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.eval()

    if torch.cuda.is_available():
        _device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")

    _model = _model.to(_device)
    logger.info("Model loaded on %s.", _device)


# ── Request / Response Models ───────────────────────────────────────


class GenerateRequest(BaseModel):
    """Request body for molecule generation."""

    n: int = Field(100, ge=1, le=10_000, description="Number of molecules.")
    temperature: float = Field(
        1.0, gt=0, le=3.0, description="Sampling temperature."
    )
    screen: bool = Field(False, description="Apply MolScreen filtering.")
    max_sa: float = Field(4.0, description="Max SA score for screening.")
    min_qed: float = Field(0.3, description="Min QED for screening.")


class ConditionalRequest(BaseModel):
    """Request for conditional generation."""

    n: int = Field(100, ge=1, le=10_000)
    temperature: float = Field(1.0, gt=0)
    mw: float = Field(350.0, description="Target molecular weight.")
    logp: float = Field(2.5, description="Target LogP.")
    qed: float = Field(0.7, description="Target QED.")


class InterpolateRequest(BaseModel):
    """Request for latent interpolation."""

    smiles_a: str = Field(..., description="Starting SMILES.")
    smiles_b: str = Field(..., description="Ending SMILES.")
    n_steps: int = Field(10, ge=2, le=50)


class EvaluateRequest(BaseModel):
    """Request for metric evaluation."""

    generated_smiles: list[str] = Field(..., description="Generated SMILES.")
    reference_smiles: Optional[list[str]] = Field(
        None, description="Reference SMILES for novelty."
    )


class ScreenRequest(BaseModel):
    """Request for MolScreen filtering."""

    smiles: list[str] = Field(..., description="SMILES to screen.")
    max_sa: float = Field(4.0)
    min_qed: float = Field(0.3)


class GenerateResponse(BaseModel):
    """Response from generation endpoints."""

    smiles: list[str]
    count: int
    temperature: float


class MetricsResponse(BaseModel):
    """Response from evaluation endpoint."""

    validity: float
    uniqueness: float
    novelty: float
    diversity: float
    n_generated: int
    n_valid: int
    n_unique: int


# ── Endpoints ───────────────────────────────────────────────────────


@app.get("/health")
def health_check():
    """Service health check."""
    return {"status": "ok", "service": "genmol"}


@app.post("/generate", response_model=GenerateResponse)
def generate_molecules(req: GenerateRequest):
    """Generate molecules by sampling from the VAE prior."""
    try:
        _load_model()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {e}",
        )

    from .generate import generate, screen_generated

    smiles = generate(
        _model,
        _tokenizer,
        n=req.n,
        temperature=req.temperature,
        device=_device,
    )

    if req.screen:
        results = screen_generated(
            smiles, max_sa_score=req.max_sa, min_qed=req.min_qed
        )
        smiles = [r["smiles"] for r in results]

    return GenerateResponse(
        smiles=smiles,
        count=len(smiles),
        temperature=req.temperature,
    )


@app.post("/interpolate")
def interpolate_molecules(req: InterpolateRequest):
    """Interpolate between two molecules in latent space."""
    try:
        _load_model()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    from .generate import interpolate

    smiles = interpolate(
        _model,
        _tokenizer,
        req.smiles_a,
        req.smiles_b,
        n_steps=req.n_steps,
        device=_device,
    )

    return {"smiles": smiles, "n_steps": len(smiles)}


@app.post("/evaluate", response_model=MetricsResponse)
def evaluate_generation(req: EvaluateRequest):
    """Compute generation quality metrics."""
    from .evaluation.metrics import compute_all_metrics

    metrics = compute_all_metrics(
        req.generated_smiles,
        req.reference_smiles,
    )
    return MetricsResponse(**metrics)


@app.post("/screen")
def screen_molecules(req: ScreenRequest):
    """Screen molecules through MolScreen."""
    from .generate import screen_generated

    results = screen_generated(
        req.smiles, max_sa_score=req.max_sa, min_qed=req.min_qed
    )
    return {"results": results, "passed": len(results), "total": len(req.smiles)}
