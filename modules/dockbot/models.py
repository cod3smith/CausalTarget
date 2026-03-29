"""
DockBot Data Models
====================

Pydantic models for molecular docking data.

All models are JSON-serialisable for seamless integration with the FastAPI
layer and downstream pipeline modules (GenMol, CausalTarget).

Key biology concepts
--------------------
* **Binding affinity** is reported in kcal/mol.  More negative = tighter
  binding.  A compound with -9 kcal/mol binds ~1000x tighter than one
  at -5 kcal/mol (exponential via Boltzmann: dG = -RT ln Kd).

* **Binding pocket** (a.k.a. binding site) is a cavity on the protein
  surface where a drug molecule fits.  Shape complementarity and
  electrostatic matching determine whether a molecule binds.

* **RMSD** (Root-Mean-Square Deviation) measures how far a docked pose
  deviates from a reference.  Lower RMSD = more reproducible pose.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ======================================================================
# Enums
# ======================================================================


class BindingSiteMethod(str, Enum):
    """How the binding site was determined."""

    LIGAND = "ligand"
    FPOCKET = "fpocket"
    MANUAL = "manual"


class JobStatus(str, Enum):
    """Status of an asynchronous screening job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


# ======================================================================
# Core domain models
# ======================================================================


class BindingSite(BaseModel):
    """Defines the docking search space -- a rectangular box centred on
    the protein binding pocket.

    AutoDock Vina searches for low-energy ligand poses *within* this box.
    A typical box is 20-30 A on each side.
    """

    center_x: float = Field(..., description="X coordinate of box centre (A).")
    center_y: float = Field(..., description="Y coordinate of box centre (A).")
    center_z: float = Field(..., description="Z coordinate of box centre (A).")
    size_x: float = Field(default=20.0, description="Box width along X (A).")
    size_y: float = Field(default=20.0, description="Box width along Y (A).")
    size_z: float = Field(default=20.0, description="Box width along Z (A).")
    method: BindingSiteMethod = Field(
        ..., description="Detection method used to define this site."
    )
    confidence: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Confidence in site definition (0-1).",
    )
    residues: list[str] = Field(
        default_factory=list,
        description="Key residues lining the pocket.",
    )


class DockingPose(BaseModel):
    """A single docked pose returned by Vina.

    Vina typically returns 1-20 poses ranked by binding affinity.
    The best pose (rank 1) has the most negative affinity.
    """

    rank: int = Field(..., ge=1, description="Pose rank (1 = best).")
    affinity_kcal_mol: float = Field(
        default=0.0,
        description="Predicted binding affinity in kcal/mol.",
    )
    rmsd_lb: float = Field(
        default=0.0, description="RMSD lower bound from best pose (A)."
    )
    rmsd_ub: float = Field(
        default=0.0, description="RMSD upper bound from best pose (A)."
    )
    pdbqt: str = Field(
        default="", description="PDBQT-format string of this pose."
    )


class DockingResult(BaseModel):
    """Full docking output for a single molecule against a target."""

    ligand_name: str = Field(default="", description="Human-readable molecule name.")
    ligand_smiles: str = Field(default="", description="Canonical SMILES of the ligand.")
    poses: list[DockingPose] = Field(
        default_factory=list,
        description="All returned poses, sorted by affinity (best first).",
    )
    receptor_path: str = Field(default="", description="Path to receptor PDBQT used.")
    binding_site: BindingSite | None = Field(
        default=None, description="The binding site used for docking."
    )


class ScreeningResult(BaseModel):
    """Aggregated results from a virtual-screening campaign."""

    results: list[DockingResult] = Field(
        default_factory=list,
        description="Individual docking results sorted by best affinity.",
    )
    total_ligands: int = Field(default=0, description="Total ligands submitted.")
    completed: int = Field(default=0, description="Number successfully docked.")
    elapsed_seconds: float = Field(default=0.0, description="Wall-clock runtime.")


class ProteinInfo(BaseModel):
    """Metadata about a prepared protein target."""

    pdb_id: str = Field(..., description="4-character PDB accession code.")
    name: str = Field(default="", description="Protein name / title from PDB.")
    resolution: float = Field(default=0.0, description="Crystallographic resolution (A).")
    pdbqt_path: str = Field(default="", description="Path to prepared PDBQT file.")
    binding_site: BindingSite | None = Field(
        default=None, description="Detected or specified binding site."
    )
    num_residues: int = Field(default=0, description="Number of amino acid residues.")
    removed_ligands: list[str] = Field(
        default_factory=list,
        description="Heteroatom residue names removed during preparation.",
    )


class ScreeningJob(BaseModel):
    """Tracks the state of an asynchronous screening run."""

    job_id: str = Field(..., description="Unique job identifier (UUID).")
    target_pdb: str = Field(..., description="PDB ID of the target protein.")
    binding_site: BindingSite | None = Field(default=None)
    total_molecules: int = Field(default=0)
    completed: int = Field(default=0)
    status: JobStatus = Field(default=JobStatus.PENDING)
    results: list[DockingResult] = Field(default_factory=list)
    error_message: str = Field(default="")


# ======================================================================
# API request / response models
# ======================================================================


class DockRequest(BaseModel):
    """Request to dock a single molecule."""

    smiles: str = Field(..., description="SMILES string of the ligand.")
    name: str = Field(default="ligand", description="Optional molecule name.")
    receptor_pdbqt: str = Field(
        ..., description="PDB ID or path to a prepared PDBQT receptor file."
    )
    binding_site: BindingSite | None = Field(
        default=None,
        description="Binding site. Auto-detected from co-crystal ligand if omitted.",
    )
    exhaustiveness: int | None = Field(default=32, ge=1, le=128)
    n_poses: int | None = Field(default=9, ge=1, le=20)


class ScreenRequest(BaseModel):
    """Request to screen a library of molecules."""

    receptor_pdbqt: str = Field(
        ..., description="PDB ID or path to a prepared PDBQT receptor file."
    )
    ligands: list[tuple[str, str]] = Field(
        ..., min_length=1, description="List of (smiles, name) tuples to screen."
    )
    binding_site: BindingSite | None = Field(
        default=None, description="Binding site. Auto-detected if omitted."
    )
    exhaustiveness: int | None = Field(default=16, ge=1, le=128)
    n_poses: int | None = Field(default=5, ge=1, le=20)
    workers: int = Field(default=0, ge=0, le=32, description="Parallel workers (0=auto).")


class BindingSiteRequest(BaseModel):
    """Request to detect binding site for a protein."""

    pdb_id: str = Field(..., description="4-character PDB ID.")
    method: BindingSiteMethod = Field(default=BindingSiteMethod.LIGAND)
    padding: float = Field(default=10.0, ge=0.0, description="Padding around ligand (A).")
