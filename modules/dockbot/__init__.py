"""
DockBot — Automated Molecular Docking Pipeline
=================================================

DockBot automates the full molecular docking workflow:

1. **Protein preparation** — fetch from PDB, clean, convert to PDBQT.
2. **Ligand preparation** — SMILES → 3D → MMFF94 minimise → PDBQT.
3. **Binding-site detection** — co-crystal ligand / fpocket / manual.
4. **Docking** — AutoDock Vina (Python bindings or CLI).
5. **Parallel screening** — multiprocessing with resume capability.
6. **Composite scoring** — Vina affinity + QED + SA + filter pass-rate.
7. **Visualisation** — py3Dmol 3D viewers + matplotlib summary charts.
8. **Reporting** — self-contained HTML reports with ranked tables.

Quick start::

    from modules.dockbot import prepare_protein, prepare_ligand, dock

    protein = prepare_protein("1BNA")
    mol, pdbqt = prepare_ligand_pdbqt("CC(=O)Oc1ccccc1C(O)=O", name="aspirin")
    result = dock(protein["pdbqt"], pdbqt, site)
"""

from .models import (
    BindingSite,
    BindingSiteMethod,
    BindingSiteRequest,
    DockingPose,
    DockingResult,
    DockRequest,
    JobStatus,
    ProteinInfo,
    ScreeningJob,
    ScreeningResult,
    ScreenRequest,
)

__all__ = [
    # Models
    "BindingSite",
    "BindingSiteMethod",
    "BindingSiteRequest",
    "DockingPose",
    "DockingResult",
    "DockRequest",
    "JobStatus",
    "ProteinInfo",
    "ScreeningJob",
    "ScreeningResult",
    "ScreenRequest",
    # Submodules (lazy access via modules.dockbot.<submodule>)
    "protein_prep",
    "ligand_prep",
    "binding_site",
    "docker",
    "parallel",
    "scorer",
    "viz",
    "report",
]
