"""
DockBot Parallel Screening
=============================

Docks a library of ligands against a single protein target using
multiprocessing.

Virtual screening overview
--------------------------
In early-stage drug discovery we often need to dock thousands–millions
of compounds against a target to identify *hits* (molecules predicted
to bind strongly).  This is called **virtual screening** (VS).

Each docking run is independent — we can parallelise across CPU cores.
This module uses Python's ``multiprocessing`` pool for embarrassingly-
parallel execution, with ``tqdm`` progress bars for interactive
feedback.

Resume capability
-----------------
If a screening run is interrupted, previously completed results are
saved to a JSON-lines file.  Re-running the same job will skip already-
docked ligands automatically.
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .docker import VinaConfig, dock
from .ligand_prep import prepare_ligand_pdbqt
from .models import BindingSite, DockingResult, ScreeningResult

logger = logging.getLogger(__name__)


@dataclass
class ScreeningJob:
    """Definition of a virtual-screening job."""

    receptor_pdbqt: str | Path
    """Path to the receptor PDBQT file."""

    binding_site: BindingSite
    """Search box for docking."""

    ligands: list[tuple[str, str]]
    """List of ``(smiles, name)`` tuples to dock."""

    output_dir: Path = Path("screening_results")
    """Directory for result files."""

    config: VinaConfig | None = None
    """Vina run parameters."""

    n_workers: int = 0
    """Number of parallel workers (0 = number of CPUs)."""


def run_screening(job: ScreeningJob) -> ScreeningResult:
    """Execute a virtual screening campaign.

    Parameters
    ----------
    job:
        Screening job specification.

    Returns
    -------
    ScreeningResult
        Aggregated results sorted by best affinity.
    """
    job.output_dir.mkdir(parents=True, exist_ok=True)
    results_file = job.output_dir / "results.jsonl"

    # Load previously completed results (for resume)
    completed = _load_completed(results_file)
    logger.info(
        "Screening %d ligands (%d already completed).",
        len(job.ligands),
        len(completed),
    )

    # Filter out already-completed ligands
    pending = [
        (smi, name) for smi, name in job.ligands if name not in completed
    ]

    if not pending:
        logger.info("All ligands already docked — nothing to do.")
        return _finalise(completed, job)

    n_workers = job.n_workers or min(mp.cpu_count(), len(pending))
    config = job.config or VinaConfig()
    receptor_path = str(Path(job.receptor_pdbqt).resolve())

    # Build task arguments
    tasks = [
        (receptor_path, smi, name, job.binding_site, config)
        for smi, name in pending
    ]

    start = time.perf_counter()
    new_results: list[DockingResult] = []

    try:
        # Try to import tqdm for progress bars
        from tqdm import tqdm  # type: ignore[import-untyped]
        progress: Callable = lambda it, **kw: tqdm(it, **kw)
    except ImportError:
        progress = lambda it, **kw: it  # no-op wrapper

    if n_workers == 1:
        # Sequential — easier to debug
        for task in progress(tasks, desc="Docking", unit="mol"):
            result = _dock_one(task)
            if result is not None:
                new_results.append(result)
                _append_result(results_file, result)
    else:
        # Parallel
        with mp.Pool(n_workers) as pool:
            for result in progress(
                pool.imap_unordered(_dock_one, tasks),
                total=len(tasks),
                desc="Docking",
                unit="mol",
            ):
                if result is not None:
                    new_results.append(result)
                    _append_result(results_file, result)

    elapsed = time.perf_counter() - start
    logger.info(
        "Screening complete: %d/%d succeeded in %.1f s.",
        len(new_results),
        len(pending),
        elapsed,
    )

    # Merge new results with previously completed
    all_results = list(completed.values()) + new_results
    return _finalise_results(all_results, job, elapsed)


def _dock_one(args: tuple) -> DockingResult | None:
    """Dock a single ligand (designed for use with multiprocessing)."""
    receptor_path, smiles, name, site, config = args

    try:
        mol, pdbqt = prepare_ligand_pdbqt(smiles, name=name)
        if mol is None or not pdbqt:
            logger.warning("Ligand prep failed: %s (%s).", name, smiles)
            return None

        result = dock(
            receptor_pdbqt=receptor_path,
            ligand_pdbqt=pdbqt,
            site=site,
            config=config,
            ligand_name=name,
            ligand_smiles=smiles,
        )
        return result

    except Exception:
        logger.exception("Docking failed for %s (%s).", name, smiles)
        return None


# ── Persistence helpers ─────────────────────────────────────────────

def _load_completed(results_file: Path) -> dict[str, DockingResult]:
    """Load previously completed results from a JSONL file."""
    completed: dict[str, DockingResult] = {}

    if not results_file.exists():
        return completed

    for line in results_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            result = DockingResult.model_validate(data)
            completed[result.ligand_name] = result
        except Exception:
            logger.debug("Skipping malformed result line.")

    return completed


def _append_result(results_file: Path, result: DockingResult) -> None:
    """Append a single result to the JSONL file."""
    with results_file.open("a") as f:
        f.write(result.model_dump_json() + "\n")


def _finalise(
    completed: dict[str, DockingResult],
    job: ScreeningJob,
) -> ScreeningResult:
    """Build a ScreeningResult from already-completed results only."""
    return _finalise_results(list(completed.values()), job, 0.0)


def _finalise_results(
    results: list[DockingResult],
    job: ScreeningJob,
    elapsed: float,
) -> ScreeningResult:
    """Build a sorted ScreeningResult from a list of DockingResults."""
    # Sort by best affinity (most negative first)
    results.sort(
        key=lambda r: r.poses[0].affinity_kcal_mol if r.poses else 0.0,
    )

    # Write a summary JSON
    summary = {
        "total_ligands": len(job.ligands),
        "completed": len(results),
        "elapsed_seconds": round(elapsed, 1),
        "top_hits": [
            {
                "name": r.ligand_name,
                "smiles": r.ligand_smiles,
                "affinity": r.poses[0].affinity_kcal_mol if r.poses else None,
            }
            for r in results[:20]
        ],
    }
    summary_path = job.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return ScreeningResult(
        results=results,
        total_ligands=len(job.ligands),
        completed=len(results),
        elapsed_seconds=round(elapsed, 1),
    )
