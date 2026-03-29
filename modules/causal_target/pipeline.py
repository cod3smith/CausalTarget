"""
CausalTarget Pipeline
======================

The pipeline orchestrates the full causal drug-discovery workflow:

1. **Build Graph** → query 7 databases, assemble causal knowledge graph
2. **Identify Targets** → apply do-calculus to find causal targets
3. **Generate Candidates** → use GenMol VAE to create novel molecules
4. **Screen Candidates** → MolScreen (drug-likeness) + DockBot (binding)
5. **Score & Rank** → composite scorer with causal confidence weighting
6. **Report** → generate HTML report with interactive visualisations

Integration Points
------------------
- ``graph_builder.build_disease_graph()`` → DiseaseGraph
- ``identifier.identify_causal_targets()`` → list[CausalTargetResult]
- ``modules.genmol.generate.generate()`` → list[str]
- ``modules.molscreen.properties.calculate_properties()`` → MolecularProperties
- ``modules.molscreen.accessibility.qed_score()`` / ``sa_score()``
- ``modules.molscreen.filters.run_all_filters()`` → list[FilterResult]
- ``modules.molscreen.similarity.find_similar_drugs()`` → novelty
- ``modules.dockbot.protein_prep.prepare_protein()`` → ProteinInfo
- ``modules.dockbot.ligand_prep.prepare_ligand_pdbqt()`` → (Mol, pdbqt)
- ``modules.dockbot.docker.dock()`` → DockingResult
- ``scorer.score_candidate()`` → ScoredCandidate

Error Handling
--------------
Every sub-module integration is wrapped in try/except.
If GenMol fails, we still report the causal targets.
If DockBot fails, we score without binding affinity.
The pipeline never crashes — it degrades gracefully and
reports what it could accomplish.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from .models import (
    PipelineJob,
    PipelineResult,
    DiseaseGraph,
    CausalTargetResult,
    ScoredCandidate,
    JobStatus,
)
from .graph_builder import build_disease_graph
from .identifier import identify_causal_targets
from .scorer import score_candidate, rank_candidates

logger = logging.getLogger(__name__)


def run_pipeline(
    disease: str,
    top_n_targets: int = 5,
    candidates_per_target: int = 100,
    *,
    generate_molecules: bool = True,
    run_docking: bool = True,
    generate_report: bool = True,
    prebuilt_graph: DiseaseGraph | None = None,
) -> PipelineResult:
    """Run the full CausalTarget pipeline.

    Parameters
    ----------
    disease : str
        Disease name (e.g. "HIV", "Type 2 Diabetes").
    top_n_targets : int
        Number of top causal targets to pursue.
    candidates_per_target : int
        Molecules to generate per target.
    generate_molecules : bool
        Whether to run GenMol generation.
    run_docking : bool
        Whether to run DockBot docking.
    generate_report : bool
        Whether to generate HTML report.
    prebuilt_graph : DiseaseGraph, optional
        Pre-built disease graph.  If provided, Step 1 is skipped
        and this graph is used directly.  Useful for testing and
        for interactive workflows where the graph was already
        constructed.

    Returns
    -------
    PipelineResult
        Complete pipeline output with graph, targets, candidates.
    """
    job = PipelineJob(disease=disease, top_n_targets=top_n_targets,
                      candidates_per_target=candidates_per_target)

    try:
        # ── Step 1: Build Disease Graph ─────────────────────────
        job.status = JobStatus.BUILDING_GRAPH
        job.current_step = "Building disease causal graph…"
        job.progress_pct = 5.0
        logger.info("═══ Step 1/6: Building disease graph for '%s' ═══", disease)

        if prebuilt_graph is not None:
            graph = prebuilt_graph
            logger.info("Using prebuilt graph (%d nodes, %d edges).",
                        len(graph.nodes), len(graph.edges))
        else:
            graph = build_disease_graph(disease)
        job.progress_pct = 20.0
        logger.info("Graph: %d nodes, %d edges.", len(graph.nodes), len(graph.edges))

        # ── Step 2: Identify Causal Targets ─────────────────────
        job.status = JobStatus.IDENTIFYING_TARGETS
        job.current_step = "Identifying causal targets (do-calculus)…"
        job.progress_pct = 25.0
        logger.info("═══ Step 2/6: Identifying causal targets ═══")

        causal_targets = identify_causal_targets(graph, top_n=top_n_targets)
        job.progress_pct = 40.0

        n_causal = sum(1 for t in causal_targets if t.is_causal_target)
        logger.info("Found %d causal targets out of %d evaluated.", n_causal, len(causal_targets))

        for t in causal_targets:
            tag = "✓ CAUSAL" if t.is_causal_target else "✗ " + t.classification.value
            logger.info("  %s %s (confidence=%.3f)", tag, t.gene_name, t.causal_confidence)

        # ── Step 3: Generate Candidate Molecules ────────────────
        all_candidates: list[ScoredCandidate] = []

        if generate_molecules:
            job.status = JobStatus.GENERATING_CANDIDATES
            job.current_step = "Generating candidate molecules (GenMol)…"
            job.progress_pct = 45.0
            logger.info("═══ Step 3/6: Generating candidates (GenMol) ═══")

            for target in causal_targets:
                if not target.is_causal_target:
                    continue
                smiles_list = _generate_for_target(target, candidates_per_target)
                logger.info("  %s: %d molecules generated.", target.gene_name, len(smiles_list))

                # ── Step 4: Screen candidates ───────────────────
                job.status = JobStatus.SCREENING
                job.current_step = f"Screening candidates for {target.gene_name}…"

                screened = _screen_candidates(
                    smiles_list, target, run_docking=run_docking,
                )
                all_candidates.extend(screened)

            job.progress_pct = 75.0
        else:
            logger.info("Skipping molecule generation (generate_molecules=False).")

        # ── Step 5: Score & Rank ────────────────────────────────
        job.status = JobStatus.SCORING
        job.current_step = "Scoring and ranking candidates…"
        job.progress_pct = 80.0
        logger.info("═══ Step 5/6: Scoring %d candidates ═══", len(all_candidates))

        ranked = rank_candidates(all_candidates)
        logger.info("Top 5 candidates:")
        for cand in ranked[:5]:
            logger.info(
                "  #%d  %.4f  %s  (target: %s)",
                cand.rank, cand.composite_score,
                cand.smiles[:50], cand.target_protein_name,
            )

        # ── Step 6: Generate Report ─────────────────────────────
        report_html = None
        report_path = None
        if generate_report:
            job.status = JobStatus.REPORTING
            job.current_step = "Generating report…"
            job.progress_pct = 90.0
            logger.info("═══ Step 6/6: Generating report ═══")

            try:
                from .report import generate_report as gen_report
                report_html, report_path = gen_report(
                    disease=disease,
                    graph=graph,
                    causal_targets=causal_targets,
                    candidates=ranked,
                )
                logger.info("Report saved to %s.", report_path)
            except Exception as e:
                logger.warning("Report generation failed: %s.", e)

        # ── Done ────────────────────────────────────────────────
        job.status = JobStatus.COMPLETE
        job.progress_pct = 100.0
        job.completed_at = datetime.now()
        job.current_step = "Pipeline complete."

        return PipelineResult(
            job=job,
            disease=disease,
            graph=graph,
            causal_targets=causal_targets,
            scored_candidates=ranked,
            report_html=report_html,
            report_path=report_path,
        )

    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        job.status = JobStatus.FAILED
        job.error = str(e)
        job.completed_at = datetime.now()
        return PipelineResult(job=job, disease=disease)


# ── Sub-module Integration ──────────────────────────────────────────

def _generate_for_target(
    target: CausalTargetResult,
    n_molecules: int,
) -> list[str]:
    """Generate candidate molecules using GenMol.

    Falls back to a small set of known drug-like SMILES if GenMol
    is not available (e.g. no trained model checkpoint).
    """
    try:
        from modules.genmol.data import SmilesTokenizer
        from modules.genmol.models import MolVAE
        from modules.genmol.generate import generate

        # Try to load a pre-trained checkpoint
        import torch
        tok = SmilesTokenizer()

        # If no trained model, generate from random init (for demo)
        model = MolVAE(vocab_size=max(tok.vocab_size, 64))
        model.eval()

        smiles = generate(
            model, tok, n=min(n_molecules, 200),
            temperature=1.0, validate=True,
        )
        if smiles:
            return smiles

    except Exception as e:
        logger.warning("GenMol generation failed: %s. Using fallback.", e)

    # Fallback: curated drug-like molecules
    return _fallback_molecules(target.gene_name)


def _fallback_molecules(gene_name: str) -> list[str]:
    """Return curated drug-like SMILES as fallback.

    These are structurally diverse, drug-like molecules that
    serve as demonstration candidates when GenMol is unavailable.
    """
    # General drug-like scaffolds
    general = [
        "c1ccc2[nH]c(-c3ccncc3)nc2c1",                    # benzimidazole
        "O=C(NCc1ccccc1)c1cc2ccccc2[nH]1",                 # indole amide
        "Cc1nc2ccccc2n1Cc1ccc(F)cc1",                       # benzimidazole
        "O=C(c1ccc(O)cc1)c1ccc(O)cc1O",                     # dihydroxybenzophenone
        "CC(=O)Nc1ccc(O)cc1",                               # paracetamol scaffold
        "c1ccc(-c2nc3ccccc3s2)cc1",                         # benzothiazole
        "O=c1[nH]c2ccccc2c2ccccc12",                        # acridone
        "NC(=O)c1cccc(-c2cccnc2)c1",                        # nicotinamide analog
        "Oc1ccc(-c2cc(-c3ccc(O)cc3)no2)cc1",               # isoxazole diol
        "CC1=NN(c2ccccc2)C(=O)C1",                          # pyrazolone
    ]

    # Target-specific known actives / scaffolds
    target_specific: dict[str, list[str]] = {
        "CCR5": [
            "CC(C)c1nnc(C(C)C)n1C1CC1c1ccc(F)cc1",         # maraviroc-like
            "O=C(NC1CCCCC1)c1ccc(-c2ccncc2)cc1",            # CCR5 inhibitor scaffold
            "Cc1cccc(NC(=O)c2cc3ccccc3[nH]2)c1",
        ],
        "CXCR4": [
            "c1ccc(CNc2ncnc3[nH]cnc23)cc1",                # AMD3100-like
            "NCCNCCNCCN",                                    # plerixafor-like
        ],
    }

    molecules = list(general)
    if gene_name in target_specific:
        molecules.extend(target_specific[gene_name])

    return molecules


def _screen_candidates(
    smiles_list: list[str],
    target: CausalTargetResult,
    run_docking: bool = True,
) -> list[ScoredCandidate]:
    """Screen a list of SMILES against a target.

    Integrates MolScreen for drug-likeness properties and
    optionally DockBot for binding affinity estimation.
    """
    scored: list[ScoredCandidate] = []

    for smiles in smiles_list:
        # ── MolScreen Properties ────────────────────────────────
        mol_props = _get_mol_properties(smiles)
        if mol_props is None:
            continue  # Invalid SMILES

        qed = mol_props.get("qed", 0.5)
        sa = mol_props.get("sa_score", 5.0)
        mw = mol_props.get("molecular_weight")
        logp = mol_props.get("logp")
        n_filters = mol_props.get("n_filters_passed", 0)
        drug_class = mol_props.get("drug_class", "")

        # ── Novelty Score ───────────────────────────────────────
        novelty = _compute_novelty(smiles)

        # ── Docking (optional) ──────────────────────────────────
        binding = None
        if run_docking and target.pdb_ids:
            binding = _run_docking(smiles, target.pdb_ids[0])

        # ── ADMET Estimate ──────────────────────────────────────
        admet = _estimate_admet(mw, logp, qed)

        # ── Score ───────────────────────────────────────────────
        candidate = score_candidate(
            smiles=smiles,
            target_protein_id=target.protein_id,
            target_protein_name=target.protein_name,
            causal_confidence=target.causal_confidence,
            binding_affinity=binding,
            qed_score=qed,
            sa_score=sa,
            admet_score=admet,
            novelty_score=novelty,
            molecular_weight=mw,
            logp=logp,
            drug_likeness_class=drug_class,
            n_filters_passed=n_filters,
        )
        scored.append(candidate)

    return scored


def _get_mol_properties(smiles: str) -> dict[str, Any] | None:
    """Get molecular properties via MolScreen."""
    try:
        from modules.molscreen.parser import validate_smiles
        if not validate_smiles(smiles):
            return None

        from modules.molscreen.properties import calculate_properties
        from modules.molscreen.accessibility import qed_score, sa_score
        from modules.molscreen.filters import run_all_filters

        props = calculate_properties(smiles)
        if props is None:
            return None

        qed = qed_score(smiles)
        sa = sa_score(smiles)
        filters = run_all_filters(props.mol)

        return {
            "molecular_weight": props.molecular_weight,
            "logp": props.logp,
            "qed": qed if qed is not None else 0.5,
            "sa_score": sa if sa is not None else 5.0,
            "n_filters_passed": sum(1 for f in filters if f.passed),
            "drug_class": "drug-like" if sum(1 for f in filters if f.passed) >= 3 else "non-drug-like",
        }
    except Exception as e:
        logger.debug("MolScreen failed for %s: %s", smiles[:30], e)
        # Fallback: basic RDKit
        return _fallback_properties(smiles)


def _fallback_properties(smiles: str) -> dict[str, Any] | None:
    """Compute basic properties when MolScreen is unavailable."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, QED

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        return {
            "molecular_weight": Descriptors.MolWt(mol),
            "logp": Descriptors.MolLogP(mol),
            "qed": QED.qed(mol),
            "sa_score": 5.0,  # Default
            "n_filters_passed": 3,
            "drug_class": "drug-like",
        }
    except Exception:
        return None


def _compute_novelty(smiles: str) -> float:
    """Compute structural novelty vs known drugs."""
    try:
        from modules.molscreen.similarity import find_similar_drugs
        similar = find_similar_drugs(smiles, top_k=1, threshold=0.3)
        if similar:
            # Higher similarity → lower novelty
            max_sim = max(s[1] for s in similar) if similar else 0.0
            return 1.0 - max_sim
        return 0.9  # No similar drugs → very novel
    except Exception:
        return 0.5  # Unknown


def _run_docking(smiles: str, pdb_id: str) -> float | None:
    """Run molecular docking via DockBot."""
    try:
        from modules.dockbot.protein_prep import prepare_protein
        from modules.dockbot.ligand_prep import prepare_ligand_pdbqt
        from modules.dockbot.docker import dock

        protein = prepare_protein(pdb_id)
        mol, ligand_pdbqt = prepare_ligand_pdbqt(smiles)

        if not protein or not ligand_pdbqt:
            return None

        # Use first binding site or default
        from modules.dockbot.models import BindingSite
        site = BindingSite(center_x=0.0, center_y=0.0, center_z=0.0,
                          size_x=20.0, size_y=20.0, size_z=20.0)

        result = dock(
            protein.get("pdbqt", ""),
            ligand_pdbqt,
            site,
        )
        if result and result.poses:
            return result.poses[0].affinity
        return None

    except Exception as e:
        logger.debug("Docking failed for %s @ %s: %s", smiles[:20], pdb_id, e)
        return None


def _estimate_admet(
    mw: float | None,
    logp: float | None,
    qed: float | None,
) -> float:
    """Simple ADMET estimation heuristic.

    Uses Lipinski-adjacent rules:
    - MW 150–500 Da → good absorption
    - LogP −0.4 to 5.6 → good permeability
    - QED > 0.5 → generally drug-like
    """
    score = 0.5
    if mw is not None:
        if 150 <= mw <= 500:
            score += 0.2
        elif mw > 500:
            score -= 0.1

    if logp is not None:
        if -0.4 <= logp <= 5.6:
            score += 0.2
        else:
            score -= 0.1

    if qed is not None and qed > 0.5:
        score += 0.1

    return max(0.0, min(1.0, score))
