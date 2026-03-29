"""
Generation Quality Metrics
============================

Quantitative metrics for evaluating molecular generative models.

These metrics are standard in the field (see Polykovskiy et al.,
*"Molecular Sets (MOSES): A Benchmarking Platform for Molecular
Generation Models"*, 2020) and measure different aspects of
generation quality:

* **Validity** — fraction of generated SMILES that parse to valid
  molecules.  A good VAE should achieve >90%.
* **Uniqueness** — fraction of valid molecules that are unique
  (no duplicates).  Mode collapse shows low uniqueness.
* **Novelty** — fraction of unique molecules *not* in the training
  set.  The model should generate new molecules, not memorize.
* **Diversity** — average pairwise Tanimoto distance between
  generated molecules (1 − similarity).  Higher = more diverse.
* **FCD** (Fréchet ChemNet Distance) — analogous to FID in image
  generation.  Measures distributional similarity between generated
  and reference molecules using neural network features.
  Lower FCD = better.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")


def validity(generated_smiles: list[str]) -> float:
    """Fraction of SMILES that are valid molecules.

    Parameters
    ----------
    generated_smiles : list[str]
        Raw SMILES from the decoder.

    Returns
    -------
    float
        Validity rate in [0, 1].
    """
    if not generated_smiles:
        return 0.0

    valid = sum(
        1 for s in generated_smiles
        if s and Chem.MolFromSmiles(s) is not None
    )
    return valid / len(generated_smiles)


def uniqueness(valid_smiles: list[str]) -> float:
    """Fraction of valid SMILES that are unique (canonical).

    Parameters
    ----------
    valid_smiles : list[str]
        Valid SMILES strings.

    Returns
    -------
    float
        Uniqueness rate in [0, 1].
    """
    if not valid_smiles:
        return 0.0

    canonical: set[str] = set()
    for s in valid_smiles:
        mol = Chem.MolFromSmiles(s)
        if mol:
            canonical.add(Chem.MolToSmiles(mol))

    return len(canonical) / len(valid_smiles)


def novelty(
    generated_smiles: list[str],
    training_smiles: set[str],
) -> float:
    """Fraction of generated molecules not in the training set.

    Parameters
    ----------
    generated_smiles : list[str]
        Valid, unique generated SMILES.
    training_smiles : set[str]
        Set of canonical training SMILES.

    Returns
    -------
    float
        Novelty rate in [0, 1].
    """
    if not generated_smiles:
        return 0.0

    novel = sum(
        1 for s in generated_smiles
        if s not in training_smiles
    )
    return novel / len(generated_smiles)


def internal_diversity(
    smiles_list: list[str],
    sample_size: int = 1000,
    seed: int = 42,
) -> float:
    """Average pairwise Tanimoto distance (1 − similarity).

    A measure of structural diversity: higher values mean the
    generated molecules are more spread across chemical space.

    Uses Morgan fingerprints (radius 2, 2048 bits).

    Parameters
    ----------
    smiles_list : list[str]
        Valid SMILES.
    sample_size : int
        Max molecules to compare (for speed).
    seed : int
        Random seed for subsampling.

    Returns
    -------
    float
        Internal diversity in [0, 1].
    """
    # Generate fingerprints
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            fps.append(gen.GetFingerprint(mol))

    if len(fps) < 2:
        return 0.0

    # Subsample if needed
    if len(fps) > sample_size:
        rng = np.random.RandomState(seed)
        indices = rng.choice(len(fps), sample_size, replace=False)
        fps = [fps[i] for i in indices]

    # Compute pairwise distances
    n = len(fps)
    total_dist = 0.0
    n_pairs = 0

    for i in range(n):
        for j in range(i + 1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            total_dist += (1.0 - sim)
            n_pairs += 1

    return total_dist / max(n_pairs, 1)


def compute_all_metrics(
    generated_smiles: list[str],
    training_smiles: Optional[list[str]] = None,
    diversity_sample: int = 1000,
) -> dict[str, float]:
    """Compute all standard generation metrics at once.

    Parameters
    ----------
    generated_smiles : list[str]
        Raw SMILES from the model.
    training_smiles : list[str], optional
        Training set for novelty computation.
    diversity_sample : int
        Sample size for diversity computation.

    Returns
    -------
    dict[str, float]
        Metric name → value.
    """
    # Validity
    val = validity(generated_smiles)

    # Get valid canonical SMILES
    valid_smiles: list[str] = []
    for s in generated_smiles:
        if not s:
            continue
        mol = Chem.MolFromSmiles(s)
        if mol:
            valid_smiles.append(Chem.MolToSmiles(mol))

    # Uniqueness
    unique_smiles = list(set(valid_smiles))
    uniq = len(unique_smiles) / max(len(valid_smiles), 1)

    # Novelty
    nov = 0.0
    if training_smiles is not None:
        training_set = set(training_smiles)
        nov = novelty(unique_smiles, training_set)

    # Diversity
    div = internal_diversity(unique_smiles, sample_size=diversity_sample)

    metrics = {
        "validity": val,
        "uniqueness": uniq,
        "novelty": nov,
        "diversity": div,
        "n_generated": len(generated_smiles),
        "n_valid": len(valid_smiles),
        "n_unique": len(unique_smiles),
    }

    logger.info(
        "Metrics: validity=%.3f uniqueness=%.3f novelty=%.3f diversity=%.3f",
        val,
        uniq,
        nov,
        div,
    )

    return metrics
