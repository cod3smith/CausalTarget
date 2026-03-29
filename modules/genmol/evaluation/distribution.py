"""
Property Distribution Analysis
================================

Compares the property distributions of generated molecules against
the training set using histograms and statistical tests.

Why distribution matching matters
---------------------------------
A good generative model should produce molecules whose properties
(MW, LogP, HBD, HBA, QED, SA score, etc.) follow the same
distribution as the training data.

* **Visual comparison** — overlapping histograms reveal if the
  generated distribution is shifted, narrower, or has missing modes.
* **KS test** (Kolmogorov–Smirnov) — a non-parametric statistical
  test that measures the maximum distance between two empirical
  CDFs.  A KS p-value > 0.05 suggests the distributions are not
  significantly different.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski

logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")


def compute_properties(smiles_list: list[str]) -> dict[str, list[float]]:
    """Compute molecular properties for a list of SMILES.

    Returns a dict of property name → list of values, suitable
    for histogram plotting and statistical tests.

    Properties computed:
    * ``mw`` — molecular weight
    * ``logp`` — Wildman-Crippen LogP
    * ``hbd`` — hydrogen bond donors
    * ``hba`` — hydrogen bond acceptors
    * ``tpsa`` — topological polar surface area
    * ``rotbonds`` — rotatable bonds
    * ``rings`` — number of rings
    * ``heavy_atoms`` — heavy atom count

    Parameters
    ----------
    smiles_list : list[str]
        Valid SMILES strings.

    Returns
    -------
    dict[str, list[float]]
        Property distributions.
    """
    properties: dict[str, list[float]] = {
        "mw": [],
        "logp": [],
        "hbd": [],
        "hba": [],
        "tpsa": [],
        "rotbonds": [],
        "rings": [],
        "heavy_atoms": [],
    }

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        properties["mw"].append(Descriptors.ExactMolWt(mol))
        properties["logp"].append(Descriptors.MolLogP(mol))
        properties["hbd"].append(float(Lipinski.NumHDonors(mol)))
        properties["hba"].append(float(Lipinski.NumHAcceptors(mol)))
        properties["tpsa"].append(Descriptors.TPSA(mol))
        properties["rotbonds"].append(float(Lipinski.NumRotatableBonds(mol)))
        properties["rings"].append(float(Descriptors.RingCount(mol)))
        properties["heavy_atoms"].append(float(mol.GetNumHeavyAtoms()))

    return properties


def ks_test(
    generated: list[float],
    reference: list[float],
) -> dict[str, float]:
    """Two-sample Kolmogorov–Smirnov test.

    The KS statistic measures the maximum vertical distance between
    the empirical CDFs of two samples.  Larger values indicate the
    distributions are more different.

    Parameters
    ----------
    generated : list[float]
        Property values from generated molecules.
    reference : list[float]
        Property values from training/reference molecules.

    Returns
    -------
    dict
        ``{"statistic": float, "p_value": float}``
    """
    from scipy.stats import ks_2samp

    stat, p_val = ks_2samp(generated, reference)
    return {"statistic": float(stat), "p_value": float(p_val)}


def compare_distributions(
    generated_smiles: list[str],
    reference_smiles: list[str],
) -> dict[str, dict[str, float]]:
    """Compare property distributions between generated and reference sets.

    Computes the KS test for each property.

    Parameters
    ----------
    generated_smiles : list[str]
        Valid SMILES from the model.
    reference_smiles : list[str]
        Training or reference SMILES.

    Returns
    -------
    dict[str, dict]
        ``{property_name: {"statistic": ..., "p_value": ...}}``
    """
    gen_props = compute_properties(generated_smiles)
    ref_props = compute_properties(reference_smiles)

    results: dict[str, dict[str, float]] = {}

    for prop_name in gen_props:
        if gen_props[prop_name] and ref_props[prop_name]:
            ks_result = ks_test(gen_props[prop_name], ref_props[prop_name])
            results[prop_name] = ks_result
            logger.info(
                "KS test [%s]: D=%.4f, p=%.4f",
                prop_name,
                ks_result["statistic"],
                ks_result["p_value"],
            )
        else:
            results[prop_name] = {"statistic": float("nan"), "p_value": 0.0}

    return results


def distribution_summary(
    smiles_list: list[str],
) -> dict[str, dict[str, float]]:
    """Compute summary statistics for molecular properties.

    Parameters
    ----------
    smiles_list : list[str]
        Valid SMILES.

    Returns
    -------
    dict[str, dict]
        ``{property: {"mean": ..., "std": ..., "min": ..., "max": ..., "median": ...}}``
    """
    props = compute_properties(smiles_list)
    summary: dict[str, dict[str, float]] = {}

    for name, values in props.items():
        if values:
            arr = np.array(values)
            summary[name] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "median": float(np.median(arr)),
                "count": len(values),
            }

    return summary
