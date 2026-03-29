"""
MolScreen Data Models
=====================

Pydantic models for molecular screening data. These models define the schema
for molecular properties, filter results, and full screening reports.

All models are serialisable to JSON for seamless integration with the
FastAPI layer and downstream pipeline modules (DockBot, GenMol).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DrugLikelihoodCategory(str, Enum):
    """Classification of a molecule's drug-likeness profile.

    - **DRUG_LIKE**: Passes Lipinski + Veber — looks like an oral drug.
    - **LEAD_LIKE**: Has some violations but is still a viable starting point
      for medicinal-chemistry optimisation.
    - **NOT_DRUG_LIKE**: Fails too many filters to be a practical oral drug
      candidate without significant structural modification.
    """

    DRUG_LIKE = "Drug-like"
    LEAD_LIKE = "Lead-like"
    NOT_DRUG_LIKE = "Not drug-like"


class MolecularProperties(BaseModel):
    """Calculated physico-chemical properties of a molecule.

    Each property has a direct impact on whether a compound can become a
    successful drug.  The docstrings below explain *why* each matters.
    """

    smiles: str = Field(
        ...,
        description="Canonical SMILES string — the standardised text representation of the molecule.",
    )

    molecular_weight: float = Field(
        ...,
        description=(
            "Molecular weight in Daltons. Drugs need to be absorbed from the gut and "
            "distributed through the blood — molecules above ~500 Da struggle to cross "
            "biological membranes (Lipinski's Rule of Five)."
        ),
    )

    logp: float = Field(
        ...,
        description=(
            "Octanol-water partition coefficient (LogP). Measures hydrophobicity — how "
            "much a molecule prefers oil over water. Cell membranes are lipid bilayers, "
            "so drugs need moderate LogP (ideally 1–3) to cross them. Too high → poor "
            "solubility; too low → can't permeate membranes."
        ),
    )

    hbd: int = Field(
        ...,
        description=(
            "Hydrogen Bond Donors — count of NH and OH groups. These form hydrogen bonds "
            "with water, which must be broken to cross a lipid membrane. More donors = "
            "harder membrane permeation. Lipinski sets the limit at ≤ 5."
        ),
    )

    hba: int = Field(
        ...,
        description=(
            "Hydrogen Bond Acceptors — count of N and O atoms. Like donors, these "
            "hydrogen-bond to water and hinder membrane crossing. Lipinski limit: ≤ 10."
        ),
    )

    tpsa: float = Field(
        ...,
        description=(
            "Topological Polar Surface Area in Å². The sum of surfaces of polar atoms "
            "(O, N, and attached H). TPSA predicts intestinal absorption and blood-brain "
            "barrier penetration: ≤ 140 Å² for oral absorption, ≤ 90 Å² for CNS drugs."
        ),
    )

    rotatable_bonds: int = Field(
        ...,
        description=(
            "Number of rotatable bonds — measures molecular flexibility. Too many "
            "(> 10) reduces oral bioavailability because the molecule loses too much "
            "entropy when binding to a target (Veber rules)."
        ),
    )

    num_rings: int = Field(
        ...,
        description=(
            "Total number of rings. Rings provide rigidity and shape to a molecule. "
            "Most drugs have 1–5 rings; too many can hurt solubility."
        ),
    )

    num_aromatic_rings: int = Field(
        ...,
        description=(
            "Number of aromatic rings (flat, electron-rich rings like benzene). "
            "Aromatic rings aid target binding via π-stacking but too many (> 3) "
            "increase planarity, which causes aggregation and poor solubility."
        ),
    )

    fsp3: float = Field(
        ...,
        description=(
            "Fraction of sp3-hybridised carbons. Measures three-dimensionality — "
            "higher Fsp3 (≥ 0.42) correlates with higher clinical success rates "
            "because 3D molecules are more selective and soluble than flat ones."
        ),
    )

    molar_refractivity: float = Field(
        ...,
        description=(
            "Molar refractivity (MR). Related to molecular volume and polarisability. "
            "Used in the Ghose filter (40–130) as a proxy for molecular size and "
            "electronic character."
        ),
    )


class FilterResult(BaseModel):
    """Result of a single drug-likeness filter evaluation.

    Each filter encodes decades of pharmaceutical industry knowledge about what
    makes a molecule likely to succeed as an oral drug.
    """

    name: str = Field(
        ...,
        description="Filter name, e.g. 'Lipinski', 'Veber', 'PAINS'.",
    )

    passed: bool = Field(
        ...,
        description="Whether the molecule passed this filter.",
    )

    violations: list[str] = Field(
        default_factory=list,
        description=(
            "Human-readable descriptions of each violation, e.g. "
            "'MW 623.4 exceeds limit of 500'."
        ),
    )


class SimilarDrug(BaseModel):
    """A reference drug that is structurally similar to the query molecule."""

    name: str = Field(..., description="Drug name.")
    smiles: str = Field(..., description="Canonical SMILES of the approved drug.")
    similarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Tanimoto similarity (0–1) based on Morgan fingerprints.",
    )
    indication: str = Field(
        default="",
        description="Primary therapeutic indication, if known.",
    )


class ScreeningReport(BaseModel):
    """Complete screening report for a single molecule.

    This is the primary output of MolScreen and the data contract consumed by
    downstream modules (DockBot for docking, GenMol for generative design).
    """

    smiles: str = Field(
        ...,
        description="Input SMILES (canonicalised if valid).",
    )

    valid: bool = Field(
        ...,
        description="Whether the SMILES could be parsed into a valid molecule.",
    )

    properties: MolecularProperties | None = Field(
        default=None,
        description="Calculated molecular properties (None if SMILES is invalid).",
    )

    filters: list[FilterResult] = Field(
        default_factory=list,
        description="Results of all drug-likeness filters.",
    )

    sa_score: float = Field(
        default=0.0,
        ge=1.0,
        le=10.0,
        description=(
            "Synthetic Accessibility score (1–10). 1 = trivially easy to synthesise, "
            "10 = practically impossible. Most approved drugs score 2–5."
        ),
    )

    qed_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Quantitative Estimate of Drug-likeness (0–1). Integrates multiple "
            "desirability functions for drug-relevant properties into a single score. "
            "Higher is more drug-like."
        ),
    )

    drug_likeness_summary: str = Field(
        default=DrugLikelihoodCategory.NOT_DRUG_LIKE.value,
        description="Overall drug-likeness classification.",
    )

    similar_drugs: list[SimilarDrug] = Field(
        default_factory=list,
        description="Top structurally similar FDA-approved drugs.",
    )

    model_config = {"json_schema_extra": {"example": {}}}


class ComparisonReport(BaseModel):
    """Side-by-side comparison of two molecules."""

    molecule_a: ScreeningReport
    molecule_b: ScreeningReport
    property_differences: dict[str, float] = Field(
        default_factory=dict,
        description="Absolute differences for each numeric property.",
    )
    tanimoto_similarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Tanimoto similarity between the two molecules.",
    )


class BatchRequest(BaseModel):
    """Request model for batch screening."""

    smiles_list: list[str] = Field(
        ...,
        min_length=1,
        description="List of SMILES strings to screen.",
    )


class ScreenRequest(BaseModel):
    """Request model for single molecule screening."""

    smiles: str = Field(..., description="SMILES string or drug name to screen.")


class CompareRequest(BaseModel):
    """Request model for molecule comparison."""

    smiles_a: str = Field(..., description="First SMILES string.")
    smiles_b: str = Field(..., description="Second SMILES string.")


class SimilarRequest(BaseModel):
    """Request model for similarity search."""

    smiles: str = Field(..., description="Query SMILES string.")
    top_n: int = Field(default=5, ge=1, le=50, description="Number of results.")
