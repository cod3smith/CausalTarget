"""
Molecule Generation
====================

Generates novel molecules from a trained VAE by sampling from the
latent space and decoding to SMILES.

Generation modes
----------------
1. **Random sampling** — draw ``z ~ N(0, I)`` and decode.  The
   simplest approach; generates diverse molecules.

2. **Temperature-controlled** — scale logits by ``1/T`` before
   softmax.  Lower T → more conservative (higher probability
   tokens), higher T → more diverse (and more invalid).

3. **Beam search** — maintain ``beam_width`` candidate sequences at
   each step, expanding the most probable ones.  Produces higher
   validity but lower diversity.

4. **Latent interpolation** — decode a series of points between two
   latent vectors to observe smooth structural transformations.

5. **Conditional generation** (CVAE) — specify desired molecular
   properties and generate molecules matching those targets.

Post-generation filtering
-------------------------
Generated SMILES are validated with RDKit, deduplicated, and
optionally filtered through the MolScreen pipeline for drug-likeness.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from rdkit import Chem, RDLogger

from .data.tokenizer import SmilesTokenizer
from .models.vae import MolVAE
from .models.cvae import MolCVAE, PropertyNormalizer

logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")


def _decode_token_ids(
    token_ids: torch.Tensor,
    tokenizer: SmilesTokenizer,
) -> list[str]:
    """Decode a batch of token-ID tensors to SMILES strings.

    Parameters
    ----------
    token_ids : Tensor[B, L]
        Generated token IDs.
    tokenizer : SmilesTokenizer
        Tokenizer for decoding.

    Returns
    -------
    list[str]
        Decoded SMILES (may be invalid).
    """
    results = []
    for row in token_ids:
        smi = tokenizer.decode(row.tolist(), strip_special=True)
        results.append(smi)
    return results


def _validate_smiles(smiles_list: list[str]) -> list[str]:
    """Filter to only valid SMILES and canonicalize them.

    Parameters
    ----------
    smiles_list : list[str]
        Raw decoded SMILES.

    Returns
    -------
    list[str]
        Valid, canonical SMILES (deduplicated).
    """
    valid: list[str] = []
    seen: set[str] = set()

    for smi in smiles_list:
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            canonical = Chem.MolToSmiles(mol)
            if canonical not in seen:
                seen.add(canonical)
                valid.append(canonical)

    return valid


def generate(
    model: MolVAE,
    tokenizer: SmilesTokenizer,
    n: int = 100,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
    validate: bool = True,
    deduplicate: bool = True,
) -> list[str]:
    """Generate molecules by sampling from the prior N(0, I).

    Parameters
    ----------
    model : MolVAE
        Trained VAE model.
    tokenizer : SmilesTokenizer
        Tokenizer for decoding.
    n : int
        Number of molecules to attempt generating.
    temperature : float
        Sampling temperature (0.5–1.5 recommended).
    device : torch.device, optional
        Computation device.
    validate : bool
        Only return valid SMILES.
    deduplicate : bool
        Remove duplicate molecules.

    Returns
    -------
    list[str]
        Generated (and optionally validated) SMILES.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    token_ids = model.sample(n=n, temperature=temperature, device=device)
    decoded = _decode_token_ids(token_ids, tokenizer)

    if validate:
        decoded = _validate_smiles(decoded)

    if deduplicate:
        decoded = list(dict.fromkeys(decoded))  # preserve order

    logger.info(
        "Generated %d molecules (%d requested, T=%.2f).",
        len(decoded),
        n,
        temperature,
    )
    return decoded


def generate_conditional(
    model: MolCVAE,
    tokenizer: SmilesTokenizer,
    conditions: dict[str, float],
    normalizer: PropertyNormalizer,
    n: int = 100,
    temperature: float = 1.0,
    device: Optional[torch.device] = None,
    validate: bool = True,
) -> list[str]:
    """Generate molecules with desired properties (CVAE).

    Parameters
    ----------
    model : MolCVAE
        Trained conditional VAE.
    tokenizer : SmilesTokenizer
        Tokenizer for decoding.
    conditions : dict[str, float]
        Desired properties, e.g. ``{"mw": 350, "logp": 2.5, "qed": 0.7}``.
    normalizer : PropertyNormalizer
        Fitted normalizer for property scaling.
    n : int
        Number of molecules to generate.
    temperature : float
        Sampling temperature.
    device : torch.device, optional
        Device.
    validate : bool
        Only return valid SMILES.

    Returns
    -------
    list[str]
        Generated SMILES matching the desired conditions.
    """
    if device is None:
        device = next(model.parameters()).device

    # Normalize conditions
    norm_vals = normalizer.transform(conditions)
    cond = torch.tensor(norm_vals, dtype=torch.float32, device=device)

    token_ids = model.sample(cond=cond, n=n, temperature=temperature)
    decoded = _decode_token_ids(token_ids, tokenizer)

    if validate:
        decoded = _validate_smiles(decoded)

    logger.info(
        "Conditional generation: %d valid molecules for %s.",
        len(decoded),
        conditions,
    )
    return decoded


def interpolate(
    model: MolVAE,
    tokenizer: SmilesTokenizer,
    smiles_a: str,
    smiles_b: str,
    n_steps: int = 10,
    device: Optional[torch.device] = None,
) -> list[str]:
    """Interpolate between two molecules in latent space.

    Encodes both molecules, then decodes ``n_steps`` evenly-spaced
    points along the line between them.  This reveals how the model
    organises chemical space — nearby points should be structurally
    similar.

    Parameters
    ----------
    model : MolVAE
        Trained VAE.
    tokenizer : SmilesTokenizer
        Tokenizer.
    smiles_a : str
        Starting molecule SMILES.
    smiles_b : str
        Ending molecule SMILES.
    n_steps : int
        Number of interpolation points (including endpoints).
    device : torch.device, optional
        Device.

    Returns
    -------
    list[str]
        SMILES at each interpolation point (some may be invalid).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Encode both molecules
    ids_a = torch.tensor(
        [tokenizer.encode(smiles_a)], dtype=torch.long, device=device
    )
    ids_b = torch.tensor(
        [tokenizer.encode(smiles_b)], dtype=torch.long, device=device
    )

    with torch.no_grad():
        z_a = model.encode(ids_a)  # [1, D]
        z_b = model.encode(ids_b)  # [1, D]

    # Linear interpolation
    alphas = torch.linspace(0, 1, n_steps, device=device)
    z_interp = torch.stack(
        [z_a * (1 - a) + z_b * a for a in alphas]
    ).squeeze(1)  # [n_steps, D]

    # Decode each point
    with torch.no_grad():
        token_ids = model.decode(z_interp, greedy=True)

    decoded = _decode_token_ids(token_ids, tokenizer)

    logger.info(
        "Interpolation: %s → %s (%d steps).",
        smiles_a,
        smiles_b,
        n_steps,
    )
    return decoded


def screen_generated(
    smiles_list: list[str],
    *,
    use_molscreen: bool = True,
    max_sa_score: float = 4.0,
    min_qed: float = 0.3,
) -> list[dict]:
    """Filter generated molecules through MolScreen.

    Integrates with the ``modules.molscreen`` pipeline to assess
    drug-likeness, synthetic accessibility, and QED.

    Parameters
    ----------
    smiles_list : list[str]
        Valid SMILES to screen.
    use_molscreen : bool
        Whether to run the full MolScreen pipeline.
    max_sa_score : float
        Maximum synthetic accessibility score (lower = easier to
        synthesise).  Typical drug cutoff is ~4.0.
    min_qed : float
        Minimum QED score (higher = more drug-like).

    Returns
    -------
    list[dict]
        Screening results for molecules that pass filters.
    """
    results: list[dict] = []

    if not use_molscreen:
        return [{"smiles": s} for s in smiles_list]

    try:
        from modules.molscreen.properties import calculate_properties
        from modules.molscreen.filters import run_all_filters, classify_drug_likeness
        from modules.molscreen.accessibility import sa_score, qed_score
    except ImportError:
        logger.warning(
            "MolScreen not available. Returning unscreened molecules."
        )
        return [{"smiles": s} for s in smiles_list]

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        # Compute properties
        props = calculate_properties(smi)
        if props is None:
            continue

        # Run filters
        filters = run_all_filters(mol)
        classification = classify_drug_likeness(filters)

        # Accessibility
        sa = sa_score(smi)
        qed = qed_score(smi)

        if sa is not None and sa > max_sa_score:
            continue
        if qed is not None and qed < min_qed:
            continue

        results.append({
            "smiles": smi,
            "properties": props.model_dump() if hasattr(props, "model_dump") else {},
            "drug_likeness": classification,
            "sa_score": sa,
            "qed_score": qed,
            "passed_filters": [f.name for f in filters if f.passed],
        })

    logger.info(
        "Screening: %d / %d molecules passed (SA ≤ %.1f, QED ≥ %.1f).",
        len(results),
        len(smiles_list),
        max_sa_score,
        min_qed,
    )
    return results
