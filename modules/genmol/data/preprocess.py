"""
SMILES Preprocessing
=====================

Cleans, standardizes, and optionally augments SMILES strings before
feeding them to the generative model.

Why preprocess?
---------------
Raw SMILES from databases can be noisy:

* **Salts & fragments** — e.g. ``CC.Cl`` (methylamine hydrochloride)
  contains a counter-ion we don't want.  We keep the largest fragment.
* **Non-canonical** — the same molecule can have many SMILES
  representations.  Canonicalization ensures one-to-one mapping.
* **Stereochemistry** — ``@`` / ``@@`` markers add complexity.
  For initial VAE training, removing them simplifies the task.
* **Charged species** — formal charges ``[NH4+]`` are valid but rare;
  we optionally neutralize them.

SMILES augmentation
-------------------
RDKit can generate multiple *valid* SMILES for the same molecule by
randomizing the atom traversal order.  This is a free form of data
augmentation that helps the model learn SMILES syntax more robustly.
"""

from __future__ import annotations

import logging
import random
from typing import Optional

from rdkit import Chem, RDLogger
from rdkit.Chem import SaltRemover

logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    """Return the largest fragment from a molecule (remove salts/ions).

    Disconnected SMILES like ``CC.[Na+]`` have multiple fragments.
    We keep the largest one by heavy-atom count.
    """
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) == 1:
        return mol
    return max(frags, key=lambda f: f.GetNumHeavyAtoms())


def _neutralize_charges(mol: Chem.Mol) -> Chem.Mol:
    """Neutralize simple charges (e.g. [NH3+] → N, [O-] → O).

    Uses a SMARTS-based approach to convert common charged groups
    to their neutral form.  This makes the SMILES distribution
    simpler for the generative model to learn.
    """
    pattern = Chem.MolFromSmarts(
        "[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]"
    )
    at_matches = mol.GetSubstructMatches(pattern)
    if not at_matches:
        return mol

    emol = Chem.RWMol(mol)
    for (idx,) in at_matches:
        atom = emol.GetAtomWithIdx(idx)
        charge = atom.GetFormalCharge()
        hcount = atom.GetTotalNumHs()
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(hcount - charge)
        atom.UpdatePropertyCache()

    Chem.SanitizeMol(emol)
    return emol.GetMol()


def clean_smiles(
    smiles: str,
    *,
    remove_stereo: bool = True,
    neutralize: bool = True,
    keep_largest: bool = True,
    max_length: int = 120,
) -> Optional[str]:
    """Clean and standardize a single SMILES string.

    Parameters
    ----------
    smiles : str
        Raw SMILES.
    remove_stereo : bool
        Strip stereochemistry markers (``@``, ``/``, ``\\``).
    neutralize : bool
        Neutralize simple charges.
    keep_largest : bool
        Keep only the largest fragment (remove salts).
    max_length : int
        Discard SMILES longer than this after cleaning.

    Returns
    -------
    str or None
        Cleaned canonical SMILES, or ``None`` if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    if keep_largest:
        mol = _largest_fragment(mol)

    if neutralize:
        try:
            mol = _neutralize_charges(mol)
        except Exception:
            pass  # keep the molecule as-is

    if remove_stereo:
        Chem.RemoveStereochemistry(mol)

    canonical = Chem.MolToSmiles(mol)

    if len(canonical) > max_length - 2:  # room for <SOS> and <EOS>
        return None

    return canonical


def preprocess_dataset(
    smiles_list: list[str],
    *,
    remove_stereo: bool = True,
    neutralize: bool = True,
    keep_largest: bool = True,
    max_length: int = 120,
) -> list[str]:
    """Clean a list of SMILES strings, dropping invalid/duplicate entries.

    Parameters
    ----------
    smiles_list : list[str]
        Raw SMILES from the download step.
    remove_stereo : bool
        Strip stereochemistry.
    neutralize : bool
        Neutralize charges.
    keep_largest : bool
        Remove salts/fragments.
    max_length : int
        Maximum SMILES length (characters).

    Returns
    -------
    list[str]
        Deduplicated, cleaned SMILES.
    """
    seen: set[str] = set()
    cleaned: list[str] = []

    for smi in smiles_list:
        result = clean_smiles(
            smi,
            remove_stereo=remove_stereo,
            neutralize=neutralize,
            keep_largest=keep_largest,
            max_length=max_length,
        )
        if result and result not in seen:
            seen.add(result)
            cleaned.append(result)

    logger.info(
        "Preprocessing: %d → %d SMILES (%d dropped).",
        len(smiles_list),
        len(cleaned),
        len(smiles_list) - len(cleaned),
    )
    return cleaned


def augment_smiles(
    smiles: str,
    n_augmentations: int = 5,
) -> list[str]:
    """Generate multiple valid SMILES for the same molecule.

    RDKit can produce different SMILES by randomizing the atom
    traversal order (``doRandom=True``).  Each variant is syntactically
    different but chemically identical.

    This is a powerful data augmentation strategy because it teaches
    the model that many strings map to the same molecule, improving
    generalization.

    Parameters
    ----------
    smiles : str
        Canonical SMILES.
    n_augmentations : int
        Number of random SMILES to generate.

    Returns
    -------
    list[str]
        List of augmented SMILES (may include the original).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]

    variants: set[str] = {smiles}
    attempts = 0
    max_attempts = n_augmentations * 10

    while len(variants) < n_augmentations + 1 and attempts < max_attempts:
        randomized = Chem.MolToSmiles(mol, doRandom=True)
        variants.add(randomized)
        attempts += 1

    return list(variants)


def augment_dataset(
    smiles_list: list[str],
    n_augmentations: int = 5,
    max_total: Optional[int] = None,
) -> list[str]:
    """Augment a full dataset with randomized SMILES.

    Parameters
    ----------
    smiles_list : list[str]
        Clean SMILES to augment.
    n_augmentations : int
        Number of extra variants per molecule.
    max_total : int, optional
        Cap the total dataset size.

    Returns
    -------
    list[str]
        Augmented SMILES list (shuffled).
    """
    augmented: list[str] = []
    for smi in smiles_list:
        augmented.extend(augment_smiles(smi, n_augmentations))

    random.shuffle(augmented)

    if max_total and len(augmented) > max_total:
        augmented = augmented[:max_total]

    logger.info(
        "Augmentation: %d → %d SMILES (×%d).",
        len(smiles_list),
        len(augmented),
        n_augmentations,
    )
    return augmented
