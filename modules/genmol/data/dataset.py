"""
PyTorch Dataset for SMILES
===========================

Wraps tokenized SMILES into a ``torch.utils.data.Dataset`` with
train/validation/test splits and a custom collation function.

Design decisions
----------------
* **90 / 5 / 5 split** — standard for generative models where we
  don't need a large test set; validation is mainly for monitoring
  reconstruction loss.
* **max_len = 120** — covers >99 % of drug-like molecules (median
  SMILES length in ChEMBL is ~45 characters).
* **Collate with padding** — the tokenizer pads to ``max_length``,
  so all tensors in a batch have the same shape.  This is simpler
  than dynamic padding and only wastes a small amount of memory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .tokenizer import SmilesTokenizer

logger = logging.getLogger(__name__)


class SmilesDataset(Dataset):
    """A PyTorch Dataset of tokenized SMILES strings.

    Each item is a ``dict`` with:
    * ``input_ids`` — integer tensor of shape ``(max_length,)``
    * ``smiles`` — the original SMILES string (for evaluation)

    Parameters
    ----------
    smiles_list : list[str]
        Cleaned SMILES strings.
    tokenizer : SmilesTokenizer
        A tokenizer with a built vocabulary.
    """

    def __init__(
        self,
        smiles_list: list[str],
        tokenizer: SmilesTokenizer,
    ):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.smiles_list)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        smi = self.smiles_list[idx]
        ids = self.tokenizer.encode(smi, add_sos=True, add_eos=True, pad=True)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "smiles": smi,
        }


def split_dataset(
    dataset: SmilesDataset,
    train_frac: float = 0.90,
    val_frac: float = 0.05,
    seed: int = 42,
) -> tuple[Subset, Subset, Subset]:
    """Split a SmilesDataset into train / validation / test subsets.

    Uses a fixed random seed for reproducibility.

    Parameters
    ----------
    dataset : SmilesDataset
        The full dataset.
    train_frac : float
        Fraction for training (default 0.90).
    val_frac : float
        Fraction for validation (default 0.05).
        Test fraction = 1 − train − val.
    seed : int
        Random seed.

    Returns
    -------
    tuple[Subset, Subset, Subset]
        (train, val, test) subsets.
    """
    n = len(dataset)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    indices = indices.tolist()

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    logger.info(
        "Split: %d train / %d val / %d test.",
        len(train_idx),
        len(val_idx),
        len(test_idx),
    )

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


def collate_smiles(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    """Custom collate function for SMILES batches.

    Stacks ``input_ids`` tensors and collects SMILES strings.

    Parameters
    ----------
    batch : list[dict]
        List of items from ``SmilesDataset.__getitem__``.

    Returns
    -------
    dict
        ``{"input_ids": Tensor[B, L], "smiles": list[str]}``
    """
    input_ids = torch.stack([item["input_ids"] for item in batch])
    smiles = [item["smiles"] for item in batch]
    return {"input_ids": input_ids, "smiles": smiles}


def create_dataloaders(
    smiles_list: list[str],
    tokenizer: SmilesTokenizer,
    batch_size: int = 256,
    train_frac: float = 0.90,
    val_frac: float = 0.05,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """One-call factory to create train/val/test DataLoaders.

    Parameters
    ----------
    smiles_list : list[str]
        Cleaned SMILES.
    tokenizer : SmilesTokenizer
        Tokenizer with built vocabulary.
    batch_size : int
        Batch size (default 256).
    train_frac : float
        Training fraction.
    val_frac : float
        Validation fraction.
    seed : int
        Random seed for split.
    num_workers : int
        DataLoader workers (0 = main process).

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        (train_loader, val_loader, test_loader)
    """
    dataset = SmilesDataset(smiles_list, tokenizer)
    train_ds, val_ds, test_ds = split_dataset(
        dataset, train_frac, val_frac, seed
    )

    common = dict(
        collate_fn=collate_smiles,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, **common
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, **common
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, **common
    )

    logger.info(
        "DataLoaders created: batch_size=%d, train=%d batches, "
        "val=%d batches, test=%d batches.",
        batch_size,
        len(train_loader),
        len(val_loader),
        len(test_loader),
    )

    return train_loader, val_loader, test_loader
