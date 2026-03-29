"""Tests for molecule generation and screening."""

import pytest
import torch

from modules.genmol.data.tokenizer import SmilesTokenizer
from modules.genmol.models.vae import MolVAE
from modules.genmol.generate import (
    generate,
    interpolate,
    screen_generated,
    _validate_smiles,
)


TRAIN_SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)O",
    "CC(=O)NC1=CC=C(O)C=C1",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "OC(=O)C1=CC=CC=C1O",
    "CC(=O)OC1=CC=CC=C1C(O)=O",
    "C1CCCCC1",
    "CC=CC",
    "CCN(CC)CC",
]


@pytest.fixture
def tokenizer():
    tok = SmilesTokenizer(max_length=60)
    tok.build_vocab(TRAIN_SMILES)
    return tok


@pytest.fixture
def model(tokenizer):
    m = MolVAE(
        vocab_size=tokenizer.vocab_size,
        embed_dim=32,
        hidden_dim=64,
        latent_dim=16,
        num_layers=2,
        dropout=0.0,
        max_length=60,
        pad_idx=tokenizer.pad_idx,
    )
    m.eval()
    return m


class TestValidation:
    """Test SMILES validation."""

    def test_valid_smiles(self):
        valid = _validate_smiles(["CCO", "c1ccccc1", "CC(=O)O"])
        assert len(valid) == 3

    def test_invalid_smiles_filtered(self):
        valid = _validate_smiles(["CCO", "INVALID", "XX!!"])
        assert len(valid) == 1
        assert valid[0] == "CCO"

    def test_duplicates_removed(self):
        valid = _validate_smiles(["CCO", "OCC", "CCO"])
        # CCO and OCC are the same molecule canonicalized
        assert len(valid) == 1

    def test_empty_strings(self):
        valid = _validate_smiles(["", "", "CCO"])
        assert len(valid) == 1


class TestGeneration:
    """Test molecule generation from VAE."""

    def test_generate_returns_list(self, model, tokenizer):
        # Small untrained model — may not produce valid SMILES
        # but should return a list
        result = generate(
            model, tokenizer, n=10, temperature=1.0, validate=False
        )
        assert isinstance(result, list)
        assert len(result) <= 10

    def test_generate_with_validation(self, model, tokenizer):
        result = generate(
            model, tokenizer, n=20, temperature=1.0, validate=True
        )
        assert isinstance(result, list)
        # All returned should be valid
        from rdkit import Chem
        for smi in result:
            assert Chem.MolFromSmiles(smi) is not None


class TestInterpolation:
    """Test latent space interpolation."""

    def test_interpolate_length(self, model, tokenizer):
        result = interpolate(
            model, tokenizer, "CCO", "c1ccccc1", n_steps=5
        )
        assert len(result) == 5

    def test_interpolate_returns_strings(self, model, tokenizer):
        result = interpolate(
            model, tokenizer, "CCO", "CC(=O)O", n_steps=3
        )
        assert all(isinstance(s, str) for s in result)


class TestScreening:
    """Test MolScreen integration."""

    def test_screen_valid_molecules(self):
        smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
        results = screen_generated(smiles, use_molscreen=True)
        # At least some should pass
        assert isinstance(results, list)

    def test_screen_no_molscreen(self):
        smiles = ["CCO", "c1ccccc1"]
        results = screen_generated(smiles, use_molscreen=False)
        assert len(results) == 2
        assert all("smiles" in r for r in results)
