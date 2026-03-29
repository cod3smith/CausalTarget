"""Tests for the Molecular VAE."""

import pytest
import torch

from modules.genmol.data.tokenizer import SmilesTokenizer
from modules.genmol.models.vae import MolVAE, MolEncoder, MolDecoder, vae_loss


# ── Fixtures ────────────────────────────────────────────────────────
SAMPLE_SMILES = [
    "CCO",
    "c1ccccc1",
    "CC(=O)O",
    "CC(=O)NC1=CC=C(O)C=C1",
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
]


@pytest.fixture
def tokenizer():
    tok = SmilesTokenizer(max_length=60)
    tok.build_vocab(SAMPLE_SMILES)
    return tok


@pytest.fixture
def model(tokenizer):
    return MolVAE(
        vocab_size=tokenizer.vocab_size,
        embed_dim=32,
        hidden_dim=64,
        latent_dim=16,
        num_layers=2,
        dropout=0.0,
        max_length=60,
        pad_idx=tokenizer.pad_idx,
    )


@pytest.fixture
def batch(tokenizer):
    ids = [tokenizer.encode(s) for s in SAMPLE_SMILES]
    return torch.tensor(ids, dtype=torch.long)


class TestEncoder:
    """Test the GRU encoder."""

    def test_output_shapes(self, tokenizer, batch):
        encoder = MolEncoder(
            vocab_size=tokenizer.vocab_size,
            embed_dim=32,
            hidden_dim=64,
            latent_dim=16,
            num_layers=2,
            pad_idx=tokenizer.pad_idx,
        )
        mu, logvar = encoder(batch)
        assert mu.shape == (len(SAMPLE_SMILES), 16)
        assert logvar.shape == (len(SAMPLE_SMILES), 16)


class TestDecoder:
    """Test the GRU decoder."""

    def test_teacher_forcing(self, tokenizer, batch):
        decoder = MolDecoder(
            vocab_size=tokenizer.vocab_size,
            embed_dim=32,
            hidden_dim=64,
            latent_dim=16,
            num_layers=2,
            max_length=60,
            pad_idx=tokenizer.pad_idx,
        )
        z = torch.randn(len(SAMPLE_SMILES), 16)
        logits = decoder(z, target=batch)
        # logits shape: [B, L-1, V]
        assert logits.shape == (
            len(SAMPLE_SMILES),
            59,  # max_length - 1
            tokenizer.vocab_size,
        )

    def test_autoregressive(self, tokenizer):
        decoder = MolDecoder(
            vocab_size=tokenizer.vocab_size,
            embed_dim=32,
            hidden_dim=64,
            latent_dim=16,
            num_layers=2,
            max_length=60,
            pad_idx=tokenizer.pad_idx,
        )
        z = torch.randn(2, 16)
        logits = decoder(z, target=None)
        assert logits.shape == (2, 59, tokenizer.vocab_size)


class TestVAE:
    """Test the full VAE pipeline."""

    def test_forward(self, model, batch):
        model.train()
        logits, mu, logvar = model(batch)
        assert logits.shape[0] == len(SAMPLE_SMILES)
        assert logits.shape[2] == model.encoder.embedding.num_embeddings
        assert mu.shape == (len(SAMPLE_SMILES), 16)

    def test_encode(self, model, batch):
        model.eval()
        z = model.encode(batch)
        assert z.shape == (len(SAMPLE_SMILES), 16)

    def test_decode(self, model):
        model.eval()
        z = torch.randn(3, 16)
        token_ids = model.decode(z, greedy=True)
        assert token_ids.shape == (3, 59)

    def test_sample(self, model):
        model.eval()
        token_ids = model.sample(n=5, temperature=1.0)
        assert token_ids.shape[0] == 5
        assert token_ids.shape[1] == 59

    def test_reparameterize_train(self, model):
        model.train()
        mu = torch.zeros(4, 16)
        logvar = torch.zeros(4, 16)
        z = model.reparameterize(mu, logvar)
        # In training, noise is added so z ≠ mu
        assert z.shape == (4, 16)

    def test_reparameterize_eval(self, model):
        model.eval()
        mu = torch.zeros(4, 16)
        logvar = torch.zeros(4, 16)
        z = model.reparameterize(mu, logvar)
        # In eval, z == mu (no noise)
        assert torch.allclose(z, mu)


class TestVAELoss:
    """Test the β-VAE loss function."""

    def test_loss_shapes(self, model, batch):
        model.train()
        logits, mu, logvar = model(batch)
        total, recon, kl = vae_loss(
            logits, batch, mu, logvar, beta=1.0, pad_idx=model.pad_idx
        )
        assert total.dim() == 0  # scalar
        assert recon.dim() == 0
        assert kl.dim() == 0

    def test_kl_zero_for_standard_normal(self):
        mu = torch.zeros(10, 16)
        logvar = torch.zeros(10, 16)
        logits = torch.randn(10, 59, 30)
        target = torch.randint(0, 30, (10, 60))
        _, _, kl = vae_loss(logits, target, mu, logvar)
        assert kl.item() < 1e-5  # KL should be ~0

    def test_beta_scales_kl(self, model, batch):
        model.train()
        logits, mu, logvar = model(batch)
        t1, _, _ = vae_loss(logits, batch, mu, logvar, beta=0.01)
        t2, _, _ = vae_loss(logits, batch, mu, logvar, beta=1.0)
        # Higher β → higher total loss (assuming KL > 0)
        assert t2.item() >= t1.item() - 1e-6
