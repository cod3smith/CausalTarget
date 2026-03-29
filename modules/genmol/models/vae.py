"""
Molecular VAE (Variational Autoencoder)
========================================

A β-VAE that learns a continuous latent representation of drug-like
molecules from their SMILES strings.

Architecture overview
---------------------
::

    SMILES ──► Encoder (bidirectional GRU) ──► μ, log σ²
                                                   │
                                              z ~ N(μ, σ²)
                                                   │
              Decoder (autoregressive GRU) ◄───────┘
                     │
                 SMILES (reconstructed)

Why a VAE for molecules?
------------------------
* **Smooth latent space** — nearby points decode to similar molecules,
  enabling *interpolation* between drugs and *optimisation* by gradient
  descent in latent space.
* **Generative** — sampling ``z ~ N(0, I)`` produces novel, valid
  molecules.
* **Disentangled** — β > 1 encourages disentangled representations
  where individual latent dimensions correspond to molecular properties.

Key design choices
------------------
* **GRU over LSTM** — fewer parameters, trains faster on SMILES
  (sequence lengths < 120 don't need LSTM's extra gates).
* **3-layer bidirectional encoder** — captures both forward and
  backward context in the SMILES string.
* **Teacher forcing** — during training, the decoder receives the
  ground-truth previous token instead of its own prediction.
  This stabilises early training and speeds convergence.
* **KL annealing** — the β weight on the KL divergence is linearly
  increased from 0.01 → 1.0 over the first 10 epochs.  Without
  annealing, the model can fall into "posterior collapse" where the
  encoder ignores the input and the decoder becomes an unconditional
  language model.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MolEncoder(nn.Module):
    """Bidirectional GRU encoder that maps token sequences to (μ, log σ²).

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary.
    embed_dim : int
        Embedding dimension (default 64).
    hidden_dim : int
        GRU hidden dimension (default 256).
    latent_dim : int
        Dimension of the latent vector *z* (default 128).
    num_layers : int
        Number of stacked GRU layers (default 3).
    dropout : float
        Dropout between GRU layers (default 0.2).
    pad_idx : int
        Index of the ``<PAD>`` token for embedding masking (default 0).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.latent_dim = latent_dim

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Bidirectional → 2 × hidden_dim per layer
        # We concatenate final forward + backward hidden states
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode token IDs to (μ, log σ²).

        Parameters
        ----------
        x : Tensor[B, L]
            Batch of padded token ID sequences.

        Returns
        -------
        mu : Tensor[B, latent_dim]
        logvar : Tensor[B, latent_dim]
        """
        embedded = self.embedding(x)  # [B, L, embed_dim]
        _, hidden = self.gru(embedded)
        # hidden: [num_layers * 2, B, hidden_dim]

        # Take the last layer's forward and backward hidden states
        # Forward:  hidden[-2]  → [B, hidden_dim]
        # Backward: hidden[-1]  → [B, hidden_dim]
        h_cat = torch.cat([hidden[-2], hidden[-1]], dim=-1)  # [B, 2H]

        mu = self.fc_mu(h_cat)          # [B, latent_dim]
        logvar = self.fc_logvar(h_cat)  # [B, latent_dim]
        return mu, logvar


class MolDecoder(nn.Module):
    """Autoregressive GRU decoder that reconstructs SMILES from *z*.

    During training, uses **teacher forcing**: the ground-truth
    previous token is fed as input at each time step.

    During generation, uses its own predictions autoregressively.

    Parameters
    ----------
    vocab_size : int
        Number of tokens in the vocabulary.
    embed_dim : int
        Embedding dimension (default 64).
    hidden_dim : int
        GRU hidden dimension (default 256).
    latent_dim : int
        Dimension of the latent vector *z* (default 128).
    num_layers : int
        Number of stacked GRU layers (default 3).
    dropout : float
        Dropout between GRU layers (default 0.2).
    max_length : int
        Maximum sequence length for generation (default 120).
    pad_idx : int
        Index of the ``<PAD>`` token (default 0).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        max_length: int = 120,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_length = max_length

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )

        # z is projected and used to initialize the GRU hidden state
        self.z_to_hidden = nn.Linear(latent_dim, hidden_dim * num_layers)

        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def _init_hidden(self, z: torch.Tensor) -> torch.Tensor:
        """Convert latent vector *z* to initial GRU hidden state.

        Parameters
        ----------
        z : Tensor[B, latent_dim]

        Returns
        -------
        Tensor[num_layers, B, hidden_dim]
        """
        h = self.z_to_hidden(z)  # [B, hidden_dim * num_layers]
        h = h.view(-1, self.num_layers, self.hidden_dim)  # [B, L, H]
        h = h.permute(1, 0, 2).contiguous()  # [L, B, H]
        return torch.tanh(h)

    def forward(
        self,
        z: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode latent vector to logits over the vocabulary.

        Parameters
        ----------
        z : Tensor[B, latent_dim]
            Latent vectors.
        target : Tensor[B, L], optional
            Ground-truth token IDs for teacher forcing.
            If ``None``, decodes autoregressively (slower).

        Returns
        -------
        Tensor[B, L, vocab_size]
            Logits at each position.
        """
        hidden = self._init_hidden(z)
        batch_size = z.size(0)

        if target is not None:
            # Teacher forcing: feed the full target sequence
            # Input is target[:, :-1], targets for loss are target[:, 1:]
            embedded = self.embedding(target[:, :-1])  # [B, L-1, E]
            output, _ = self.gru(embedded, hidden)     # [B, L-1, H]
            logits = self.output_proj(output)           # [B, L-1, V]
            return logits

        # Autoregressive decoding (generation mode)
        # Start with <SOS> token (index 1)
        device = z.device
        input_tok = torch.ones(
            batch_size, 1, dtype=torch.long, device=device
        )  # <SOS>

        outputs: list[torch.Tensor] = []

        for _ in range(self.max_length - 1):
            embedded = self.embedding(input_tok)       # [B, 1, E]
            output, hidden = self.gru(embedded, hidden)  # [B, 1, H]
            logits = self.output_proj(output)           # [B, 1, V]
            outputs.append(logits)

            # Greedy: take argmax as next input
            input_tok = logits.argmax(dim=-1)           # [B, 1]

        return torch.cat(outputs, dim=1)  # [B, L-1, V]


class MolVAE(nn.Module):
    """Molecular Variational Autoencoder.

    Combines ``MolEncoder`` and ``MolDecoder`` with the
    reparameterization trick and β-weighted KL divergence.

    The **reparameterization trick** allows backpropagation through
    the stochastic sampling step:

    .. math::

        z = \\mu + \\sigma \\cdot \\epsilon, \\quad \\epsilon \\sim N(0, I)

    This makes the sampling differentiable — gradients flow through
    μ and σ, not through the random noise ε.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    embed_dim : int
        Embedding dimension.
    hidden_dim : int
        GRU hidden dimension.
    latent_dim : int
        Latent space dimension.
    num_layers : int
        GRU layers.
    dropout : float
        Dropout rate.
    max_length : int
        Maximum sequence length.
    pad_idx : int
        Padding token index.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        max_length: int = 120,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_length = max_length
        self.pad_idx = pad_idx

        self.encoder = MolEncoder(
            vocab_size, embed_dim, hidden_dim, latent_dim,
            num_layers, dropout, pad_idx,
        )
        self.decoder = MolDecoder(
            vocab_size, embed_dim, hidden_dim, latent_dim,
            num_layers, dropout, max_length, pad_idx,
        )

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Sample z using the reparameterization trick.

        Parameters
        ----------
        mu : Tensor[B, D]
            Mean of the approximate posterior.
        logvar : Tensor[B, D]
            Log-variance of the approximate posterior.

        Returns
        -------
        Tensor[B, D]
            Sampled latent vector.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu  # deterministic at eval time

    def forward(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: encode → sample → decode.

        Parameters
        ----------
        x : Tensor[B, L]
            Input token IDs.
        target : Tensor[B, L], optional
            Target token IDs for teacher forcing.
            If ``None``, uses ``x`` as target.

        Returns
        -------
        logits : Tensor[B, L-1, V]
            Reconstruction logits.
        mu : Tensor[B, D]
            Posterior mean.
        logvar : Tensor[B, D]
            Posterior log-variance.
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z, target if target is not None else x)
        return logits, mu, logvar

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to the latent mean (deterministic).

        Parameters
        ----------
        x : Tensor[B, L]
            Token IDs.

        Returns
        -------
        Tensor[B, D]
            Latent means.
        """
        mu, _ = self.encoder(x)
        return mu

    def decode(
        self,
        z: torch.Tensor,
        temperature: float = 1.0,
        greedy: bool = False,
    ) -> torch.Tensor:
        """Decode latent vectors to token sequences.

        Parameters
        ----------
        z : Tensor[B, D]
            Latent vectors.
        temperature : float
            Sampling temperature (lower = more conservative).
        greedy : bool
            If True, use argmax instead of sampling.

        Returns
        -------
        Tensor[B, L-1]
            Generated token ID sequences.
        """
        self.eval()
        with torch.no_grad():
            hidden = self.decoder._init_hidden(z)
            batch_size = z.size(0)
            device = z.device

            input_tok = torch.ones(
                batch_size, 1, dtype=torch.long, device=device
            )  # <SOS>

            generated: list[torch.Tensor] = []

            for _ in range(self.max_length - 1):
                embedded = self.decoder.embedding(input_tok)
                output, hidden = self.decoder.gru(embedded, hidden)
                logits = self.decoder.output_proj(output)  # [B, 1, V]

                if greedy:
                    input_tok = logits.argmax(dim=-1)
                else:
                    probs = F.softmax(logits.squeeze(1) / temperature, dim=-1)
                    input_tok = torch.multinomial(probs, 1)  # [B, 1]

                generated.append(input_tok)

            return torch.cat(generated, dim=1)  # [B, L-1]

    def sample(
        self,
        n: int = 1,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Sample from the prior ``N(0, I)`` and decode.

        Parameters
        ----------
        n : int
            Number of molecules to generate.
        temperature : float
            Decoding temperature.
        device : torch.device, optional
            Device for tensors.

        Returns
        -------
        Tensor[n, L-1]
            Generated token ID sequences.
        """
        if device is None:
            device = next(self.parameters()).device

        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z, temperature=temperature)


def vae_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    pad_idx: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute β-VAE loss = reconstruction + β · KL divergence.

    The reconstruction loss is cross-entropy over the vocabulary
    at each position, ignoring ``<PAD>`` tokens.

    The KL divergence measures how far the approximate posterior
    ``q(z|x)`` deviates from the prior ``p(z) = N(0, I)``:

    .. math::

        D_{KL}(q \\| p) = -\\frac{1}{2} \\sum_{j=1}^{D}
        \\left(1 + \\log \\sigma_j^2 - \\mu_j^2 - \\sigma_j^2\\right)

    Parameters
    ----------
    logits : Tensor[B, L-1, V]
        Decoder output logits.
    target : Tensor[B, L]
        Ground-truth token IDs (loss is computed against ``target[:, 1:]``).
    mu : Tensor[B, D]
        Posterior mean.
    logvar : Tensor[B, D]
        Posterior log-variance.
    beta : float
        KL weight (β in β-VAE).
    pad_idx : int
        Padding token index to ignore.

    Returns
    -------
    total_loss : Tensor
        β-VAE loss.
    recon_loss : Tensor
        Reconstruction (cross-entropy) loss.
    kl_loss : Tensor
        KL divergence.
    """
    # Target for reconstruction: shift by one (predict next token)
    target_shifted = target[:, 1:]  # [B, L-1]

    # Flatten for cross-entropy
    B, L, V = logits.shape
    recon_loss = F.cross_entropy(
        logits.reshape(B * L, V),
        target_shifted.reshape(B * L),
        ignore_index=pad_idx,
        reduction="mean",
    )

    # KL divergence: analytical formula for Gaussian → N(0,I)
    kl_loss = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    )

    total_loss = recon_loss + beta * kl_loss
    return total_loss, recon_loss, kl_loss
