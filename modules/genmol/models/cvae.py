"""
Conditional VAE (CVAE)
=======================

Extends ``MolVAE`` to condition generation on desired molecular
properties — molecular weight, LogP, and QED.

Why conditional generation?
---------------------------
A standard VAE samples ``z ~ N(0, I)`` and decodes whatever comes
out.  A **Conditional VAE** (CVAE) concatenates a property vector
``c`` to both the encoder input and the decoder's latent vector:

::

    Encoder:  (x, c) → μ, log σ²
    Decoder:  (z, c) → x̂

This teaches the model to *disentangle* molecular structure from
properties.  At generation time, you specify the desired properties
``c`` and the model generates molecules that satisfy them.

Property conditioning
---------------------
We condition on three scalar properties:

1. **Molecular weight** (MW) — controls molecular size.
2. **LogP** — controls lipophilicity (membrane permeability).
3. **QED** — quantitative estimate of drug-likeness (0–1 composite
   score combining MW, LogP, HBD, HBA, PSA, RotBonds, Alerts, AROM).

Properties are **normalized** to [0, 1] using min-max scaling based
on the training set statistics, so the model sees balanced inputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae import MolEncoder, MolDecoder, MolVAE, vae_loss


class PropertyNormalizer:
    """Min-max normalizer for molecular properties.

    Stores per-property (min, max) learned from training data
    and maps values to [0, 1].

    Parameters
    ----------
    property_names : list[str]
        Names of properties to track (e.g. ``["mw", "logp", "qed"]``).
    """

    def __init__(self, property_names: list[str] | None = None):
        self.property_names = property_names or ["mw", "logp", "qed"]
        self.mins: dict[str, float] = {}
        self.maxs: dict[str, float] = {}
        self._fitted = False

    def fit(self, property_values: dict[str, list[float]]) -> None:
        """Learn min/max from training data.

        Parameters
        ----------
        property_values : dict
            ``{property_name: [values…]}``
        """
        for name in self.property_names:
            vals = property_values[name]
            self.mins[name] = min(vals)
            self.maxs[name] = max(vals)
        self._fitted = True

    def transform(self, values: dict[str, float]) -> list[float]:
        """Normalize a single sample's properties to [0, 1].

        Parameters
        ----------
        values : dict
            ``{property_name: scalar_value}``

        Returns
        -------
        list[float]
            Normalized values in the order of ``property_names``.
        """
        assert self._fitted, "Call fit() first."
        result = []
        for name in self.property_names:
            lo, hi = self.mins[name], self.maxs[name]
            if hi - lo < 1e-8:
                result.append(0.5)
            else:
                result.append((values[name] - lo) / (hi - lo))
        return result

    def inverse_transform(self, normalized: list[float]) -> dict[str, float]:
        """Convert normalized values back to original scale.

        Parameters
        ----------
        normalized : list[float]
            Values in [0, 1].

        Returns
        -------
        dict[str, float]
            Original-scale property values.
        """
        assert self._fitted, "Call fit() first."
        result = {}
        for name, val in zip(self.property_names, normalized):
            lo, hi = self.mins[name], self.maxs[name]
            result[name] = val * (hi - lo) + lo
        return result

    def save(self, path: str | Path) -> None:
        """Save normalization parameters to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "property_names": self.property_names,
            "mins": self.mins,
            "maxs": self.maxs,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "PropertyNormalizer":
        """Load normalization parameters from JSON."""
        data = json.loads(Path(path).read_text())
        norm = cls(data["property_names"])
        norm.mins = data["mins"]
        norm.maxs = data["maxs"]
        norm._fitted = True
        return norm


class ConditionalMolEncoder(nn.Module):
    """Encoder that concatenates condition vector to the embedding.

    The condition vector ``c`` is repeated at every time step and
    concatenated with the token embedding before the GRU.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    cond_dim : int
        Dimension of the condition vector (default 3 for MW, LogP, QED).
    embed_dim : int
        Token embedding dimension.
    hidden_dim : int
        GRU hidden dimension.
    latent_dim : int
        Latent space dimension.
    num_layers : int
        Number of GRU layers.
    dropout : float
        Dropout rate.
    pad_idx : int
        Padding token index.
    """

    def __init__(
        self,
        vocab_size: int,
        cond_dim: int = 3,
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

        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )
        self.gru = nn.GRU(
            embed_dim + cond_dim,  # condition concatenated
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode (tokens, condition) → (μ, log σ²).

        Parameters
        ----------
        x : Tensor[B, L]
            Token IDs.
        cond : Tensor[B, cond_dim]
            Condition vector.

        Returns
        -------
        mu, logvar : Tensor[B, latent_dim]
        """
        embedded = self.embedding(x)  # [B, L, E]
        # Repeat condition across sequence length
        cond_expanded = cond.unsqueeze(1).expand(-1, x.size(1), -1)  # [B,L,C]
        gru_input = torch.cat([embedded, cond_expanded], dim=-1)  # [B,L,E+C]

        _, hidden = self.gru(gru_input)
        h_cat = torch.cat([hidden[-2], hidden[-1]], dim=-1)

        return self.fc_mu(h_cat), self.fc_logvar(h_cat)


class ConditionalMolDecoder(nn.Module):
    """Decoder that conditions on both z and a property vector.

    The condition vector ``c`` is concatenated with the latent ``z``
    for hidden-state initialization and with the token embedding at
    each decoding step.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    cond_dim : int
        Condition vector dimension.
    embed_dim : int
        Token embedding dimension.
    hidden_dim : int
        GRU hidden dimension.
    latent_dim : int
        Latent space dimension.
    num_layers : int
        GRU layers.
    dropout : float
        Dropout.
    max_length : int
        Max sequence length.
    pad_idx : int
        Pad token index.
    """

    def __init__(
        self,
        vocab_size: int,
        cond_dim: int = 3,
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
        self.z_to_hidden = nn.Linear(
            latent_dim + cond_dim, hidden_dim * num_layers
        )
        self.gru = nn.GRU(
            embed_dim + cond_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_dim, vocab_size)

    def _init_hidden(
        self, z: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """Initialize hidden from (z, condition)."""
        zc = torch.cat([z, cond], dim=-1)
        h = self.z_to_hidden(zc)
        h = h.view(-1, self.num_layers, self.hidden_dim)
        h = h.permute(1, 0, 2).contiguous()
        return torch.tanh(h)

    def forward(
        self,
        z: torch.Tensor,
        cond: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode (z, condition) → logits.

        Parameters
        ----------
        z : Tensor[B, latent_dim]
        cond : Tensor[B, cond_dim]
        target : Tensor[B, L], optional
            Teacher forcing targets.

        Returns
        -------
        Tensor[B, L-1, vocab_size]
        """
        hidden = self._init_hidden(z, cond)
        batch_size = z.size(0)

        if target is not None:
            embedded = self.embedding(target[:, :-1])  # [B, L-1, E]
            seq_len = embedded.size(1)
            cond_exp = cond.unsqueeze(1).expand(-1, seq_len, -1)
            gru_input = torch.cat([embedded, cond_exp], dim=-1)
            output, _ = self.gru(gru_input, hidden)
            return self.output_proj(output)

        # Autoregressive
        device = z.device
        input_tok = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        outputs: list[torch.Tensor] = []

        for _ in range(self.max_length - 1):
            embedded = self.embedding(input_tok)
            cond_step = cond.unsqueeze(1)
            gru_input = torch.cat([embedded, cond_step], dim=-1)
            output, hidden = self.gru(gru_input, hidden)
            logits = self.output_proj(output)
            outputs.append(logits)
            input_tok = logits.argmax(dim=-1)

        return torch.cat(outputs, dim=1)


class MolCVAE(nn.Module):
    """Conditional Molecular VAE.

    Extends ``MolVAE`` with property-conditioned generation.

    Parameters
    ----------
    vocab_size : int
        Vocabulary size.
    cond_dim : int
        Number of conditioning properties (default 3).
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
        Max sequence length.
    pad_idx : int
        Padding token index.
    """

    def __init__(
        self,
        vocab_size: int,
        cond_dim: int = 3,
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
        self.cond_dim = cond_dim
        self.max_length = max_length
        self.pad_idx = pad_idx

        self.encoder = ConditionalMolEncoder(
            vocab_size, cond_dim, embed_dim, hidden_dim,
            latent_dim, num_layers, dropout, pad_idx,
        )
        self.decoder = ConditionalMolDecoder(
            vocab_size, cond_dim, embed_dim, hidden_dim,
            latent_dim, num_layers, dropout, max_length, pad_idx,
        )

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with condition.

        Parameters
        ----------
        x : Tensor[B, L]
            Input token IDs.
        cond : Tensor[B, cond_dim]
            Normalized property conditions.
        target : Tensor[B, L], optional
            Teacher forcing targets.

        Returns
        -------
        logits, mu, logvar
        """
        mu, logvar = self.encoder(x, cond)
        z = self.reparameterize(mu, logvar)
        target_seq = target if target is not None else x
        logits = self.decoder(z, cond, target_seq)
        return logits, mu, logvar

    def sample(
        self,
        cond: torch.Tensor,
        n: int = 1,
        temperature: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Generate molecules conditioned on properties.

        Parameters
        ----------
        cond : Tensor[n, cond_dim] or Tensor[cond_dim]
            Desired property conditions (normalized to [0,1]).
        n : int
            Number to generate (only used if cond is 1D).
        temperature : float
            Decoding temperature.
        device : torch.device, optional
            Tensor device.

        Returns
        -------
        Tensor[n, L-1]
            Generated token sequences.
        """
        if device is None:
            device = next(self.parameters()).device

        if cond.dim() == 1:
            cond = cond.unsqueeze(0).expand(n, -1)

        cond = cond.to(device)
        z = torch.randn(cond.size(0), self.latent_dim, device=device)

        self.eval()
        with torch.no_grad():
            hidden = self.decoder._init_hidden(z, cond)
            batch_size = z.size(0)

            input_tok = torch.ones(
                batch_size, 1, dtype=torch.long, device=device
            )
            generated: list[torch.Tensor] = []

            for _ in range(self.max_length - 1):
                embedded = self.decoder.embedding(input_tok)
                cond_step = cond.unsqueeze(1)
                gru_input = torch.cat([embedded, cond_step], dim=-1)
                output, hidden = self.decoder.gru(gru_input, hidden)
                logits = self.decoder.output_proj(output)

                probs = F.softmax(logits.squeeze(1) / temperature, dim=-1)
                input_tok = torch.multinomial(probs, 1)
                generated.append(input_tok)

            return torch.cat(generated, dim=1)
