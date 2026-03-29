"""
VAE Training Loop
==================

Handles the full training pipeline: KL annealing, gradient clipping,
learning-rate scheduling, mixed-precision training, and checkpointing.

Training strategy
-----------------
* **KL annealing** — the β weight on the KL divergence is linearly
  increased from ``beta_start`` (0.01) to ``beta_end`` (1.0) over the
  first ``anneal_epochs`` (10) epochs.  This prevents *posterior
  collapse*, where the encoder learns to ignore the input and the
  KL term drops to zero.  By starting with a low β, the model first
  learns good reconstructions, then gradually learns a structured
  latent space.

* **Gradient clipping** — max-norm clipping at 1.0 prevents exploding
  gradients, which are common in GRU-based models on long sequences.

* **ReduceLROnPlateau** — if validation loss doesn't improve for
  ``patience`` epochs, the learning rate is reduced by a factor of 0.5.

* **Mixed precision** — ``torch.amp`` on CUDA/MPS accelerates training
  by ~2× on modern GPUs with TF32/FP16 compute.

* **Checkpoints** — model + optimizer state saved every ``save_every``
  epochs and at the best validation loss.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from .models.vae import MolVAE, vae_loss
from .models.cvae import MolCVAE
from .data.tokenizer import SmilesTokenizer

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training hyperparameters.

    All defaults are tuned for drug-like molecule generation on
    a single GPU.
    """

    # Optimiser
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    # KL annealing
    beta_start: float = 0.01
    beta_end: float = 1.0
    anneal_epochs: int = 10

    # Schedule
    epochs: int = 100
    patience: int = 5
    lr_factor: float = 0.5

    # Checkpointing
    save_every: int = 5
    checkpoint_dir: str = "checkpoints/genmol"

    # Mixed precision
    use_amp: bool = True

    # Logging
    log_every: int = 100  # log every N batches

    def to_dict(self) -> dict:
        return asdict(self)


def _get_beta(epoch: int, config: TrainConfig) -> float:
    """Compute the current β for KL annealing.

    Linear warmup from ``beta_start`` to ``beta_end`` over
    the first ``anneal_epochs`` epochs, then constant.
    """
    if epoch >= config.anneal_epochs:
        return config.beta_end
    frac = epoch / max(config.anneal_epochs, 1)
    return config.beta_start + (config.beta_end - config.beta_start) * frac


def _get_device() -> torch.device:
    """Pick the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_vae(
    model: MolVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[TrainConfig] = None,
    tokenizer: Optional[SmilesTokenizer] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Train a MolVAE model.

    Parameters
    ----------
    model : MolVAE
        The VAE to train.
    train_loader : DataLoader
        Training data.
    val_loader : DataLoader
        Validation data (for LR scheduling and early stopping).
    config : TrainConfig, optional
        Hyperparameters.  Defaults are sensible.
    tokenizer : SmilesTokenizer, optional
        Saved alongside checkpoints for later generation.
    device : torch.device, optional
        Training device.

    Returns
    -------
    dict
        Training history with keys: ``train_loss``, ``val_loss``,
        ``train_recon``, ``train_kl``, ``val_recon``, ``val_kl``,
        ``beta``, ``lr``.
    """
    if config is None:
        config = TrainConfig()
    if device is None:
        device = _get_device()

    logger.info("Training on %s.", device)
    model = model.to(device)

    optimizer = Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_factor,
        patience=config.patience,
        verbose=True,
    )

    # Mixed precision
    use_amp = config.use_amp and device.type in ("cuda",)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Checkpoint directory
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = ckpt_dir / "config.json"
    config_path.write_text(json.dumps(config.to_dict(), indent=2))

    # Save tokenizer if provided
    if tokenizer is not None:
        tokenizer.save(ckpt_dir / "tokenizer.json")

    # History
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_recon": [],
        "train_kl": [],
        "val_recon": [],
        "val_kl": [],
        "beta": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    pad_idx = model.pad_idx

    for epoch in range(config.epochs):
        t0 = time.time()
        beta = _get_beta(epoch, config)

        # ── Training ────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            x = batch["input_ids"].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device.type, enabled=use_amp):
                logits, mu, logvar = model(x)
                loss, recon, kl = vae_loss(
                    logits, x, mu, logvar, beta=beta, pad_idx=pad_idx
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()
            n_batches += 1

            if (batch_idx + 1) % config.log_every == 0:
                logger.info(
                    "  [%d/%d] batch %d: loss=%.4f recon=%.4f kl=%.4f β=%.4f",
                    epoch + 1,
                    config.epochs,
                    batch_idx + 1,
                    loss.item(),
                    recon.item(),
                    kl.item(),
                    beta,
                )

        avg_train = epoch_loss / max(n_batches, 1)
        avg_recon = epoch_recon / max(n_batches, 1)
        avg_kl = epoch_kl / max(n_batches, 1)

        # ── Validation ──────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_ids"].to(device)
                logits, mu, logvar = model(x)
                loss, recon, kl = vae_loss(
                    logits, x, mu, logvar, beta=beta, pad_idx=pad_idx
                )
                val_loss += loss.item()
                val_recon += recon.item()
                val_kl += kl.item()
                val_batches += 1

        avg_val = val_loss / max(val_batches, 1)
        avg_val_recon = val_recon / max(val_batches, 1)
        avg_val_kl = val_kl / max(val_batches, 1)

        # LR scheduling
        scheduler.step(avg_val)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d (%.1fs): train=%.4f val=%.4f "
            "recon=%.4f/%.4f kl=%.4f/%.4f β=%.4f lr=%.2e",
            epoch + 1,
            config.epochs,
            elapsed,
            avg_train,
            avg_val,
            avg_recon,
            avg_val_recon,
            avg_kl,
            avg_val_kl,
            beta,
            current_lr,
        )

        # ── Record history ──────────────────────────────────────
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["train_recon"].append(avg_recon)
        history["train_kl"].append(avg_kl)
        history["val_recon"].append(avg_val_recon)
        history["val_kl"].append(avg_val_kl)
        history["beta"].append(beta)
        history["lr"].append(current_lr)

        # ── Checkpoints ─────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val,
                    "config": config.to_dict(),
                },
                ckpt_dir / "best_model.pt",
            )
            logger.info("  ★ New best model saved (val_loss=%.4f).", avg_val)

        if (epoch + 1) % config.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val,
                    "config": config.to_dict(),
                },
                ckpt_dir / f"checkpoint_epoch_{epoch + 1}.pt",
            )

    # Save final model
    torch.save(
        {
            "epoch": config.epochs - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val,
            "config": config.to_dict(),
        },
        ckpt_dir / "final_model.pt",
    )

    # Save training history
    history_path = ckpt_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))
    logger.info("Training complete. History → %s", history_path)

    return history


def load_checkpoint(
    checkpoint_path: str | Path,
    model: MolVAE,
    optimizer: Optional[Adam] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Load a training checkpoint.

    Parameters
    ----------
    checkpoint_path : str or Path
        Path to ``.pt`` checkpoint file.
    model : MolVAE
        Model to load weights into.
    optimizer : Adam, optional
        Optimizer to restore state.
    device : torch.device, optional
        Map location for loading.

    Returns
    -------
    dict
        Checkpoint metadata (epoch, val_loss, config).
    """
    if device is None:
        device = _get_device()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    logger.info(
        "Loaded checkpoint from epoch %d (val_loss=%.4f).",
        ckpt.get("epoch", -1),
        ckpt.get("val_loss", float("inf")),
    )

    return {
        "epoch": ckpt.get("epoch", -1),
        "val_loss": ckpt.get("val_loss"),
        "config": ckpt.get("config"),
    }
