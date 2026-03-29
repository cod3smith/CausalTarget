"""
Visualisation Utilities
========================

Plotting functions for evaluating generative model performance,
including property distribution histograms, latent space maps,
training curves, and interpolation visualisations.

All plots use the NeoForge colour palette:

* Deep Navy ``#0D1B2A`` — primary background/text
* Neon Teal ``#00D4AA`` — accent/highlight
* Supporting colours for multi-series plots
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── NeoForge palette ────────────────────────────────────────────────
DEEP_NAVY = "#0D1B2A"
NEON_TEAL = "#00D4AA"
PALETTE = [NEON_TEAL, "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]


def _setup_style():
    """Apply NeoForge-inspired matplotlib style."""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#F8F9FA",
        "axes.edgecolor": DEEP_NAVY,
        "axes.labelcolor": DEEP_NAVY,
        "text.color": DEEP_NAVY,
        "xtick.color": DEEP_NAVY,
        "ytick.color": DEEP_NAVY,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
    })


def plot_training_curves(
    history: dict[str, list[float]],
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot training/validation loss curves and KL annealing schedule.

    Parameters
    ----------
    history : dict
        Training history from ``train_vae()``.
    save_path : str or Path, optional
        If given, save the figure instead of showing.
    """
    import matplotlib.pyplot as plt

    _setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Total loss
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], color=NEON_TEAL, label="Train")
    ax.plot(epochs, history["val_loss"], color="#FF6B6B", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss (Recon + β·KL)")
    ax.legend()

    # Reconstruction loss
    ax = axes[0, 1]
    ax.plot(epochs, history["train_recon"], color=NEON_TEAL, label="Train")
    ax.plot(epochs, history["val_recon"], color="#FF6B6B", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction Loss")
    ax.set_title("Reconstruction Loss (CE)")
    ax.legend()

    # KL divergence
    ax = axes[1, 0]
    ax.plot(epochs, history["train_kl"], color=NEON_TEAL, label="Train")
    ax.plot(epochs, history["val_kl"], color="#FF6B6B", label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("KL Divergence")
    ax.set_title("KL Divergence")
    ax.legend()

    # Beta schedule + LR
    ax = axes[1, 1]
    ax.plot(epochs, history["beta"], color=NEON_TEAL, label="β")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("β", color=NEON_TEAL)
    ax.tick_params(axis="y", labelcolor=NEON_TEAL)
    ax.set_title("KL Annealing β & Learning Rate")

    ax2 = ax.twinx()
    ax2.plot(epochs, history["lr"], color="#FF6B6B", label="LR", linestyle="--")
    ax2.set_ylabel("Learning Rate", color="#FF6B6B")
    ax2.tick_params(axis="y", labelcolor="#FF6B6B")

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Training curves → %s", save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_property_distributions(
    generated_smiles: list[str],
    reference_smiles: Optional[list[str]] = None,
    properties: Optional[list[str]] = None,
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot property distribution histograms.

    Overlays generated vs. reference (training) distributions
    for selected molecular properties.

    Parameters
    ----------
    generated_smiles : list[str]
        Generated SMILES.
    reference_smiles : list[str], optional
        Reference (training) SMILES for comparison.
    properties : list[str], optional
        Which properties to plot (default: all).
    save_path : str or Path, optional
        Save path for the figure.
    """
    import matplotlib.pyplot as plt
    from .distribution import compute_properties

    _setup_style()

    gen_props = compute_properties(generated_smiles)
    ref_props = compute_properties(reference_smiles) if reference_smiles else None

    if properties is None:
        properties = list(gen_props.keys())

    n = len(properties)
    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, prop in enumerate(properties):
        ax = axes[i]
        gen_vals = gen_props.get(prop, [])

        if gen_vals:
            ax.hist(
                gen_vals,
                bins=50,
                alpha=0.7,
                color=NEON_TEAL,
                label="Generated",
                density=True,
            )

        if ref_props and prop in ref_props and ref_props[prop]:
            ax.hist(
                ref_props[prop],
                bins=50,
                alpha=0.5,
                color=DEEP_NAVY,
                label="Reference",
                density=True,
            )

        ax.set_title(prop.upper())
        ax.legend(fontsize=8)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Property Distributions", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Property distributions → %s", save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_latent_space(
    latent_vectors: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = "tsne",
    save_path: Optional[str | Path] = None,
) -> None:
    """2D visualisation of the latent space using t-SNE or UMAP.

    Parameters
    ----------
    latent_vectors : ndarray [N, D]
        Latent representations (from ``model.encode()``).
    labels : ndarray [N], optional
        Colour labels (e.g. a property value for colouring points).
    method : str
        Dimensionality reduction method: ``"tsne"`` or ``"umap"``.
    save_path : str or Path, optional
        Save path.
    """
    import matplotlib.pyplot as plt

    _setup_style()

    if method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        coords = reducer.fit_transform(latent_vectors)
    elif method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            coords = reducer.fit_transform(latent_vectors)
        except ImportError:
            logger.warning("umap-learn not installed, falling back to t-SNE.")
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            coords = reducer.fit_transform(latent_vectors)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'tsne' or 'umap'.")

    fig, ax = plt.subplots(figsize=(8, 6))

    if labels is not None:
        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=labels,
            cmap="viridis",
            s=5,
            alpha=0.6,
        )
        plt.colorbar(scatter, ax=ax, label="Property value")
    else:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            color=NEON_TEAL,
            s=5,
            alpha=0.5,
        )

    ax.set_title(f"Latent Space ({method.upper()})", fontsize=14)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Latent space plot → %s", save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_interpolation(
    smiles_list: list[str],
    save_path: Optional[str | Path] = None,
) -> None:
    """Visualise a latent-space interpolation as a molecule grid.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES from ``generate.interpolate()``.
    save_path : str or Path, optional
        Save path.
    """
    import matplotlib.pyplot as plt
    from rdkit import Chem
    from rdkit.Chem import Draw

    _setup_style()

    mols = []
    labels = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mols.append(mol)
            labels.append(f"Step {i}")
        else:
            labels.append(f"Step {i} (invalid)")

    if not mols:
        logger.warning("No valid molecules in interpolation.")
        return

    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=min(5, len(mols)),
        subImgSize=(300, 200),
        legends=labels[: len(mols)],
    )

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Latent Space Interpolation", fontsize=14, fontweight="bold")

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Interpolation plot → %s", save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_metrics_summary(
    metrics: dict[str, float],
    save_path: Optional[str | Path] = None,
) -> None:
    """Bar chart of generation quality metrics.

    Parameters
    ----------
    metrics : dict
        From ``metrics.compute_all_metrics()``.
    save_path : str or Path, optional
        Save path.
    """
    import matplotlib.pyplot as plt

    _setup_style()

    # Select rate metrics (0-1 range)
    rate_metrics = {
        k: v for k, v in metrics.items()
        if k in ("validity", "uniqueness", "novelty", "diversity")
    }

    fig, ax = plt.subplots(figsize=(8, 4))

    names = list(rate_metrics.keys())
    values = list(rate_metrics.values())
    colours = PALETTE[: len(names)]

    bars = ax.bar(names, values, color=colours, edgecolor=DEEP_NAVY, width=0.6)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.3f}",
            ha="center",
            fontweight="bold",
            color=DEEP_NAVY,
        )

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Generation Quality Metrics", fontsize=14, fontweight="bold")

    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Metrics summary → %s", save_path)
        plt.close(fig)
    else:
        plt.show()
