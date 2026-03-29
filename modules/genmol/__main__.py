"""
GenMol CLI
==========

Command-line interface for the generative molecular design module.

Commands::

    genmol download   — Download drug-like molecules from ChEMBL
    genmol train      — Train the VAE on SMILES data
    genmol generate   — Generate novel molecules
    genmol evaluate   — Compute generation quality metrics
    genmol interpolate — Interpolate between two molecules
    genmol screen     — Screen generated molecules through MolScreen
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="genmol",
    help="GenMol — Generative Molecular Design with VAE.",
    rich_markup_mode="rich",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("genmol")


@app.command()
def download(
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output CSV path."
    ),
    max_molecules: int = typer.Option(
        500_000, "--max", "-n", help="Maximum molecules to download."
    ),
) -> None:
    """Download drug-like molecules from ChEMBL."""
    from .data.download import download_chembl

    path = download_chembl(output_path=output, max_molecules=max_molecules)
    typer.echo(f"✅ Data saved → {path}")


@app.command()
def train(
    data_path: Optional[str] = typer.Option(
        None, "--data", "-d", help="Path to SMILES CSV."
    ),
    epochs: int = typer.Option(100, "--epochs", "-e"),
    batch_size: int = typer.Option(256, "--batch-size", "-b"),
    latent_dim: int = typer.Option(128, "--latent-dim"),
    hidden_dim: int = typer.Option(256, "--hidden-dim"),
    lr: float = typer.Option(1e-3, "--lr"),
    checkpoint_dir: str = typer.Option(
        "checkpoints/genmol", "--checkpoint-dir", "-c"
    ),
    max_length: int = typer.Option(120, "--max-length"),
    augment: int = typer.Option(
        0, "--augment", help="SMILES augmentation factor (0 = off)."
    ),
) -> None:
    """Train the Molecular VAE."""
    import torch
    from .data.download import load_smiles
    from .data.preprocess import preprocess_dataset, augment_dataset
    from .data.tokenizer import SmilesTokenizer
    from .data.dataset import create_dataloaders
    from .models.vae import MolVAE
    from .train import train_vae, TrainConfig

    # Load data
    smiles = load_smiles(data_path)
    typer.echo(f"📊 Loaded {len(smiles)} SMILES.")

    # Preprocess
    smiles = preprocess_dataset(smiles, max_length=max_length)
    typer.echo(f"🧹 After preprocessing: {len(smiles)} SMILES.")

    # Optional augmentation
    if augment > 0:
        smiles = augment_dataset(smiles, n_augmentations=augment)
        typer.echo(f"📈 After augmentation: {len(smiles)} SMILES.")

    # Build tokenizer
    tokenizer = SmilesTokenizer(max_length=max_length)
    tokenizer.build_vocab(smiles)
    typer.echo(f"🔤 Vocabulary: {tokenizer.vocab_size} tokens.")

    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        smiles, tokenizer, batch_size=batch_size
    )

    # Build model
    model = MolVAE(
        vocab_size=tokenizer.vocab_size,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        max_length=max_length,
        pad_idx=tokenizer.pad_idx,
    )
    n_params = sum(p.numel() for p in model.parameters())
    typer.echo(f"🧠 Model: {n_params:,} parameters.")

    # Train
    config = TrainConfig(
        epochs=epochs,
        lr=lr,
        checkpoint_dir=checkpoint_dir,
    )
    history = train_vae(
        model, train_loader, val_loader,
        config=config, tokenizer=tokenizer,
    )

    typer.echo(f"✅ Training complete. Checkpoints → {checkpoint_dir}")


@app.command()
def generate(
    checkpoint: str = typer.Option(
        "checkpoints/genmol/best_model.pt",
        "--checkpoint",
        "-c",
        help="Path to model checkpoint.",
    ),
    tokenizer_path: str = typer.Option(
        "checkpoints/genmol/tokenizer.json",
        "--tokenizer",
        "-t",
        help="Path to tokenizer JSON.",
    ),
    n: int = typer.Option(100, "--num", "-n", help="Number to generate."),
    temperature: float = typer.Option(
        1.0, "--temperature", "-T", help="Sampling temperature."
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file (JSON)."
    ),
    screen: bool = typer.Option(
        False, "--screen", "-s", help="Screen through MolScreen."
    ),
) -> None:
    """Generate novel molecules from a trained VAE."""
    import torch
    from .data.tokenizer import SmilesTokenizer
    from .models.vae import MolVAE
    from .train import load_checkpoint
    from .generate import generate as gen_molecules, screen_generated

    # Load tokenizer
    tokenizer = SmilesTokenizer.load(tokenizer_path)

    # Load model
    ckpt = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    model_config = ckpt.get("config", {})

    model = MolVAE(
        vocab_size=tokenizer.vocab_size,
        latent_dim=model_config.get("latent_dim", 128),
        hidden_dim=model_config.get("hidden_dim", 256),
        max_length=tokenizer.max_length,
        pad_idx=tokenizer.pad_idx,
    )
    model.load_state_dict(ckpt["model_state_dict"])

    # Generate
    smiles = gen_molecules(
        model, tokenizer, n=n, temperature=temperature
    )
    typer.echo(f"🧪 Generated {len(smiles)} valid molecules.")

    # Optional screening
    if screen:
        results = screen_generated(smiles)
        typer.echo(f"🔬 {len(results)} passed MolScreen.")
    else:
        results = [{"smiles": s} for s in smiles]

    # Output
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(results, indent=2))
        typer.echo(f"💾 Results → {output}")
    else:
        for r in results[:20]:
            typer.echo(f"  {r['smiles']}")
        if len(results) > 20:
            typer.echo(f"  … and {len(results) - 20} more.")


@app.command()
def evaluate(
    generated_file: str = typer.Argument(
        ..., help="JSON file with generated SMILES."
    ),
    reference_file: Optional[str] = typer.Option(
        None, "--reference", "-r", help="Reference SMILES CSV."
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Save metrics JSON."
    ),
) -> None:
    """Evaluate generation quality metrics."""
    from .evaluation.metrics import compute_all_metrics
    from .data.download import load_smiles

    # Load generated SMILES
    data = json.loads(Path(generated_file).read_text())
    if isinstance(data, list) and isinstance(data[0], dict):
        gen_smiles = [d["smiles"] for d in data]
    else:
        gen_smiles = data

    # Load reference
    ref_smiles = load_smiles(reference_file) if reference_file else None

    # Compute metrics
    metrics = compute_all_metrics(gen_smiles, ref_smiles)

    typer.echo("📊 Generation Metrics:")
    for k, v in metrics.items():
        typer.echo(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    if output:
        Path(output).write_text(json.dumps(metrics, indent=2))
        typer.echo(f"💾 Metrics → {output}")


@app.command()
def interpolate(
    smiles_a: str = typer.Argument(..., help="Starting molecule SMILES."),
    smiles_b: str = typer.Argument(..., help="Ending molecule SMILES."),
    checkpoint: str = typer.Option(
        "checkpoints/genmol/best_model.pt", "--checkpoint", "-c"
    ),
    tokenizer_path: str = typer.Option(
        "checkpoints/genmol/tokenizer.json", "--tokenizer", "-t"
    ),
    steps: int = typer.Option(10, "--steps", "-n"),
    save_plot: Optional[str] = typer.Option(
        None, "--save-plot", help="Save interpolation plot."
    ),
) -> None:
    """Interpolate between two molecules in latent space."""
    import torch
    from .data.tokenizer import SmilesTokenizer
    from .models.vae import MolVAE
    from .generate import interpolate as interp
    from .evaluation.visualise import plot_interpolation

    tokenizer = SmilesTokenizer.load(tokenizer_path)

    ckpt = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    model = MolVAE(
        vocab_size=tokenizer.vocab_size,
        max_length=tokenizer.max_length,
        pad_idx=tokenizer.pad_idx,
    )
    model.load_state_dict(ckpt["model_state_dict"])

    smiles_list = interp(
        model, tokenizer, smiles_a, smiles_b, n_steps=steps
    )

    typer.echo("🔀 Interpolation:")
    for i, smi in enumerate(smiles_list):
        typer.echo(f"  Step {i}: {smi}")

    if save_plot:
        plot_interpolation(smiles_list, save_path=save_plot)
        typer.echo(f"📊 Plot → {save_plot}")


@app.command()
def screen(
    input_file: str = typer.Argument(
        ..., help="JSON with generated molecules."
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output JSON."
    ),
    max_sa: float = typer.Option(4.0, "--max-sa", help="Max SA score."),
    min_qed: float = typer.Option(0.3, "--min-qed", help="Min QED score."),
) -> None:
    """Screen generated molecules through MolScreen."""
    from .generate import screen_generated

    data = json.loads(Path(input_file).read_text())
    if isinstance(data, list) and isinstance(data[0], dict):
        smiles = [d["smiles"] for d in data]
    else:
        smiles = data

    results = screen_generated(
        smiles, max_sa_score=max_sa, min_qed=min_qed
    )

    typer.echo(f"🔬 {len(results)} / {len(smiles)} passed screening.")

    if output:
        Path(output).write_text(json.dumps(results, indent=2))
        typer.echo(f"💾 Results → {output}")
    else:
        for r in results[:10]:
            typer.echo(
                f"  {r['smiles']}  SA={r.get('sa_score', '?'):.2f}  "
                f"QED={r.get('qed_score', '?'):.2f}  "
                f"{r.get('drug_likeness', '?')}"
            )


def main() -> None:
    """Entry point for the ``genmol`` CLI."""
    app()


if __name__ == "__main__":
    main()
