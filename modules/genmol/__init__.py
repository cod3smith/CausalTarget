"""
GenMol — Generative Molecular Design
======================================

A Variational Autoencoder (VAE) for generating novel drug-like
molecules from their SMILES representations.

Sub-packages
------------
* ``data`` — SMILES tokenizer, ChEMBL download, preprocessing, dataset
* ``models`` — MolVAE (standard) and MolCVAE (property-conditioned)
* ``evaluation`` — metrics, distribution analysis, visualisation

Core modules
------------
* ``train`` — training loop with KL annealing, gradient clipping, AMP
* ``generate`` — sampling, interpolation, MolScreen integration
* ``api`` — FastAPI REST service
* ``__main__`` — Typer CLI

Quick start
-----------
>>> from modules.genmol.data import SmilesTokenizer, load_smiles
>>> from modules.genmol.models import MolVAE
>>> from modules.genmol.generate import generate
>>>
>>> smiles = load_smiles()
>>> tok = SmilesTokenizer(); tok.build_vocab(smiles)
>>> model = MolVAE(vocab_size=tok.vocab_size)
>>> # ... train the model ...
>>> generated = generate(model, tok, n=100)
"""

from .data.tokenizer import SmilesTokenizer
from .models.vae import MolVAE, vae_loss
from .models.cvae import MolCVAE, PropertyNormalizer
from .generate import generate, generate_conditional, interpolate
from .train import train_vae, TrainConfig, load_checkpoint

__all__ = [
    # Data
    "SmilesTokenizer",
    # Models
    "MolVAE",
    "MolCVAE",
    "PropertyNormalizer",
    "vae_loss",
    # Training
    "train_vae",
    "TrainConfig",
    "load_checkpoint",
    # Generation
    "generate",
    "generate_conditional",
    "interpolate",
]
