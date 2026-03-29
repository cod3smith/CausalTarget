"""
CausalBioRL — Causal Reinforcement Learning Environments for Biological System Control.

Provides Gymnasium-compatible RL environments that simulate biological systems
(gene expression, metabolic flux, cell populations) and causal RL agents that
use structural causal models as world models.

Reference:
    CausalBioRL: Causal Reinforcement Learning Environments for
    Biological System Control (2026).
"""

from modules.causalbiorl.envs.registration import register_envs

__version__ = "0.1.0"
__all__ = ["envs", "agents", "causal"]

# Register Gymnasium environments on import
register_envs()
