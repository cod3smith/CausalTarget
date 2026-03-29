"""Register CausalBioRL environments with Gymnasium."""

from __future__ import annotations

import gymnasium


def register_envs() -> None:
    """Register all CausalBioRL environments.

    Safe to call multiple times — Gymnasium silently ignores duplicates
    when the entry point is identical.
    """
    _envs = [
        {
            "id": "GeneticToggle-v0",
            "entry_point": "modules.causalbiorl.envs.toggle_switch:GeneticToggleSwitchEnv",
            "max_episode_steps": 200,
        },
        {
            "id": "MetabolicPathway-v0",
            "entry_point": "modules.causalbiorl.envs.metabolic_pathway:MetabolicPathwayEnv",
            "max_episode_steps": 500,
        },
        {
            "id": "CellGrowth-v0",
            "entry_point": "modules.causalbiorl.envs.cell_growth:CellGrowthEnv",
            "max_episode_steps": 500,
        },
    ]

    for spec in _envs:
        try:
            gymnasium.register(**spec)
        except gymnasium.error.RegistrationError:
            pass  # already registered
