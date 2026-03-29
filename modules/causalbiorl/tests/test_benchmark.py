"""
Tests for the benchmark module.
"""

from __future__ import annotations

import numpy as np
import pytest

import modules.causalbiorl  # noqa: F401
from modules.causalbiorl.benchmark import run_single


class TestBenchmarkSingle:
    def test_run_single_random(self) -> None:
        """run_single with random agent should complete quickly."""
        result = run_single(
            env_id="GeneticToggle-v0",
            agent_type="random",
            seed=0,
            n_episodes=5,
            difficulty="easy",
            verbose=False,
        )
        assert result.env_id == "GeneticToggle-v0"
        assert result.agent_type == "random"
        assert len(result.episode_rewards) == 5
        assert result.total_steps > 0

    def test_run_single_causal(self) -> None:
        """run_single with causal agent should complete (short run)."""
        result = run_single(
            env_id="GeneticToggle-v0",
            agent_type="causal",
            seed=0,
            n_episodes=5,
            difficulty="easy",
            verbose=False,
        )
        assert len(result.episode_rewards) == 5

    @pytest.mark.parametrize("env_id", [
        "GeneticToggle-v0",
        "MetabolicPathway-v0",
        "CellGrowth-v0",
    ])
    def test_random_agent_all_envs(self, env_id: str) -> None:
        result = run_single(
            env_id=env_id,
            agent_type="random",
            seed=42,
            n_episodes=3,
            difficulty="easy",
            verbose=False,
        )
        assert len(result.episode_rewards) == 3
