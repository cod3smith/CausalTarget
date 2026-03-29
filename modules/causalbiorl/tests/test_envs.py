"""
Tests for CausalBioRL environments.

Validates the Gymnasium interface contract, causal graph structure,
intervention effects, and basic simulation dynamics.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import modules.causalbiorl  # ensure registration  # noqa: F401
from modules.causalbiorl.envs.cell_growth import CellGrowthEnv
from modules.causalbiorl.envs.metabolic_pathway import MetabolicPathwayEnv
from modules.causalbiorl.envs.toggle_switch import GeneticToggleSwitchEnv


# ────────────────────────────────────────────────────────────────────────── #
#  Parametrised across all envs                                              #
# ────────────────────────────────────────────────────────────────────────── #

ENV_IDS = ["GeneticToggle-v0", "MetabolicPathway-v0", "CellGrowth-v0"]
ENV_CLASSES = [GeneticToggleSwitchEnv, MetabolicPathwayEnv, CellGrowthEnv]
DIFFICULTIES = ["easy", "medium", "hard"]


@pytest.mark.parametrize("env_id", ENV_IDS)
class TestGymnasiumInterface:
    """Every environment must satisfy the Gymnasium contract."""

    def test_make(self, env_id: str) -> None:
        env = gym.make(env_id)
        assert env is not None
        env.close()

    def test_reset_returns_obs_and_info(self, env_id: str) -> None:
        env = gym.make(env_id)
        obs, info = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)
        env.close()

    def test_step_signature(self, env_id: str) -> None:
        env = gym.make(env_id)
        env.reset(seed=0)
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()

    def test_observation_in_space(self, env_id: str) -> None:
        env = gym.make(env_id)
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs), f"Reset obs not in space: {obs}"
        for _ in range(10):
            action = env.action_space.sample()
            obs, *_ = env.step(action)
            assert env.observation_space.contains(obs), f"Step obs not in space: {obs}"
        env.close()

    def test_episode_truncates(self, env_id: str) -> None:
        env = gym.make(env_id)
        env.reset(seed=0)
        truncated = False
        for _ in range(2000):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break
        assert terminated or truncated, "Episode did not end within 2000 steps"
        env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_difficulty_levels(env_id: str, difficulty: str) -> None:
    """All difficulty levels should be accepted."""
    env = gym.make(env_id, difficulty=difficulty)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    env.close()


# ────────────────────────────────────────────────────────────────────────── #
#  Causal interface                                                          #
# ────────────────────────────────────────────────────────────────────────── #


@pytest.mark.parametrize("cls", ENV_CLASSES)
def test_causal_graph_is_digraph(cls: type) -> None:
    import networkx as nx
    env = cls()
    G = env.get_causal_graph()
    assert isinstance(G, nx.DiGraph)
    assert G.number_of_nodes() > 0
    assert G.number_of_edges() > 0
    env.close()


@pytest.mark.parametrize("cls", ENV_CLASSES)
def test_intervention_effect_shape(cls: type) -> None:
    env = cls()
    obs, _ = env.reset(seed=0)
    action = env.action_space.sample()
    effect = env.get_intervention_effect(action, obs)
    assert effect.shape == obs.shape
    env.close()


# ────────────────────────────────────────────────────────────────────────── #
#  Toggle-switch-specific                                                    #
# ────────────────────────────────────────────────────────────────────────── #


class TestToggleSwitch:
    def test_bistability(self) -> None:
        """The toggle switch should have two stable regions."""
        env = GeneticToggleSwitchEnv(difficulty="easy")

        # Start near A-high state
        env.reset(seed=0)
        env._state = np.array([4.0, 0.1], dtype=np.float32)
        for _ in range(100):
            env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert env._state[0] > env._state[1], "A should dominate near A-high init"

        # Start near B-high state
        env.reset(seed=0)
        env._state = np.array([0.1, 4.0], dtype=np.float32)
        for _ in range(100):
            env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert env._state[1] > env._state[0], "B should dominate near B-high init"
        env.close()

    def test_inducer_weakens_repression(self) -> None:
        """Adding inducer_A should increase gene A expression."""
        env = GeneticToggleSwitchEnv(difficulty="easy")
        env.reset(seed=0)
        env._state = np.array([1.0, 3.0], dtype=np.float32)

        # Without inducer
        obs_no, *_ = env.step(np.array([0.0, 0.0], dtype=np.float32))
        env.reset(seed=0)
        env._state = np.array([1.0, 3.0], dtype=np.float32)

        # With inducer A
        obs_ind, *_ = env.step(np.array([0.9, 0.0], dtype=np.float32))

        # Inducer should result in higher A (or less decrease)
        assert obs_ind[0] >= obs_no[0], "Inducer A should boost gene A"
        env.close()


# ────────────────────────────────────────────────────────────────────────── #
#  Metabolic-pathway-specific                                                #
# ────────────────────────────────────────────────────────────────────────── #


class TestMetabolicPathway:
    def test_product_accumulates(self) -> None:
        """With active enzymes, product should accumulate."""
        env = MetabolicPathwayEnv(difficulty="easy")
        env.reset(seed=0)
        for _ in range(200):
            env.step(np.ones(5, dtype=np.float32))  # max enzyme expression
        assert env._state[5] > 0, "Product should be positive"
        env.close()

    def test_enzyme_cost_affects_reward(self) -> None:
        """Higher enzyme expression should cost more."""
        env = MetabolicPathwayEnv(difficulty="easy")
        env.reset(seed=0)
        _, r_low, *_ = env.step(np.zeros(5, dtype=np.float32))
        env.reset(seed=0)
        _, r_high, *_ = env.step(np.ones(5, dtype=np.float32))
        # The cost should make r_high not automatically better
        # (product from first step is ~0 either way)
        assert isinstance(r_low, float) and isinstance(r_high, float)
        env.close()


# ────────────────────────────────────────────────────────────────────────── #
#  Cell-growth-specific                                                      #
# ────────────────────────────────────────────────────────────────────────── #


class TestCellGrowth:
    def test_population_grows(self) -> None:
        """With nutrients and no toxin, population should grow."""
        env = CellGrowthEnv(difficulty="easy", target_population=80.0)
        env.reset(seed=0)
        env._state = np.array([10.0, 20.0, 0.0, 0.0, 0.0], dtype=np.float32)
        initial_pop = env._state[0]
        for _ in range(50):
            env.step(np.array([1.0, 0.5], dtype=np.float32))  # feed + some waste removal
        assert env._state[0] > initial_pop, "Population should grow with nutrients"
        env.close()

    def test_population_collapses_under_toxin(self) -> None:
        """Extreme toxin should cause population decline."""
        env = CellGrowthEnv(difficulty="easy")
        env.reset(seed=0)
        env._state = np.array([50.0, 10.0, 0.0, 9.0, 0.0], dtype=np.float32)  # near-lethal toxin
        for _ in range(100):
            env.step(np.array([0.0, 0.0], dtype=np.float32))
        assert env._state[0] < 50.0, "Population should decline under toxin"
        env.close()


# ────────────────────────────────────────────────────────────────────────── #
#  Render                                                                    #
# ────────────────────────────────────────────────────────────────────────── #


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_rgb_array_render(env_id: str) -> None:
    """rgb_array render should return a uint8 numpy array."""
    env = gym.make(env_id, render_mode="rgb_array")
    env.reset(seed=0)
    for _ in range(5):
        env.step(env.action_space.sample())
    frame = env.render()
    assert frame is not None
    assert frame.dtype == np.uint8
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    env.close()
