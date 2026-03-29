"""
Tests for the causal RL agent and its components.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import modules.causalbiorl  # noqa: F401
from modules.causalbiorl.agents.causal_agent import CausalAgent, TransitionBuffer
from modules.causalbiorl.causal.discovery import CausalDiscovery, NeuralDiscovery
from modules.causalbiorl.causal.planner import CausalPlanner
from modules.causalbiorl.causal.scm import StructuralCausalModel


# ────────────────────────────────────────────────────────────────────────── #
#  Transition Buffer                                                         #
# ────────────────────────────────────────────────────────────────────────── #


class TestTransitionBuffer:
    def test_add_and_length(self) -> None:
        buf = TransitionBuffer(capacity=100)
        for i in range(10):
            buf.add(
                np.array([float(i)]),
                np.array([0.5]),
                1.0,
                np.array([float(i + 1)]),
            )
        assert len(buf) == 10

    def test_capacity_overflow(self) -> None:
        buf = TransitionBuffer(capacity=5)
        for i in range(10):
            buf.add(np.array([float(i)]), np.array([0.0]), 0.0, np.array([float(i + 1)]))
        assert len(buf) == 5

    def test_as_arrays(self) -> None:
        buf = TransitionBuffer()
        for i in range(3):
            buf.add(np.array([float(i), 0.0]), np.array([0.5]), 1.0, np.array([float(i + 1), 0.0]))
        s, a, ns = buf.as_arrays()
        assert s.shape == (3, 2)
        assert a.shape == (3, 1)
        assert ns.shape == (3, 2)


# ────────────────────────────────────────────────────────────────────────── #
#  Neural Causal Discovery                                                   #
# ────────────────────────────────────────────────────────────────────────── #


class TestNeuralDiscovery:
    def test_discovers_graph(self) -> None:
        """Neural discovery should return a DiGraph with nodes."""
        rng = np.random.default_rng(42)
        n = 200
        s = rng.normal(0, 1, (n, 2)).astype(np.float32)
        a = rng.uniform(0, 1, (n, 1)).astype(np.float32)
        # Simple linear dynamics: s' = s + a
        ns = s + a

        disc = NeuralDiscovery(epochs=50, threshold=0.3)
        G = disc.discover(s, a, ns, node_names=["s0", "s1", "a0", "s'0", "s'1"])
        assert G.number_of_nodes() > 0

    def test_unified_interface(self) -> None:
        rng = np.random.default_rng(0)
        s = rng.normal(0, 1, (100, 2)).astype(np.float32)
        a = rng.uniform(0, 1, (100, 1)).astype(np.float32)
        ns = s + a
        disc = CausalDiscovery(method="neural", epochs=30)
        G = disc.discover(s, a, ns)
        assert G.number_of_nodes() > 0


# ────────────────────────────────────────────────────────────────────────── #
#  SCM                                                                       #
# ────────────────────────────────────────────────────────────────────────── #


class TestSCM:
    def _make_simple_scm(self):
        import networkx as nx
        G = nx.DiGraph()
        G.add_nodes_from(["s0", "s1", "a0"])
        G.add_edge("s0", "s0")
        G.add_edge("a0", "s0")
        G.add_edge("s1", "s1")
        G.add_edge("a0", "s1")
        return StructuralCausalModel(G, state_dim=2, action_dim=1, mechanism="linear", epochs=100)

    def test_fit_and_predict(self) -> None:
        scm = self._make_simple_scm()
        rng = np.random.default_rng(0)
        s = rng.normal(0, 1, (200, 2)).astype(np.float32)
        a = rng.uniform(0, 1, (200, 1)).astype(np.float32)
        ns = s + np.tile(a, (1, 2))  # simple known dynamics

        loss = scm.fit(s, a, ns)
        assert loss < 1.0, f"SCM training loss too high: {loss}"

        pred = scm.predict(s[0], a[0])
        assert pred.shape == (2,)

    def test_do_operator(self) -> None:
        scm = self._make_simple_scm()
        rng = np.random.default_rng(0)
        s = rng.normal(0, 1, (200, 2)).astype(np.float32)
        a = rng.uniform(0, 1, (200, 1)).astype(np.float32)
        ns = s + np.tile(a, (1, 2))
        scm.fit(s, a, ns)

        state = np.array([1.0, 2.0], dtype=np.float32)
        action = np.array([0.5], dtype=np.float32)

        # Normal prediction
        pred_normal = scm.predict(state, action)

        # Intervention: fix s0 to 0
        pred_do = scm.do({"s0": 0.0}, state, action)
        assert pred_do[0] == 0.0, "do(s0=0) should fix s0"


# ────────────────────────────────────────────────────────────────────────── #
#  Planner                                                                   #
# ────────────────────────────────────────────────────────────────────────── #


class TestPlanner:
    def test_plan_returns_action(self) -> None:
        import networkx as nx
        G = nx.DiGraph()
        G.add_nodes_from(["s0", "a0"])
        G.add_edge("s0", "s0")
        G.add_edge("a0", "s0")
        scm = StructuralCausalModel(G, state_dim=1, action_dim=1, mechanism="linear", epochs=50)
        rng_np = np.random.default_rng(0)
        s = rng_np.normal(0, 1, (100, 1)).astype(np.float32)
        a = rng_np.uniform(0, 1, (100, 1)).astype(np.float32)
        scm.fit(s, a, s + a)

        planner = CausalPlanner(
            scm=scm,
            reward_fn=lambda s, a: -float(np.abs(s[0])),
            action_dim=1,
            method="cem",
            n_samples=50,
        )
        action = planner.plan(np.array([2.0], dtype=np.float32))
        assert action.shape == (1,)
        assert 0.0 <= action[0] <= 1.0


# ────────────────────────────────────────────────────────────────────────── #
#  Full Causal Agent (short run)                                             #
# ────────────────────────────────────────────────────────────────────────── #


class TestCausalAgent:
    def test_short_training_run(self) -> None:
        """Agent should complete a short training loop without errors."""
        env = gym.make("GeneticToggle-v0", difficulty="easy")
        agent = CausalAgent(
            env,
            discovery_method="neural",
            mechanism="linear",
            warmup_steps=50,
            rediscover_interval=5,
            refit_interval=2,
            planning_samples=20,
            planning_horizon=1,
            seed=42,
        )
        metrics = agent.train(n_episodes=15, verbose=False)
        assert len(metrics["episode_rewards"]) == 15
        assert metrics["total_steps"] > 0
        env.close()

    def test_act_before_training(self) -> None:
        """act() should work even without training (random fallback)."""
        env = gym.make("GeneticToggle-v0")
        agent = CausalAgent(env, seed=0)
        obs, _ = env.reset()
        action = agent.act(obs)
        assert env.action_space.contains(action)
        env.close()
