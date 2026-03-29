"""
Novel causal RL agent — uses structural causal models as world models.

Architecture:
    1. **Causal Discovery** — learns the causal graph from observed
       transitions (neural or PC algorithm).
    2. **Causal World Model** — fits an SCM to the discovered graph.
    3. **Causal Planner** — selects actions via do-calculus on the SCM.
    4. **Model-Based RL loop** — collect → discover → fit SCM → plan → act.

The agent maintains a replay buffer of transitions and periodically
re-discovers the causal structure and re-fits the SCM.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

from modules.causalbiorl.causal.discovery import CausalDiscovery
from modules.causalbiorl.causal.planner import CausalPlanner
from modules.causalbiorl.causal.scm import StructuralCausalModel


class TransitionBuffer:
    """Simple circular buffer for ``(s, a, r, s')`` transitions."""

    def __init__(self, capacity: int = 50_000) -> None:
        self.capacity = capacity
        self.states: list[NDArray[np.floating]] = []
        self.actions: list[NDArray[np.floating]] = []
        self.rewards: list[float] = []
        self.next_states: list[NDArray[np.floating]] = []

    def add(
        self,
        state: NDArray[np.floating],
        action: NDArray[np.floating],
        reward: float,
        next_state: NDArray[np.floating],
    ) -> None:
        if len(self.states) >= self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
        self.states.append(state.copy())
        self.actions.append(action.copy())
        self.rewards.append(reward)
        self.next_states.append(next_state.copy())

    def as_arrays(
        self,
    ) -> tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.actions, dtype=np.float32),
            np.array(self.next_states, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.states)


class CausalAgent:
    """Causal RL agent with SCM-based world model.

    Parameters
    ----------
    env : gym.Env
        The environment to act in.
    discovery_method : ``"pc"`` | ``"neural"``
        Causal discovery backend.
    mechanism : ``"linear"`` | ``"neural"``
        SCM edge parameterisation.
    planning_method : ``"grid"`` | ``"cem"``
        Action selection strategy.
    warmup_steps : int
        Collect this many random transitions before first discovery.
    rediscover_interval : int
        Re-run discovery every N episodes.
    refit_interval : int
        Re-fit SCM every N episodes.
    buffer_capacity : int
        Replay buffer size.
    planning_samples : int
        Number of candidate actions evaluated per step.
    planning_horizon : int
        Multi-step look-ahead depth.
    seed : int | None
        Random seed.
    """

    def __init__(
        self,
        env: gym.Env,
        discovery_method: Literal["pc", "neural"] = "neural",
        mechanism: Literal["linear", "neural"] = "neural",
        planning_method: Literal["grid", "cem"] = "cem",
        warmup_steps: int = 500,
        rediscover_interval: int = 20,
        refit_interval: int = 5,
        buffer_capacity: int = 50_000,
        planning_samples: int = 200,
        planning_horizon: int = 3,
        seed: int | None = None,
    ) -> None:
        self.env = env
        self.warmup_steps = warmup_steps
        self.rediscover_interval = rediscover_interval
        self.refit_interval = refit_interval
        self.planning_samples = planning_samples
        self.planning_horizon = planning_horizon

        self.state_dim = int(np.prod(env.observation_space.shape))  # type: ignore[union-attr]
        self.action_dim = int(np.prod(env.action_space.shape))  # type: ignore[union-attr]

        self.rng = np.random.default_rng(seed)
        self.buffer = TransitionBuffer(capacity=buffer_capacity)
        self.discovery = CausalDiscovery(method=discovery_method)

        self._mechanism_type = mechanism
        self._planning_method = planning_method
        self._scm: StructuralCausalModel | None = None
        self._planner: CausalPlanner | None = None
        self._reward_fn: Callable[..., float] | None = None

        # Metrics
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def train(
        self,
        n_episodes: int = 500,
        reward_fn: Callable[[NDArray[np.floating], NDArray[np.floating]], float] | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Run the full causal model-based RL training loop.

        Parameters
        ----------
        n_episodes : int
            Total training episodes.
        reward_fn : callable | None
            External reward function ``(state, action) → float``.
            If ``None``, the environment reward is used for planning.
        verbose : bool
            Show progress bar.

        Returns
        -------
        dict with training metrics.
        """
        self._reward_fn = reward_fn or self._default_reward_fn

        total_steps = 0
        pbar = tqdm(range(n_episodes), desc="CausalAgent", disable=not verbose)

        for ep in pbar:
            state, _ = self.env.reset(seed=int(self.rng.integers(0, 2**31)))
            ep_reward = 0.0
            ep_len = 0

            terminated = truncated = False
            while not (terminated or truncated):
                # Choose action
                if total_steps < self.warmup_steps or self._planner is None:
                    action = self.env.action_space.sample()
                else:
                    action = self._planner.plan(state, rng=self.rng)

                next_state, reward, terminated, truncated, _ = self.env.step(action)
                self.buffer.add(state, action, float(reward), next_state)

                state = next_state
                ep_reward += float(reward)
                ep_len += 1
                total_steps += 1

            self.episode_rewards.append(ep_reward)
            self.episode_lengths.append(ep_len)

            # Periodic causal discovery + SCM fitting
            if len(self.buffer) >= self.warmup_steps:
                if ep % self.rediscover_interval == 0:
                    self._run_discovery()
                if ep % self.refit_interval == 0 and self._scm is not None:
                    self._fit_scm()

            if verbose and ep % 10 == 0:
                recent = self.episode_rewards[-10:]
                pbar.set_postfix(
                    reward=f"{np.mean(recent):.2f}",
                    buf=len(self.buffer),
                    scm="✓" if self._scm else "✗",
                )

        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "total_steps": total_steps,
        }

    def act(self, state: NDArray[np.floating]) -> NDArray[np.floating]:
        """Select a single action for the given state."""
        if self._planner is None:
            return self.env.action_space.sample()
        return self._planner.plan(state, rng=self.rng)

    def get_learned_graph(self):
        """Return the most recently learned causal graph."""
        if self._scm is not None:
            return self._scm.graph
        return None

    # ------------------------------------------------------------------ #
    #  Internals                                                           #
    # ------------------------------------------------------------------ #

    def _run_discovery(self) -> None:
        states, actions, next_states = self.buffer.as_arrays()
        node_names = (
            [f"s{i}" for i in range(self.state_dim)]
            + [f"a{i}" for i in range(self.action_dim)]
            + [f"s'{i}" for i in range(self.state_dim)]
        )
        graph = self.discovery.discover(states, actions, next_states, node_names)
        self._build_scm(graph)

    def _build_scm(self, graph) -> None:
        """Construct an SCM from the discovered graph and fit it."""
        import networkx as nx

        # Map discovery node names to a clean SCM graph
        scm_graph = nx.DiGraph()
        all_names = (
            [f"s{i}" for i in range(self.state_dim)]
            + [f"a{i}" for i in range(self.action_dim)]
        )
        scm_graph.add_nodes_from(all_names)

        for u, v in graph.edges():
            # Only keep edges among state/action nodes
            if u in all_names and v.startswith("s"):
                # Map s'i back to si for SCM prediction
                target = v.replace("'", "")
                if target in all_names:
                    scm_graph.add_edge(u, target)

        self._scm = StructuralCausalModel(
            graph=scm_graph,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            mechanism=self._mechanism_type,
        )
        self._fit_scm()

    def _fit_scm(self) -> None:
        if self._scm is None:
            return
        states, actions, next_states = self.buffer.as_arrays()
        self._scm.fit(states, actions, next_states)

        # Rebuild planner with updated SCM
        assert self._reward_fn is not None
        self._planner = CausalPlanner(
            scm=self._scm,
            reward_fn=self._reward_fn,
            action_dim=self.action_dim,
            method=self._planning_method,
            n_samples=self.planning_samples,
            horizon=self.planning_horizon,
        )

    @staticmethod
    def _default_reward_fn(
        state: NDArray[np.floating], action: NDArray[np.floating]
    ) -> float:
        """Fallback reward: negative L2 norm of state (drive to zero)."""
        return -float(np.linalg.norm(state))
