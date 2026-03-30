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

from collections import deque
from typing import Any, Callable, Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray
from tqdm import tqdm

from modules.causalbiorl.causal.discovery import CausalDiscovery
from modules.causalbiorl.causal.planner import CausalPlanner, HierarchicalPlanner
from modules.causalbiorl.causal.scm import StructuralCausalModel


class TransitionBuffer:
    """Circular buffer for ``(s, a, r, s')`` transitions.

    Uses :class:`collections.deque` for O(1) insertion/eviction
    instead of ``list.pop(0)`` which is O(n).
    """

    def __init__(self, capacity: int = 50_000) -> None:
        self.capacity = capacity
        self.states: deque[NDArray[np.floating]] = deque(maxlen=capacity)
        self.actions: deque[NDArray[np.floating]] = deque(maxlen=capacity)
        self.rewards: deque[float] = deque(maxlen=capacity)
        self.next_states: deque[NDArray[np.floating]] = deque(maxlen=capacity)

    def add(
        self,
        state: NDArray[np.floating],
        action: NDArray[np.floating],
        reward: float,
        next_state: NDArray[np.floating],
    ) -> None:
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

    def reward_arrays(
        self,
    ) -> tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """Return ``(states, actions, rewards)`` as numpy arrays."""
        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.actions, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.states)


# ────────────────────────────────────────────────────────────────────────── #
#  Learned reward model                                                      #
# ────────────────────────────────────────────────────────────────────────── #


class RewardPredictor(nn.Module):
    """MLP that learns ``reward(state, action)`` from buffer data.

    This replaces the naïve "drive state to zero" default and ensures
    that the causal planner optimises the **actual** environment reward.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self._device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.net(x).squeeze(-1)

    def fit(
        self,
        states: NDArray[np.floating],
        actions: NDArray[np.floating],
        rewards: NDArray[np.floating],
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> float:
        """Train reward predictor; returns final MSE loss."""
        x = torch.tensor(
            np.concatenate([states, actions], axis=1),
            dtype=torch.float32,
            device=self._device,
        )
        y = torch.tensor(rewards, dtype=torch.float32, device=self._device)

        optimiser = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        self.train()
        final_loss = 0.0
        for _ in range(epochs):
            pred = self(x)
            loss = loss_fn(pred, y)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            final_loss = loss.item()
        self.eval()
        return final_loss

    def predict(
        self,
        state: NDArray[np.floating],
        action: NDArray[np.floating],
    ) -> float:
        """Predict reward for a single ``(state, action)`` pair."""
        with torch.no_grad():
            x = torch.tensor(
                np.concatenate([state.ravel(), action.ravel()]),
                dtype=torch.float32,
                device=self._device,
            ).unsqueeze(0)
            return float(self(x).item())


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
        self._hierarchical_planner: HierarchicalPlanner | None = None
        self._reward_fn: Callable[..., float] | None = None
        self._reward_model = RewardPredictor(self.state_dim, self.action_dim)

        # Detect hierarchical mode (DrugDiscovery-v0)
        self._hierarchical_mode = self.action_dim > 10  # heuristic: latent_dim > 10

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
        self._reward_fn = reward_fn or self._reward_model.predict

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
                    if self._hierarchical_mode and self._hierarchical_planner is not None:
                        action = self._hierarchical_planner.plan(state, rng=self.rng)
                    else:
                        action = self.env.action_space.sample()
                else:
                    if self._hierarchical_mode and self._hierarchical_planner is not None:
                        action = self._hierarchical_planner.plan(state, rng=self.rng)
                    else:
                        action = self._planner.plan(state, rng=self.rng)

                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.buffer.add(state, action, float(reward), next_state)

                # Update hierarchical planner with reward
                if self._hierarchical_mode and self._hierarchical_planner is not None:
                    target_idx = self._parse_target_idx(action)
                    self._hierarchical_planner.update(target_idx, float(reward))

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

    def init_hierarchical_planner(
        self,
        n_targets: int = 5,
        latent_dim: int = 128,
    ) -> None:
        """Initialise the hierarchical planner for drug discovery.

        Call this before training if the environment is
        ``DrugDiscovery-v0``.  The planner uses UCB for target
        selection and CEM for molecule generation.
        """
        self._hierarchical_mode = True
        self._hierarchical_planner = HierarchicalPlanner(
            scm=self._scm,
            reward_fn=self._reward_fn,
            n_targets=n_targets,
            latent_dim=latent_dim,
        )

    def _parse_target_idx(self, action: NDArray[np.floating]) -> int:
        """Extract target index from hierarchical action."""
        if len(action) < 2:
            return 0
        n_targets = getattr(self, "_n_targets", 5)
        normalised = (float(action[0]) + 1.0) / 2.0
        return max(0, min(int(normalised * n_targets), n_targets - 1))

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

        # Fit reward predictor from buffered environment rewards
        r_states, r_actions, r_rewards = self.buffer.reward_arrays()
        self._reward_model.fit(r_states, r_actions, r_rewards)

        # Rebuild planner with updated SCM + learned reward
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
        """Legacy fallback reward: negative L2 norm of state.

        Deprecated — the agent now uses :class:`RewardPredictor` to
        learn the environment's actual reward function from buffered
        transitions.  This method is kept only for backward compat
        if ``reward_fn`` is explicitly passed as ``_default_reward_fn``.
        """
        return -float(np.linalg.norm(state))
