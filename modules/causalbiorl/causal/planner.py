"""
Causal planner — action selection via do-calculus on the learned SCM.

Given an SCM, the planner evaluates candidate actions by simulating
their *interventional* effects (Pearl's do-operator) and selects the
action that maximises the expected reward.

Two planning strategies are provided:

* **Grid search** — discretise the action space and evaluate each point.
* **CEM (Cross-Entropy Method)** — iterative sampling for high-dimensional
  action spaces.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray

from modules.causalbiorl.causal.scm import StructuralCausalModel


class CausalPlanner:
    """Select actions by reasoning about interventions through the SCM.

    Parameters
    ----------
    scm : StructuralCausalModel
        Fitted structural causal model.
    reward_fn : callable
        ``reward_fn(state, action) → float``.  Used to score predicted
        outcomes.
    method : ``"grid"`` | ``"cem"``
        Planning strategy.
    action_dim : int
        Dimensionality of the action space.
    n_samples : int
        For grid: total grid points.  For CEM: samples per iteration.
    horizon : int
        Multi-step look-ahead depth (default 1 = greedy).
    cem_iterations : int
        Number of CEM refinement rounds.
    cem_elite_frac : float
        Fraction of top samples retained in CEM.
    """

    def __init__(
        self,
        scm: StructuralCausalModel,
        reward_fn: Callable[[NDArray[np.floating], NDArray[np.floating]], float],
        action_dim: int,
        method: Literal["grid", "cem"] = "cem",
        n_samples: int = 200,
        horizon: int = 1,
        cem_iterations: int = 5,
        cem_elite_frac: float = 0.1,
    ) -> None:
        self.scm = scm
        self.reward_fn = reward_fn
        self.action_dim = action_dim
        self.method = method
        self.n_samples = n_samples
        self.horizon = horizon
        self.cem_iterations = cem_iterations
        self.cem_elite_frac = cem_elite_frac

    # ------------------------------------------------------------------ #
    #  Public                                                              #
    # ------------------------------------------------------------------ #

    def plan(
        self,
        state: NDArray[np.floating],
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.floating]:
        """Return the best action for the given state.

        Uses the SCM's ``do`` operator to predict the interventional
        effect of each candidate action.
        """
        if rng is None:
            rng = np.random.default_rng()

        if self.method == "grid":
            return self._plan_grid(state)
        return self._plan_cem(state, rng)

    # ------------------------------------------------------------------ #
    #  Grid search (low-dim actions)                                       #
    # ------------------------------------------------------------------ #

    def _plan_grid(self, state: NDArray[np.floating]) -> NDArray[np.floating]:
        # Uniform grid over [0, 1]^d
        n_per_dim = max(int(self.n_samples ** (1.0 / self.action_dim)), 2)
        grids = [np.linspace(0.0, 1.0, n_per_dim) for _ in range(self.action_dim)]
        mesh = np.meshgrid(*grids, indexing="ij")
        candidates = np.stack([g.ravel() for g in mesh], axis=-1).astype(np.float32)

        best_reward = -np.inf
        best_action = candidates[0]

        for action in candidates:
            reward = self._rollout(state, action)
            if reward > best_reward:
                best_reward = reward
                best_action = action

        return best_action

    # ------------------------------------------------------------------ #
    #  Cross-Entropy Method (high-dim actions)                             #
    # ------------------------------------------------------------------ #

    def _plan_cem(
        self,
        state: NDArray[np.floating],
        rng: np.random.Generator,
    ) -> NDArray[np.floating]:
        mean = np.full(self.action_dim, 0.5, dtype=np.float32)
        std = np.full(self.action_dim, 0.25, dtype=np.float32)
        n_elite = max(int(self.n_samples * self.cem_elite_frac), 1)

        for _ in range(self.cem_iterations):
            samples = rng.normal(loc=mean, scale=std, size=(self.n_samples, self.action_dim))
            samples = np.clip(samples, 0.0, 1.0).astype(np.float32)

            rewards = np.array([self._rollout(state, a) for a in samples])
            elite_idx = np.argsort(rewards)[-n_elite:]
            elite = samples[elite_idx]

            mean = elite.mean(axis=0)
            std = elite.std(axis=0) + 1e-6  # prevent collapse

        return mean

    # ------------------------------------------------------------------ #
    #  Multi-step rollout                                                  #
    # ------------------------------------------------------------------ #

    def _rollout(self, state: NDArray[np.floating], action: NDArray[np.floating]) -> float:
        """Simulate *horizon* steps with constant action, summing reward."""
        total_reward = 0.0
        s = state.copy()
        for _ in range(self.horizon):
            # Build intervention dict — action dimensions
            intervention: dict[str, float] = {}
            action_start = self.scm.state_dim
            for k in range(self.action_dim):
                idx = action_start + k
                if idx < len(self.scm.all_names):
                    intervention[self.scm.all_names[idx]] = float(action[k])

            next_s = self.scm.do(intervention, s, action)
            total_reward += self.reward_fn(next_s, action)
            s = next_s
        return total_reward
