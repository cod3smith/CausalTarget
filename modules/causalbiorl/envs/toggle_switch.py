"""
GeneticToggle-v0 — Gymnasium environment for the genetic toggle switch.

Based on Gardner, Cantor & Collins (2000) "Construction of a genetic toggle
switch in Escherichia coli". *Nature*, 403, 339–342.

The toggle switch consists of two mutually repressing genes.  The system is
bistable: depending on initial conditions it settles to one of two steady
states (gene-A high / gene-B low, or vice-versa).

ODEs (with inducer modification):
    dA/dt = α₁ / (1 + (B * (1 - inducer_A))^β) − δ·A
    dB/dt = α₂ / (1 + (A * (1 - inducer_B))^γ) − δ·B

Adding inducer_A weakens gene-B's effective repression of gene-A, and
symmetrically for inducer_B.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray


class GeneticToggleSwitchEnv(gym.Env):
    """Gymnasium environment for the genetic toggle switch.

    Parameters
    ----------
    render_mode : str | None
        ``"human"`` opens a matplotlib window; ``"rgb_array"`` returns an
        image array.
    difficulty : str
        ``"easy"`` (no noise), ``"medium"`` (moderate noise every 50 steps),
        ``"hard"`` (heavy noise every 20 steps + large perturbations).
    target : tuple[float, float]
        Desired steady-state expression ``(A_target, B_target)``.
    max_steps : int
        Maximum number of steps per episode (default 200).
    dt : float
        Integration time step (default 0.1).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # -- default kinetic parameters (Gardner et al. 2000) --------------------
    ALPHA1: float = 4.0   # max production rate gene A
    ALPHA2: float = 4.0   # max production rate gene B
    BETA: float = 2.5     # Hill coefficient (B → A repression)
    GAMMA: float = 2.5    # Hill coefficient (A → B repression)
    DELTA: float = 0.5    # degradation rate

    # -- difficulty presets --------------------------------------------------
    _DIFFICULTY = {
        "easy":   {"noise_std": 0.0,  "perturb_interval": 0,   "perturb_mag": 0.0},
        "medium": {"noise_std": 0.05, "perturb_interval": 50,  "perturb_mag": 0.3},
        "hard":   {"noise_std": 0.15, "perturb_interval": 20,  "perturb_mag": 0.8},
    }

    def __init__(
        self,
        render_mode: str | None = None,
        difficulty: str = "medium",
        target: tuple[float, float] = (3.0, 0.5),
        max_steps: int = 200,
        dt: float = 0.1,
    ) -> None:
        super().__init__()
        assert difficulty in self._DIFFICULTY, f"Unknown difficulty '{difficulty}'"
        self.render_mode = render_mode
        self.difficulty = difficulty
        self.target = np.array(target, dtype=np.float32)
        self.max_steps = max_steps
        self.dt = dt

        diff = self._DIFFICULTY[difficulty]
        self._noise_std: float = diff["noise_std"]
        self._perturb_interval: int = diff["perturb_interval"]
        self._perturb_mag: float = diff["perturb_mag"]

        # Spaces — concentrations can grow but we clip for observation
        high = np.array([10.0, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=np.zeros(2, dtype=np.float32), high=high)
        self.action_space = spaces.Box(
            low=np.zeros(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
        )

        # Internal state
        self._state: NDArray[np.floating] = np.zeros(2, dtype=np.float32)
        self._step_count: int = 0
        self._history: list[NDArray[np.floating]] = []
        self._action_history: list[NDArray[np.floating]] = []
        self._fig: plt.Figure | None = None
        self._ax: list[plt.Axes] | None = None

    # --------------------------------------------------------------------- #
    #  Gymnasium API                                                         #
    # --------------------------------------------------------------------- #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.floating], dict[str, Any]]:
        super().reset(seed=seed)
        # Random initial concentrations near the unstable saddle point
        self._state = self.np_random.uniform(low=0.5, high=3.5, size=(2,)).astype(np.float32)
        self._step_count = 0
        self._history = [self._state.copy()]
        self._action_history = []
        return self._state.copy(), self._get_info()

    def step(
        self, action: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], float, bool, bool, dict[str, Any]]:
        action = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)
        self._action_history.append(action.copy())

        A, B = float(self._state[0]), float(self._state[1])
        ind_A, ind_B = float(action[0]), float(action[1])

        # Noise on production rates
        noise_a = self.np_random.normal(0, self._noise_std) if self._noise_std > 0 else 0.0
        noise_b = self.np_random.normal(0, self._noise_std) if self._noise_std > 0 else 0.0

        # Perturbations
        if self._perturb_interval > 0 and self._step_count % self._perturb_interval == 0 and self._step_count > 0:
            noise_a += self.np_random.normal(0, self._perturb_mag)
            noise_b += self.np_random.normal(0, self._perturb_mag)

        # Effective repression weakened by inducers
        effective_B = max(B * (1.0 - ind_A), 0.0)
        effective_A = max(A * (1.0 - ind_B), 0.0)

        alpha1 = max(self.ALPHA1 + noise_a, 0.0)
        alpha2 = max(self.ALPHA2 + noise_b, 0.0)

        dA = alpha1 / (1.0 + effective_B ** self.BETA) - self.DELTA * A
        dB = alpha2 / (1.0 + effective_A ** self.GAMMA) - self.DELTA * B

        A_new = max(A + dA * self.dt, 0.0)
        B_new = max(B + dB * self.dt, 0.0)

        self._state = np.array([A_new, B_new], dtype=np.float32)
        self._state = np.clip(self._state, self.observation_space.low, self.observation_space.high)
        self._history.append(self._state.copy())
        self._step_count += 1

        reward = self._compute_reward()
        terminated = False
        truncated = self._step_count >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return self._state.copy(), reward, terminated, truncated, self._get_info()

    def render(self) -> NDArray[np.uint8] | None:
        history = np.array(self._history)
        if self.render_mode == "human":
            if self._fig is None:
                matplotlib.use("TkAgg")
                self._fig, self._ax = plt.subplots(1, 1, figsize=(8, 4))
                plt.ion()
            ax = self._ax
            assert ax is not None
            ax.clear()  # type: ignore[union-attr]
            t = np.arange(len(history)) * self.dt
            ax.plot(t, history[:, 0], label="Gene A", color="#2196F3")  # type: ignore[union-attr]
            ax.plot(t, history[:, 1], label="Gene B", color="#F44336")  # type: ignore[union-attr]
            ax.axhline(self.target[0], ls="--", color="#2196F3", alpha=0.4, label="A target")  # type: ignore[union-attr]
            ax.axhline(self.target[1], ls="--", color="#F44336", alpha=0.4, label="B target")  # type: ignore[union-attr]
            ax.set_xlabel("Time")  # type: ignore[union-attr]
            ax.set_ylabel("Concentration")  # type: ignore[union-attr]
            ax.set_title("Genetic Toggle Switch")  # type: ignore[union-attr]
            ax.legend(loc="upper right")  # type: ignore[union-attr]
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            return None

        if self.render_mode == "rgb_array":
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            t = np.arange(len(history)) * self.dt
            ax.plot(t, history[:, 0], label="Gene A", color="#2196F3")
            ax.plot(t, history[:, 1], label="Gene B", color="#F44336")
            ax.axhline(self.target[0], ls="--", color="#2196F3", alpha=0.4)
            ax.axhline(self.target[1], ls="--", color="#F44336", alpha=0.4)
            ax.set_xlabel("Time")
            ax.set_ylabel("Concentration")
            ax.set_title("Genetic Toggle Switch")
            ax.legend(loc="upper right")
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            data = np.asarray(buf)[:, :, :3].copy()
            plt.close(fig)
            return data

        return None

    def close(self) -> None:
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None

    # --------------------------------------------------------------------- #
    #  Causal interface                                                      #
    # --------------------------------------------------------------------- #

    @staticmethod
    def get_causal_graph() -> nx.DiGraph:
        """Return the ground-truth causal DAG for the toggle switch.

        Nodes: A, B, inducer_A, inducer_B
        Edges encode the mutual repression and inducer effects.
        """
        G = nx.DiGraph()
        G.add_nodes_from(["A", "B", "inducer_A", "inducer_B"])
        # Mutual repression
        G.add_edge("B", "A", relation="repression")
        G.add_edge("A", "B", relation="repression")
        # Self-dynamics (degradation + production)
        G.add_edge("A", "A", relation="self-dynamics")
        G.add_edge("B", "B", relation="self-dynamics")
        # Inducer effects — weaken repression
        G.add_edge("inducer_A", "A", relation="activation")
        G.add_edge("inducer_B", "B", relation="activation")
        return G

    def get_intervention_effect(
        self, action: NDArray[np.floating], state: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Return the true causal effect of *action* at *state*.

        Computes one-step state change via the ODEs with the given
        action (inducer concentrations) applied.
        """
        A, B = float(state[0]), float(state[1])
        ind_A, ind_B = float(action[0]), float(action[1])
        effective_B = max(B * (1.0 - ind_A), 0.0)
        effective_A = max(A * (1.0 - ind_B), 0.0)
        dA = self.ALPHA1 / (1.0 + effective_B ** self.BETA) - self.DELTA * A
        dB = self.ALPHA2 / (1.0 + effective_A ** self.GAMMA) - self.DELTA * B
        next_state = np.array([max(A + dA * self.dt, 0.0), max(B + dB * self.dt, 0.0)], dtype=np.float32)
        return next_state - state.astype(np.float32)

    # --------------------------------------------------------------------- #
    #  Internal helpers                                                      #
    # --------------------------------------------------------------------- #

    def _compute_reward(self) -> float:
        distance = float(np.linalg.norm(self._state - self.target))
        return -distance

    def _get_info(self) -> dict[str, Any]:
        return {
            "step": self._step_count,
            "distance_to_target": float(np.linalg.norm(self._state - self.target)),
            "state": self._state.copy(),
        }
