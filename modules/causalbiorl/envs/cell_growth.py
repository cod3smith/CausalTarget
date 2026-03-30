"""
CellGrowth-v0 — Gymnasium environment for cell population homeostasis.

Simulates a microbial cell population growing under logistic constraints
with nutrient limitation, waste accumulation, and external toxin pulses.

Dynamics:
    dN/dt     = r·N·(1 − N/K)·(nut/(nut + Kn))·(1 − tox/tox_lethal) − d·N
    dnut/dt   = feed_rate − c_rate·N
    dwaste/dt = w_prod·N − w_removal
    dtox/dt   = toxin_pulse(t) − tox_deg·tox

The agent controls *feed_rate* and *waste_removal_rate* to maintain a
target population size despite environmental disturbances.
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


class CellGrowthEnv(gym.Env):
    """Gymnasium environment for cell population homeostasis.

    Parameters
    ----------
    render_mode : str | None
        ``"human"`` or ``"rgb_array"``.
    difficulty : str
        ``"easy"`` / ``"medium"`` / ``"hard"``.
    target_population : float
        Desired steady-state population.
    max_steps : int
        Episode length (default 500).
    dt : float
        Integration time step (default 0.05).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    # Kinetic constants
    R: float = 0.5            # intrinsic growth rate
    K: float = 100.0          # carrying capacity
    KN: float = 2.0           # nutrient half-saturation
    CONSUMPTION: float = 0.05 # nutrient consumption per cell
    WASTE_PROD: float = 0.02  # waste production per cell
    DEATH_RATE: float = 0.05  # basal death rate
    TOX_LETHAL: float = 10.0  # lethal toxin concentration
    TOX_DEG: float = 0.1      # toxin degradation rate
    MAX_FEED: float = 5.0     # max feed rate
    MAX_WASTE_REMOVAL: float = 3.0

    STATE_NAMES: list[str] = ["population", "nutrients", "waste", "toxin", "growth_rate"]

    _DIFFICULTY = {
        "easy":   {"tox_pulse_mag": 0.0,  "tox_interval": (200, 300), "noise_std": 0.0},
        "medium": {"tox_pulse_mag": 3.0,  "tox_interval": (50, 100),  "noise_std": 0.02},
        "hard":   {"tox_pulse_mag": 6.0,  "tox_interval": (30, 60),   "noise_std": 0.08},
    }

    def __init__(
        self,
        render_mode: str | None = None,
        difficulty: str = "medium",
        target_population: float = 50.0,
        max_steps: int = 500,
        dt: float = 0.05,
    ) -> None:
        super().__init__()
        assert difficulty in self._DIFFICULTY
        self.render_mode = render_mode
        self.difficulty = difficulty
        self.target_pop = target_population
        self.max_steps = max_steps
        self.dt = dt

        diff = self._DIFFICULTY[difficulty]
        self._tox_pulse_mag: float = diff["tox_pulse_mag"]
        self._tox_interval: tuple[int, int] = diff["tox_interval"]
        self._noise_std: float = diff["noise_std"]

        # State: [N, nutrients, waste, toxin, growth_rate]
        high = np.array([200.0, 50.0, 50.0, 20.0, 5.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=np.zeros(5, dtype=np.float32), high=high)

        # Action: [feed_rate_frac, waste_removal_frac]  ∈ [0, 1]
        self.action_space = spaces.Box(
            low=np.zeros(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
        )

        self._state = np.zeros(5, dtype=np.float32)
        self._step_count: int = 0
        self._next_tox_step: int = 0
        self._history: list[NDArray[np.floating]] = []
        self._fig: plt.Figure | None = None
        self._axes: list[plt.Axes] | None = None

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
        N0 = self.np_random.uniform(20.0, 80.0)
        nut0 = self.np_random.uniform(5.0, 15.0)
        self._state = np.array([N0, nut0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._step_count = 0
        self._next_tox_step = int(self.np_random.integers(*self._tox_interval)) if self._tox_pulse_mag > 0 else self.max_steps + 1
        self._history = [self._state.copy()]
        return self._state.copy(), self._get_info()

    def step(
        self, action: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], float, bool, bool, dict[str, Any]]:
        action = np.clip(action, 0.0, 1.0).astype(np.float32)
        feed_rate = float(action[0]) * self.MAX_FEED
        waste_removal = float(action[1]) * self.MAX_WASTE_REMOVAL

        N, nut, waste, tox, _ = (float(x) for x in self._state)

        # Toxin pulse
        tox_pulse = 0.0
        if self._step_count >= self._next_tox_step:
            tox_pulse = self._tox_pulse_mag * self.np_random.uniform(0.5, 1.5)
            self._next_tox_step = self._step_count + int(self.np_random.integers(*self._tox_interval))

        # Growth rate
        nutrient_factor = nut / (nut + self.KN) if (nut + self.KN) > 0 else 0.0
        toxin_factor = max(1.0 - tox / self.TOX_LETHAL, 0.0)
        growth_rate = self.R * (1.0 - N / self.K) * nutrient_factor * toxin_factor

        # Noise
        noise = self.np_random.normal(0, self._noise_std) if self._noise_std > 0 else 0.0

        # ODEs
        dN = (growth_rate + noise) * N - self.DEATH_RATE * N
        dnut = feed_rate - self.CONSUMPTION * N
        dwaste = self.WASTE_PROD * N - waste_removal
        dtox = tox_pulse - self.TOX_DEG * tox

        N_new = max(N + dN * self.dt, 0.0)
        nut_new = max(nut + dnut * self.dt, 0.0)
        waste_new = max(waste + dwaste * self.dt, 0.0)
        tox_new = max(tox + dtox * self.dt, 0.0)

        self._state = np.array(
            [N_new, nut_new, waste_new, tox_new, max(growth_rate, 0.0)],
            dtype=np.float32,
        )
        self._state = np.clip(self._state, self.observation_space.low, self.observation_space.high)
        self._history.append(self._state.copy())
        self._step_count += 1

        reward = self._compute_reward(action)
        terminated = N_new <= 0.1  # population collapsed
        truncated = self._step_count >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return self._state.copy(), reward, terminated, truncated, self._get_info()

    def render(self) -> NDArray[np.uint8] | None:
        history = np.array(self._history)
        t = np.arange(len(history)) * self.dt

        def _draw(axes: list[plt.Axes]) -> None:
            labels = self.STATE_NAMES
            colours = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]
            for i, ax in enumerate(axes):
                ax.clear()
                ax.plot(t, history[:, i], color=colours[i], linewidth=1.2)
                ax.set_ylabel(labels[i], fontsize=8)
                if i == 0:
                    ax.axhline(self.target_pop, ls="--", color="grey", alpha=0.5)
            axes[-1].set_xlabel("Time")
            axes[0].set_title("Cell Growth Environment")

        if self.render_mode == "human":
            if self._fig is None:
                matplotlib.use("TkAgg")
                self._fig, self._axes = plt.subplots(5, 1, figsize=(9, 8), sharex=True)
                plt.ion()
            _draw(list(self._axes))  # type: ignore[arg-type]
            self._fig.tight_layout()
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            return None

        if self.render_mode == "rgb_array":
            cur_backend = matplotlib.get_backend()
            matplotlib.use("Agg")
            fig, axes = plt.subplots(5, 1, figsize=(9, 8), sharex=True)
            _draw(list(axes))
            fig.tight_layout()
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            data = np.asarray(buf)[:, :, :3].copy()
            plt.close(fig)
            matplotlib.use(cur_backend)
            return data

        return None

    def close(self) -> None:
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._axes = None

    # --------------------------------------------------------------------- #
    #  Causal interface                                                      #
    # --------------------------------------------------------------------- #

    @staticmethod
    def get_causal_graph() -> nx.DiGraph:
        """Ground-truth causal DAG for the cell-growth system."""
        G = nx.DiGraph()
        G.add_nodes_from(["N", "nutrients", "waste", "toxin", "growth_rate",
                          "feed_rate", "waste_removal"])
        # Population dynamics
        G.add_edge("growth_rate", "N", relation="growth")
        G.add_edge("N", "N", relation="logistic")
        G.add_edge("nutrients", "growth_rate", relation="nutrient_limitation")
        G.add_edge("toxin", "growth_rate", relation="toxin_inhibition")
        G.add_edge("N", "nutrients", relation="consumption")
        G.add_edge("N", "waste", relation="waste_production")
        # Control inputs
        G.add_edge("feed_rate", "nutrients", relation="feeding")
        G.add_edge("waste_removal", "waste", relation="removal")
        # Toxin self-dynamics
        G.add_edge("toxin", "toxin", relation="degradation")
        return G

    def get_intervention_effect(
        self, action: NDArray[np.floating], state: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """True causal effect of the control action at the given state."""
        feed_rate = float(action[0]) * self.MAX_FEED
        waste_removal = float(action[1]) * self.MAX_WASTE_REMOVAL
        N, nut, waste, tox, _ = (float(x) for x in state)

        nutrient_factor = nut / (nut + self.KN) if (nut + self.KN) > 0 else 0.0
        toxin_factor = max(1.0 - tox / self.TOX_LETHAL, 0.0)
        growth_rate = self.R * (1.0 - N / self.K) * nutrient_factor * toxin_factor

        dN = growth_rate * N - self.DEATH_RATE * N
        dnut = feed_rate - self.CONSUMPTION * N
        dwaste = self.WASTE_PROD * N - waste_removal
        dtox = -self.TOX_DEG * tox  # no pulse in true causal effect

        delta = np.array([
            dN * self.dt,
            dnut * self.dt,
            dwaste * self.dt,
            dtox * self.dt,
            0.0,  # growth_rate is derived, not dynamically updated
        ], dtype=np.float32)
        return delta

    # --------------------------------------------------------------------- #
    #  Helpers                                                               #
    # --------------------------------------------------------------------- #

    def _compute_reward(self, action: NDArray[np.floating]) -> float:
        pop_deviation = abs(float(self._state[0]) - self.target_pop)
        resource_cost = float(np.sum(action)) * 0.1
        return -(pop_deviation + resource_cost)

    def _get_info(self) -> dict[str, Any]:
        return {
            "step": self._step_count,
            "population": float(self._state[0]),
            "deviation_from_target": abs(float(self._state[0]) - self.target_pop),
        }
