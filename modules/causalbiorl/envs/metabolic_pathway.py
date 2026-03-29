"""
MetabolicPathway-v0 — Gymnasium environment for a simplified 5-enzyme
metabolic pathway.

Pathway topology:
    S → [E1] → M1 → [E2] → M2 → [E3] → M3 → [E4] → M4 → [E5] → P

Each enzymatic step follows Michaelis-Menten kinetics:
    v_i = V_max_i · [substrate_i] / (K_m_i + [substrate_i])

where V_max_i scales with the controllable enzyme expression level.

The agent controls the expression level of each of the five enzymes.
Expressing enzymes costs metabolic resources, so the agent must balance
product yield against expression burden.
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


class MetabolicPathwayEnv(gym.Env):
    """Gymnasium environment for a 5-enzyme metabolic pathway.

    Parameters
    ----------
    render_mode : str | None
        ``"human"`` or ``"rgb_array"``.
    difficulty : str
        ``"easy"`` / ``"medium"`` / ``"hard"`` — controls noise level.
    substrate_feed : float
        Constant influx rate of substrate S per time step.
    cost_factor : float
        Coefficient for enzyme expression cost.
    max_steps : int
        Episode length (default 500).
    dt : float
        Integration time step (default 0.01).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    N_ENZYMES: int = 5
    # Default Michaelis-Menten parameters per enzyme
    KM_DEFAULT: float = 1.0
    VMAX_SCALE: float = 10.0     # V_max = VMAX_SCALE * expression_level
    ENZYME_DECAY: float = 0.1    # enzyme turnover

    _DIFFICULTY = {
        "easy":   {"noise_std": 0.0,  "km_noise": 0.0},
        "medium": {"noise_std": 0.02, "km_noise": 0.1},
        "hard":   {"noise_std": 0.05, "km_noise": 0.3},
    }

    # Names for the 11-D state vector
    STATE_NAMES: list[str] = [
        "S", "M1", "M2", "M3", "M4", "P",
        "E1_level", "E2_level", "E3_level", "E4_level", "E5_level",
    ]

    def __init__(
        self,
        render_mode: str | None = None,
        difficulty: str = "medium",
        substrate_feed: float = 2.0,
        cost_factor: float = 0.1,
        max_steps: int = 500,
        dt: float = 0.01,
    ) -> None:
        super().__init__()
        assert difficulty in self._DIFFICULTY
        self.render_mode = render_mode
        self.difficulty = difficulty
        self.substrate_feed = substrate_feed
        self.cost_factor = cost_factor
        self.max_steps = max_steps
        self.dt = dt

        diff = self._DIFFICULTY[difficulty]
        self._noise_std: float = diff["noise_std"]
        self._km_noise: float = diff["km_noise"]

        # Michaelis-Menten Km per enzyme (can be perturbed)
        self._km = np.full(self.N_ENZYMES, self.KM_DEFAULT, dtype=np.float32)

        # State: [S, M1, M2, M3, M4, P, E1..E5]
        high = np.full(11, 50.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=np.zeros(11, dtype=np.float32), high=high)

        # Action: enzyme expression levels
        self.action_space = spaces.Box(
            low=np.zeros(self.N_ENZYMES, dtype=np.float32),
            high=np.ones(self.N_ENZYMES, dtype=np.float32),
        )

        self._state = np.zeros(11, dtype=np.float32)
        self._step_count: int = 0
        self._history: list[NDArray[np.floating]] = []
        self._fig: plt.Figure | None = None
        self._ax: plt.Axes | None = None

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
        state = np.zeros(11, dtype=np.float32)
        state[0] = self.np_random.uniform(1.0, 5.0)          # initial substrate
        state[6:11] = self.np_random.uniform(0.1, 0.5, 5)    # initial enzyme levels
        self._state = state
        self._step_count = 0
        self._history = [self._state.copy()]

        # Perturb Km values for difficulty
        if self._km_noise > 0:
            self._km = np.clip(
                self.KM_DEFAULT + self.np_random.normal(0, self._km_noise, self.N_ENZYMES),
                0.1, 5.0,
            ).astype(np.float32)
        else:
            self._km = np.full(self.N_ENZYMES, self.KM_DEFAULT, dtype=np.float32)

        return self._state.copy(), self._get_info()

    def step(
        self, action: NDArray[np.floating]
    ) -> tuple[NDArray[np.floating], float, bool, bool, dict[str, Any]]:
        action = np.clip(action, self.action_space.low, self.action_space.high).astype(np.float32)

        S, M1, M2, M3, M4, P = (float(self._state[i]) for i in range(6))
        E = self._state[6:11].astype(float)

        # Compute Michaelis-Menten fluxes
        substrates = [S, M1, M2, M3, M4]
        fluxes = np.zeros(self.N_ENZYMES, dtype=np.float64)
        for i in range(self.N_ENZYMES):
            vmax = self.VMAX_SCALE * E[i]
            km = float(self._km[i])
            sub = substrates[i]
            fluxes[i] = vmax * sub / (km + sub + 1e-8)

        # Add noise to fluxes
        if self._noise_std > 0:
            fluxes += self.np_random.normal(0, self._noise_std, self.N_ENZYMES)
            fluxes = np.maximum(fluxes, 0.0)

        # ODE update
        dS = self.substrate_feed - fluxes[0]
        dM1 = fluxes[0] - fluxes[1]
        dM2 = fluxes[1] - fluxes[2]
        dM3 = fluxes[2] - fluxes[3]
        dM4 = fluxes[3] - fluxes[4]
        dP = fluxes[4]

        # Enzyme dynamics: expression drives level towards action, with decay
        dE = (action - E) * 1.0 - self.ENZYME_DECAY * E

        new_state = np.zeros(11, dtype=np.float32)
        new_state[0] = max(S + dS * self.dt, 0.0)
        new_state[1] = max(M1 + dM1 * self.dt, 0.0)
        new_state[2] = max(M2 + dM2 * self.dt, 0.0)
        new_state[3] = max(M3 + dM3 * self.dt, 0.0)
        new_state[4] = max(M4 + dM4 * self.dt, 0.0)
        new_state[5] = max(P + dP * self.dt, 0.0)
        new_state[6:11] = np.maximum(E + dE * self.dt, 0.0).astype(np.float32)

        self._state = np.clip(new_state, self.observation_space.low, self.observation_space.high)
        self._history.append(self._state.copy())
        self._step_count += 1

        reward = self._compute_reward(action)
        terminated = False
        truncated = self._step_count >= self.max_steps

        if self.render_mode == "human":
            self.render()

        return self._state.copy(), reward, terminated, truncated, self._get_info()

    def render(self) -> NDArray[np.uint8] | None:
        history = np.array(self._history)
        t = np.arange(len(history)) * self.dt

        def _draw(ax: plt.Axes) -> None:
            ax.clear()
            ax.plot(t, history[:, 0], label="S", linewidth=1.5)
            for i in range(1, 5):
                ax.plot(t, history[:, i], label=f"M{i}", linewidth=1.0, alpha=0.7)
            ax.plot(t, history[:, 5], label="P", linewidth=2.0, color="green")
            ax.set_xlabel("Time")
            ax.set_ylabel("Concentration")
            ax.set_title("Metabolic Pathway")
            ax.legend(loc="upper right", fontsize=7, ncol=2)

        if self.render_mode == "human":
            if self._fig is None:
                matplotlib.use("TkAgg")
                self._fig, self._ax = plt.subplots(figsize=(9, 4))
                plt.ion()
            _draw(self._ax)  # type: ignore[arg-type]
            self._fig.canvas.draw()
            self._fig.canvas.flush_events()
            return None

        if self.render_mode == "rgb_array":
            fig, ax = plt.subplots(figsize=(9, 4))
            _draw(ax)
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
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
        """Ground-truth DAG for the metabolic pathway."""
        G = nx.DiGraph()
        nodes = [
            "S", "M1", "M2", "M3", "M4", "P",
            "E1", "E2", "E3", "E4", "E5",
            "E1_expr", "E2_expr", "E3_expr", "E4_expr", "E5_expr",
        ]
        G.add_nodes_from(nodes)
        # Chain: S → M1 → M2 → M3 → M4 → P  (via enzymes)
        chain_substrates = ["S", "M1", "M2", "M3", "M4"]
        chain_products = ["M1", "M2", "M3", "M4", "P"]
        for i in range(5):
            G.add_edge(chain_substrates[i], chain_products[i], enzyme=f"E{i+1}")
            G.add_edge(f"E{i+1}", chain_products[i], relation="catalysis")
            G.add_edge(f"E{i+1}_expr", f"E{i+1}", relation="expression")
            # Substrate is consumed
            G.add_edge(chain_substrates[i], chain_substrates[i], relation="consumption")
        return G

    def get_intervention_effect(
        self, action: NDArray[np.floating], state: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """True causal effect of enzyme expression action at given state."""
        S = float(state[0])
        substrates = [S] + [float(state[i]) for i in range(1, 5)]
        E = state[6:11].astype(float)

        fluxes = np.zeros(self.N_ENZYMES, dtype=np.float64)
        for i in range(self.N_ENZYMES):
            vmax = self.VMAX_SCALE * E[i]
            km = float(self._km[i])
            fluxes[i] = vmax * substrates[i] / (km + substrates[i] + 1e-8)

        dS = self.substrate_feed - fluxes[0]
        dM = [fluxes[i] - fluxes[i + 1] for i in range(4)]
        dP = fluxes[4]
        dE = (action - E) * 1.0 - self.ENZYME_DECAY * E

        delta = np.zeros(11, dtype=np.float32)
        delta[0] = dS * self.dt
        for i in range(4):
            delta[i + 1] = dM[i] * self.dt
        delta[5] = dP * self.dt
        delta[6:11] = (dE * self.dt).astype(np.float32)
        return delta

    # --------------------------------------------------------------------- #
    #  Helpers                                                               #
    # --------------------------------------------------------------------- #

    def _compute_reward(self, action: NDArray[np.floating]) -> float:
        product = float(self._state[5])
        expression_cost = float(np.sum(action)) * self.cost_factor
        return product - expression_cost

    def _get_info(self) -> dict[str, Any]:
        return {
            "step": self._step_count,
            "product": float(self._state[5]),
            "enzyme_levels": self._state[6:11].copy(),
        }
