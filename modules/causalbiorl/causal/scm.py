"""
Structural Causal Model (SCM) fitting.

Given a causal graph (from discovery or ground truth), this module fits an
SCM where each edge ``X_i → X_j`` is parameterised by either:

* a **linear function** (fast, interpretable), or
* a **small neural network** (flexible, higher capacity).

The SCM is trained on observed transitions and supports:

* ``predict(state, action)``  — forward simulation under the current graph.
* ``do(variable, value, state)`` — Pearl's *do*-operator: set a variable to
  a fixed value and propagate effects through the graph.
"""

from __future__ import annotations

from typing import Literal

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray


# ────────────────────────────────────────────────────────────────────────── #
#  Per-edge mechanism                                                        #
# ────────────────────────────────────────────────────────────────────────── #


class _LinearMechanism(nn.Module):
    def __init__(self, n_parents: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_parents, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class _NeuralMechanism(nn.Module):
    def __init__(self, n_parents: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_parents, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ────────────────────────────────────────────────────────────────────────── #
#  Structural Causal Model                                                   #
# ────────────────────────────────────────────────────────────────────────── #


class StructuralCausalModel:
    """A differentiable structural causal model fit to transition data.

    Parameters
    ----------
    graph : nx.DiGraph
        The causal graph.  Nodes should be named after state/action dims.
    state_dim : int
        Number of state dimensions.
    action_dim : int
        Number of action dimensions.
    mechanism : ``"linear"`` | ``"neural"``
        Function class for edge mechanisms.
    lr : float
        Learning rate for mechanism fitting.
    epochs : int
        Training epochs.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        state_dim: int,
        action_dim: int,
        mechanism: Literal["linear", "neural"] = "neural",
        lr: float = 1e-3,
        epochs: int = 200,
        hidden: int = 32,
    ) -> None:
        self.graph = graph
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mechanism_type = mechanism
        self.lr = lr
        self.epochs = epochs
        self.hidden = hidden
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # All node names: first state dims, then action dims
        self.all_names: list[str] = list(graph.nodes)
        # Target (endogenous) nodes — state dimensions that the SCM predicts
        self.target_names: list[str] = self.all_names[:state_dim]

        # Build per-node mechanisms
        self.mechanisms: dict[str, nn.Module] = {}
        self.parent_indices: dict[str, list[int]] = {}
        self._build_mechanisms()

    # ------------------------------------------------------------------ #

    def _build_mechanisms(self) -> None:
        name_to_idx = {n: i for i, n in enumerate(self.all_names)}
        for node in self.target_names:
            parents = list(self.graph.predecessors(node))
            # Remove self-loops for mechanism input
            parents_clean = [p for p in parents if p != node]
            if not parents_clean:
                parents_clean = [node]  # exogenous / root
            pidx = [name_to_idx[p] for p in parents_clean if p in name_to_idx]
            if not pidx:
                pidx = [name_to_idx[node]]
            self.parent_indices[node] = pidx
            n_parents = len(pidx)
            if self.mechanism_type == "linear":
                self.mechanisms[node] = _LinearMechanism(n_parents).to(self.device)
            else:
                self.mechanisms[node] = _NeuralMechanism(n_parents, self.hidden).to(self.device)

    # ------------------------------------------------------------------ #
    #  Fitting                                                             #
    # ------------------------------------------------------------------ #

    def fit(
        self,
        states: NDArray[np.floating],
        actions: NDArray[np.floating],
        next_states: NDArray[np.floating],
    ) -> float:
        """Fit mechanism parameters on transition data.

        Returns final training loss.
        """
        X = np.concatenate([states, actions], axis=1)  # (N, state+action)
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        Yt = torch.tensor(next_states, dtype=torch.float32, device=self.device)

        all_params: list[nn.Parameter] = []
        for mech in self.mechanisms.values():
            all_params.extend(mech.parameters())

        if not all_params:
            return 0.0

        optim = torch.optim.Adam(all_params, lr=self.lr)
        final_loss = 0.0

        for _ in range(self.epochs):
            total_loss = torch.tensor(0.0, device=self.device)
            for j, node in enumerate(self.target_names):
                pidx = self.parent_indices[node]
                parent_vals = Xt[:, pidx]
                pred = self.mechanisms[node](parent_vals)
                total_loss = total_loss + nn.functional.mse_loss(pred, Yt[:, j])
            optim.zero_grad()
            total_loss.backward()
            optim.step()
            final_loss = total_loss.item()

        return final_loss

    # ------------------------------------------------------------------ #
    #  Prediction / Intervention                                           #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict(
        self,
        state: NDArray[np.floating],
        action: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Forward-predict next state given current state and action."""
        x = np.concatenate([state, action])
        xt = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)

        next_state = np.zeros(self.state_dim, dtype=np.float32)
        for j, node in enumerate(self.target_names):
            pidx = self.parent_indices[node]
            parent_vals = xt[:, pidx]
            next_state[j] = self.mechanisms[node](parent_vals).item()
        return next_state

    @torch.no_grad()
    def do(
        self,
        intervention: dict[str, float],
        state: NDArray[np.floating],
        action: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Apply Pearl's do-operator.

        Fixes the intervened variables to their specified values, removes
        incoming edges (in effect), and propagates the rest through the
        SCM in topological order.

        Parameters
        ----------
        intervention : dict[str, float]
            Mapping ``variable_name → fixed_value``.
        state, action : ndarray
            Current state and action context.

        Returns
        -------
        ndarray — predicted next state under the intervention.
        """
        x = np.concatenate([state, action]).astype(np.float32)
        name_to_idx = {n: i for i, n in enumerate(self.all_names)}

        # Apply interventions to the input vector
        for var, val in intervention.items():
            if var in name_to_idx:
                x[name_to_idx[var]] = val

        xt = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Topological prediction, skipping intervened nodes
        next_state = np.zeros(self.state_dim, dtype=np.float32)
        for j, node in enumerate(self.target_names):
            if node in intervention:
                next_state[j] = intervention[node]
            else:
                pidx = self.parent_indices[node]
                parent_vals = xt[:, pidx]
                next_state[j] = self.mechanisms[node](parent_vals).item()
        return next_state

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    def get_edge_strengths(self) -> dict[tuple[str, str], float]:
        """Return approximate edge strengths (L2 norm of mechanism weights)."""
        strengths: dict[tuple[str, str], float] = {}
        for node in self.target_names:
            pidx = self.parent_indices[node]
            mech = self.mechanisms[node]
            weight_norm = sum(p.norm().item() for p in mech.parameters())
            parent_names = [self.all_names[i] for i in pidx]
            for pname in parent_names:
                strengths[(pname, node)] = weight_norm / len(parent_names)
        return strengths
