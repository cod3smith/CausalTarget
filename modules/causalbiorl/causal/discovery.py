"""
Causal graph discovery from observed state transitions.

Provides two discovery strategies:

1. **PC algorithm** — constraint-based, uses conditional independence tests
   (wraps ``causal-learn``'s ``PC`` implementation).
2. **Neural causal discovery** — a differentiable approach where a masked
   adjacency matrix is learned via gradient descent on prediction error.

Both methods consume a buffer of ``(state, action, next_state)`` transitions
and return a ``networkx.DiGraph`` representing the learned causal structure.
"""

from __future__ import annotations

from typing import Literal

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray


# ────────────────────────────────────────────────────────────────────────── #
#  PC-algorithm wrapper                                                      #
# ────────────────────────────────────────────────────────────────────────── #


class PCDiscovery:
    """Learn a causal graph via the PC algorithm (``causal-learn``)."""

    def __init__(self, alpha: float = 0.05, indep_test: str = "fisherz") -> None:
        self.alpha = alpha
        self.indep_test = indep_test

    def discover(
        self,
        transitions: NDArray[np.floating],
        node_names: list[str] | None = None,
    ) -> nx.DiGraph:
        """Run PC on the transition data.

        Parameters
        ----------
        transitions : ndarray of shape ``(N, D)``
            Each row is ``[state | action | next_state]`` concatenated.
        node_names : list[str] | None
            Optional readable names for each column.

        Returns
        -------
        nx.DiGraph
        """
        try:
            from causallearn.search.ConstraintBased.PC import pc as run_pc
        except ImportError as exc:
            raise ImportError(
                "causal-learn is required for PC discovery: pip install causal-learn"
            ) from exc

        result = run_pc(
            transitions,
            alpha=self.alpha,
            indep_test=self.indep_test,
            show_progress=False,
        )
        adj = result.G.graph  # adjacency matrix from causal-learn

        n = adj.shape[0]
        if node_names is None:
            node_names = [f"X{i}" for i in range(n)]

        G = nx.DiGraph()
        G.add_nodes_from(node_names)
        for i in range(n):
            for j in range(n):
                # causal-learn encodes: -1 → tail, 1 → arrowhead
                if adj[i, j] == -1 and adj[j, i] == 1:
                    G.add_edge(node_names[i], node_names[j])
        return G


# ────────────────────────────────────────────────────────────────────────── #
#  Neural causal discovery                                                   #
# ────────────────────────────────────────────────────────────────────────── #


class _AdjacencyMLP(nn.Module):
    """Learns a masked adjacency via soft-Gumbel thresholding."""

    def __init__(self, d_input: int, d_output: int, hidden: int = 64) -> None:
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        # Learnable logits for each potential edge (input → output)
        self.adj_logits = nn.Parameter(torch.zeros(d_input, d_output))
        self.net = nn.Sequential(
            nn.Linear(d_input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_output),
        )

    def forward(self, x: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
        # Gumbel-sigmoid for differentiable edge selection
        mask = torch.sigmoid(self.adj_logits / temperature)  # (d_in, d_out)
        x_masked = x.unsqueeze(-1) * mask.unsqueeze(0)       # (B, d_in, d_out)
        x_sum = x_masked.sum(dim=1)                          # (B, d_out)
        return self.net(x) * torch.sigmoid(x_sum)             # gated prediction

    def get_adjacency(self, threshold: float = 0.5) -> NDArray[np.floating]:
        with torch.no_grad():
            prob = torch.sigmoid(self.adj_logits).cpu().numpy()
        return (prob > threshold).astype(np.float32)


class NeuralDiscovery:
    """Learn a causal graph by optimising a masked prediction network.

    The network predicts ``next_state`` from ``[state, action]`` while
    learning a binary adjacency mask via Gumbel-sigmoid relaxation.
    An L1 penalty on the mask encourages sparsity.
    """

    def __init__(
        self,
        lr: float = 1e-3,
        sparsity_lambda: float = 0.01,
        epochs: int = 300,
        hidden: int = 64,
        threshold: float = 0.5,
    ) -> None:
        self.lr = lr
        self.sparsity_lambda = sparsity_lambda
        self.epochs = epochs
        self.hidden = hidden
        self.threshold = threshold
        self._model: _AdjacencyMLP | None = None

    def discover(
        self,
        states: NDArray[np.floating],
        actions: NDArray[np.floating],
        next_states: NDArray[np.floating],
        node_names: list[str] | None = None,
    ) -> nx.DiGraph:
        """Train the neural model and extract the adjacency graph.

        Parameters
        ----------
        states, actions, next_states : ndarray
            Transition data arrays of shape ``(N, d_s)``, ``(N, d_a)``,
            ``(N, d_s)`` respectively.
        node_names : list[str] | None
            Readable names — first ``d_s`` for state dims, then ``d_a``
            for action dims.

        Returns
        -------
        nx.DiGraph
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X = np.concatenate([states, actions], axis=1)
        Y = next_states

        d_in = X.shape[1]
        d_out = Y.shape[1]

        if node_names is None:
            node_names = [f"X{i}" for i in range(d_in + d_out)]

        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        Yt = torch.tensor(Y, dtype=torch.float32, device=device)

        model = _AdjacencyMLP(d_in, d_out, hidden=self.hidden).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            pred = model(Xt, temperature=max(0.5, 1.0 - epoch / self.epochs))
            loss_pred = nn.functional.mse_loss(pred, Yt)
            loss_sparse = self.sparsity_lambda * torch.sigmoid(model.adj_logits).sum()
            loss = loss_pred + loss_sparse
            optim.zero_grad()
            loss.backward()
            optim.step()

        self._model = model
        adj = model.get_adjacency(self.threshold)

        # Build graph: input nodes → output nodes
        G = nx.DiGraph()
        input_names = node_names[:d_in]
        output_names = node_names[d_in : d_in + d_out] if len(node_names) > d_in else [f"Y{j}" for j in range(d_out)]
        G.add_nodes_from(input_names + output_names)
        for i in range(d_in):
            for j in range(d_out):
                if adj[i, j] > 0:
                    G.add_edge(input_names[i], output_names[j])
        return G


# ────────────────────────────────────────────────────────────────────────── #
#  Unified interface                                                         #
# ────────────────────────────────────────────────────────────────────────── #


class CausalDiscovery:
    """Unified causal discovery interface.

    Parameters
    ----------
    method : ``"pc"`` | ``"neural"``
    **kwargs
        Forwarded to the underlying discovery class.
    """

    def __init__(self, method: Literal["pc", "neural"] = "neural", **kwargs: object) -> None:
        self.method = method
        if method == "pc":
            self._impl = PCDiscovery(**kwargs)  # type: ignore[arg-type]
        else:
            self._impl = NeuralDiscovery(**kwargs)  # type: ignore[arg-type]

    def discover(
        self,
        states: NDArray[np.floating],
        actions: NDArray[np.floating],
        next_states: NDArray[np.floating],
        node_names: list[str] | None = None,
    ) -> nx.DiGraph:
        if self.method == "pc":
            transitions = np.concatenate([states, actions, next_states], axis=1)
            return self._impl.discover(transitions, node_names)  # type: ignore[arg-type]
        return self._impl.discover(states, actions, next_states, node_names)  # type: ignore[arg-type]
