"""
Trajectory and causal graph visualisation utilities for CausalBioRL.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from numpy.typing import NDArray


def plot_trajectories(
    episode_rewards: dict[str, list[float]],
    title: str = "Cumulative Reward per Episode",
    window: int = 10,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot smoothed reward curves for multiple agents.

    Parameters
    ----------
    episode_rewards : dict[str, list[float]]
        Mapping ``agent_name → list_of_episode_rewards``.
    title : str
        Plot title.
    window : int
        Smoothing window size.
    save_path : str | Path | None
        If given, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colours = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]

    for idx, (name, rewards) in enumerate(episode_rewards.items()):
        r = np.array(rewards)
        if len(r) >= window:
            smoothed = np.convolve(r, np.ones(window) / window, mode="valid")
        else:
            smoothed = r
        colour = colours[idx % len(colours)]
        ax.plot(smoothed, label=name, color=colour, linewidth=1.5)
        # Shaded region for raw data
        if len(r) >= window:
            ax.fill_between(
                range(len(smoothed)),
                smoothed - r[window - 1 :].std() * 0.5,
                smoothed + r[window - 1 :].std() * 0.5,
                alpha=0.15,
                color=colour,
            )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_sample_efficiency(
    results: dict[str, list[list[float]]],
    title: str = "Sample Efficiency",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot reward vs episodes with mean ± std across seeds.

    Parameters
    ----------
    results : dict[str, list[list[float]]]
        ``agent_name → list_of_seeds[list_of_episode_rewards]``.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colours = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]

    for idx, (name, seed_rewards) in enumerate(results.items()):
        # Truncate to shortest seed
        min_len = min(len(sr) for sr in seed_rewards)
        arr = np.array([sr[:min_len] for sr in seed_rewards])
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)

        # Smooth
        window = max(min_len // 20, 1)
        if min_len >= window:
            mean_s = np.convolve(mean, np.ones(window) / window, mode="valid")
            std_s = np.convolve(std, np.ones(window) / window, mode="valid")
        else:
            mean_s, std_s = mean, std

        c = colours[idx % len(colours)]
        x = np.arange(len(mean_s))
        ax.plot(x, mean_s, label=name, color=c, linewidth=1.5)
        ax.fill_between(x, mean_s - std_s, mean_s + std_s, alpha=0.2, color=c)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (mean ± std)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_causal_graph(
    graph: nx.DiGraph,
    title: str = "Learned Causal Graph",
    save_path: str | Path | None = None,
    edge_strengths: dict[tuple[str, str], float] | None = None,
) -> plt.Figure:
    """Visualise a causal DAG using networkx spring layout.

    Parameters
    ----------
    graph : nx.DiGraph
        The causal graph to plot.
    title : str
        Plot title.
    save_path : path-like | None
        If given, save to this file.
    edge_strengths : dict | None
        Optional edge weights for line width scaling.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(graph, seed=42, k=2.0)

    # Node colours: state vs action
    node_colours = []
    for node in graph.nodes:
        if str(node).startswith("a") or "expr" in str(node).lower() or "inducer" in str(node).lower() or "feed" in str(node).lower() or "removal" in str(node).lower():
            node_colours.append("#FF9800")
        else:
            node_colours.append("#2196F3")

    # Edge widths
    if edge_strengths:
        widths = [edge_strengths.get((u, v), 1.0) * 3.0 for u, v in graph.edges()]
    else:
        widths = [1.5] * graph.number_of_edges()

    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colours, node_size=700, alpha=0.9)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8, font_weight="bold")
    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        width=widths,
        edge_color="#555555",
        alpha=0.7,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=15,
        connectionstyle="arc3,rad=0.1",
    )

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_state_trajectory(
    history: NDArray[np.floating],
    state_names: list[str],
    dt: float = 1.0,
    title: str = "State Trajectory",
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot each state dimension over time."""
    n_dims = history.shape[1]
    fig, axes = plt.subplots(n_dims, 1, figsize=(10, 2.5 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]

    t = np.arange(len(history)) * dt
    colours = plt.cm.tab10(np.linspace(0, 1, n_dims))

    for i, ax in enumerate(axes):
        name = state_names[i] if i < len(state_names) else f"dim {i}"
        ax.plot(t, history[:, i], color=colours[i], linewidth=1.2)
        ax.set_ylabel(name, fontsize=9)
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Time")
    axes[0].set_title(title)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
