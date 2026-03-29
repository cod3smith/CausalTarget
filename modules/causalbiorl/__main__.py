"""
CausalBioRL command-line interface.

Usage:
    python -m causalbiorl train --env GeneticToggle-v0 --agent causal --episodes 1000
    python -m causalbiorl benchmark --envs all --agents all --seeds 10
    python -m causalbiorl play --env GeneticToggle-v0
    python -m causalbiorl visualise --env GeneticToggle-v0 --agent causal
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Optional

import gymnasium as gym
import numpy as np
import typer

import modules.causalbiorl  # register envs  # noqa: F401

app = typer.Typer(
    name="causalbiorl",
    help="CausalBioRL — Causal RL environments for biological system control.",
    add_completion=False,
)


# ────────────────────────────────────────────────────────────────────────── #
#  train                                                                     #
# ────────────────────────────────────────────────────────────────────────── #


@app.command()
def train(
    env: Annotated[str, typer.Option(help="Gymnasium environment ID")] = "GeneticToggle-v0",
    agent: Annotated[str, typer.Option(help="Agent type: causal, ppo, sac, random")] = "causal",
    episodes: Annotated[int, typer.Option(help="Number of training episodes")] = 500,
    difficulty: Annotated[str, typer.Option(help="easy / medium / hard")] = "medium",
    seed: Annotated[Optional[int], typer.Option(help="Random seed")] = None,
    output_dir: Annotated[str, typer.Option(help="Output directory")] = "results",
) -> None:
    """Train an agent on a CausalBioRL environment."""
    from modules.causalbiorl.benchmark import _make_agent

    typer.echo(f"🧬 Training {agent} on {env} ({difficulty}) for {episodes} episodes …")
    environment = gym.make(env, difficulty=difficulty)
    ag = _make_agent(agent, environment, seed=seed or 0)
    metrics = ag.train(n_episodes=episodes, verbose=True)
    environment.close()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save reward curve
    import json
    with open(out / f"{env}_{agent}_rewards.json", "w") as f:
        json.dump(metrics["episode_rewards"], f)

    final_mean = float(np.mean(metrics["episode_rewards"][-50:]))
    typer.echo(f"✅ Done — final 50-episode mean reward: {final_mean:.3f}")


# ────────────────────────────────────────────────────────────────────────── #
#  benchmark                                                                 #
# ────────────────────────────────────────────────────────────────────────── #


@app.command()
def benchmark(
    envs: Annotated[str, typer.Option(help="Comma-separated env IDs or 'all'")] = "all",
    agents: Annotated[str, typer.Option(help="Comma-separated agent types or 'all'")] = "all",
    seeds: Annotated[int, typer.Option(help="Number of random seeds")] = 10,
    episodes: Annotated[int, typer.Option(help="Episodes per run")] = 500,
    difficulty: Annotated[str, typer.Option(help="easy / medium / hard")] = "medium",
    output_dir: Annotated[str, typer.Option(help="Output directory")] = "results",
) -> None:
    """Run the full benchmark suite."""
    from modules.causalbiorl.benchmark import ALL_AGENTS, ALL_ENVS, run_benchmark

    env_list = ALL_ENVS if envs == "all" else [e.strip() for e in envs.split(",")]
    agent_list = ALL_AGENTS if agents == "all" else [a.strip() for a in agents.split(",")]

    typer.echo(f"🧬 Benchmark: {env_list} × {agent_list} × {seeds} seeds")
    run_benchmark(
        envs=env_list,
        agents=agent_list,
        n_seeds=seeds,
        n_episodes=episodes,
        difficulty=difficulty,
        output_dir=output_dir,
    )
    typer.echo("✅ Benchmark complete — results saved to " + output_dir)


# ────────────────────────────────────────────────────────────────────────── #
#  play                                                                      #
# ────────────────────────────────────────────────────────────────────────── #


@app.command()
def play(
    env: Annotated[str, typer.Option(help="Gymnasium environment ID")] = "GeneticToggle-v0",
    difficulty: Annotated[str, typer.Option(help="easy / medium / hard")] = "easy",
    steps: Annotated[int, typer.Option(help="Max steps")] = 100,
) -> None:
    """Interactive manual control of an environment.

    Each step prompts you for action values (comma-separated floats in [0, 1]).
    """
    environment = gym.make(env, difficulty=difficulty)
    state, info = environment.reset()
    action_dim = int(np.prod(environment.action_space.shape))  # type: ignore[union-attr]

    typer.echo(f"🎮 Playing {env} ({difficulty}) — action dim = {action_dim}")
    typer.echo(f"   State: {state}")

    for t in range(steps):
        raw = typer.prompt(f"Step {t} — enter {action_dim} action values (comma-sep, 0-1)")
        try:
            action = np.array([float(x.strip()) for x in raw.split(",")], dtype=np.float32)
        except ValueError:
            typer.echo("  ⚠ Invalid input — using zeros.")
            action = np.zeros(action_dim, dtype=np.float32)

        state, reward, terminated, truncated, info = environment.step(action)
        typer.echo(f"   State: {np.round(state, 3)}  |  Reward: {reward:.3f}")

        if terminated or truncated:
            typer.echo("  Episode finished.")
            break

    environment.close()


# ────────────────────────────────────────────────────────────────────────── #
#  visualise                                                                 #
# ────────────────────────────────────────────────────────────────────────── #


@app.command()
def visualise(
    env: Annotated[str, typer.Option(help="Gymnasium environment ID")] = "GeneticToggle-v0",
    agent: Annotated[str, typer.Option(help="Agent type")] = "causal",
    episodes: Annotated[int, typer.Option(help="Training episodes before viz")] = 200,
    difficulty: Annotated[str, typer.Option(help="Difficulty")] = "medium",
    output_dir: Annotated[str, typer.Option(help="Output directory")] = "results",
) -> None:
    """Train an agent and visualise the learned causal graph + trajectories."""
    from modules.causalbiorl.benchmark import _make_agent
    from modules.causalbiorl.viz import plot_causal_graph, plot_trajectories

    typer.echo(f"🔬 Training {agent} on {env} for visualisation …")
    environment = gym.make(env, difficulty=difficulty)
    ag = _make_agent(agent, environment, seed=42)
    metrics = ag.train(n_episodes=episodes, verbose=True)
    environment.close()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Reward trajectory
    plot_trajectories(
        {agent: metrics["episode_rewards"]},
        title=f"{env} — {agent}",
        save_path=out / f"{env}_{agent}_trajectory.png",
    )

    # Causal graph (if causal agent)
    if hasattr(ag, "get_learned_graph"):
        graph = ag.get_learned_graph()
        if graph is not None:
            plot_causal_graph(
                graph,
                title=f"Learned Causal Graph — {env}",
                save_path=out / f"{env}_{agent}_causal_graph.png",
            )
            typer.echo(f"  Learned graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    typer.echo(f"✅ Visualisations saved to {out}")


# ────────────────────────────────────────────────────────────────────────── #
#  Entry point                                                               #
# ────────────────────────────────────────────────────────────────────────── #


def main() -> None:
    app()


if __name__ == "__main__":
    main()
