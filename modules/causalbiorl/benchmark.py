"""
Full benchmarking suite for CausalBioRL.

Runs all agents on all environments across multiple seeds, collects
statistics, and generates comparison plots and tables.

Key metrics:
  - Cumulative reward per episode
  - Sample efficiency (reward vs episodes)
  - Generalisation (train easy → test hard)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import pandas as pd
from tqdm import tqdm

import modules.causalbiorl  # ensure envs registered  # noqa: F401
from modules.causalbiorl.agents.baseline_agent import PPOAgent, RandomAgent, SACAgent
from modules.causalbiorl.agents.causal_agent import CausalAgent
from modules.causalbiorl.models import BenchmarkResult, EpisodeResult
from modules.causalbiorl.viz import plot_sample_efficiency, plot_trajectories

# Environment IDs
ALL_ENVS = ["GeneticToggle-v0", "MetabolicPathway-v0", "CellGrowth-v0"]
ALL_AGENTS: list[str] = ["causal", "ppo", "sac", "random"]


def _make_agent(
    agent_type: str,
    env: gym.Env,
    seed: int,
) -> CausalAgent | PPOAgent | SACAgent | RandomAgent:
    """Instantiate an agent by name."""
    if agent_type == "causal":
        return CausalAgent(env, seed=seed)
    if agent_type == "ppo":
        return PPOAgent(env, seed=seed)
    if agent_type == "sac":
        return SACAgent(env, seed=seed)
    if agent_type == "random":
        return RandomAgent(env, seed=seed)
    raise ValueError(f"Unknown agent type: {agent_type}")


def run_single(
    env_id: str,
    agent_type: str,
    seed: int,
    n_episodes: int = 500,
    difficulty: str = "medium",
    verbose: bool = False,
) -> EpisodeResult:
    """Run a single agent on a single environment for one seed."""
    env = gym.make(env_id, difficulty=difficulty)
    agent = _make_agent(agent_type, env, seed)
    metrics = agent.train(n_episodes=n_episodes, verbose=verbose)
    env.close()
    return EpisodeResult(
        env_id=env_id,
        agent_type=agent_type,
        seed=seed,
        episode_rewards=metrics["episode_rewards"],
        episode_lengths=metrics["episode_lengths"],
        total_steps=metrics["total_steps"],
    )


def run_benchmark(
    envs: list[str] | None = None,
    agents: list[str] | None = None,
    n_seeds: int = 10,
    n_episodes: int = 500,
    difficulty: str = "medium",
    output_dir: str | Path = "results",
    verbose: bool = True,
) -> list[BenchmarkResult]:
    """Run the full benchmark suite.

    Parameters
    ----------
    envs : list[str] | None
        Environment IDs to benchmark. ``None`` → all.
    agents : list[str] | None
        Agent types to benchmark. ``None`` → all.
    n_seeds : int
        Number of random seeds per experiment.
    n_episodes : int
        Episodes per run.
    difficulty : str
        Environment difficulty level.
    output_dir : path-like
        Directory for saving results.
    verbose : bool
        Show progress.

    Returns
    -------
    list[BenchmarkResult]
    """
    envs = envs or ALL_ENVS
    agents = agents or ALL_AGENTS
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results: list[BenchmarkResult] = []

    for env_id in envs:
        for agent_type in agents:
            if verbose:
                print(f"\n{'='*60}")
                print(f"  {env_id}  ×  {agent_type}  ({n_seeds} seeds)")
                print(f"{'='*60}")

            seed_results: list[EpisodeResult] = []
            for s in tqdm(range(n_seeds), desc=f"{env_id}/{agent_type}", disable=not verbose):
                try:
                    er = run_single(
                        env_id, agent_type, seed=s,
                        n_episodes=n_episodes,
                        difficulty=difficulty,
                        verbose=False,
                    )
                    seed_results.append(er)
                except Exception as exc:
                    if verbose:
                        print(f"  ⚠ seed {s} failed: {exc}")

            if not seed_results:
                continue

            total_rewards = [sum(r.episode_rewards) for r in seed_results]
            total_steps_list = [r.total_steps for r in seed_results]

            br = BenchmarkResult(
                env_id=env_id,
                agent_type=agent_type,
                mean_reward=float(np.mean(total_rewards)),
                std_reward=float(np.std(total_rewards)),
                mean_total_steps=float(np.mean(total_steps_list)),
                std_total_steps=float(np.std(total_steps_list)),
                per_seed=seed_results,
            )
            results.append(br)

    # Save results
    _save_results(results, out)

    # Generate plots
    _generate_plots(results, out)

    return results


def _save_results(results: list[BenchmarkResult], out: Path) -> None:
    """Save benchmark results as JSON and CSV summary."""
    # JSON
    json_path = out / "benchmark_results.json"
    with open(json_path, "w") as f:
        json.dump([r.model_dump() for r in results], f, indent=2, default=str)

    # CSV summary
    rows = []
    for r in results:
        rows.append({
            "env_id": r.env_id,
            "agent": r.agent_type,
            "mean_reward": r.mean_reward,
            "std_reward": r.std_reward,
            "mean_steps": r.mean_total_steps,
            "std_steps": r.std_total_steps,
        })
    df = pd.DataFrame(rows)
    df.to_csv(out / "benchmark_summary.csv", index=False)
    print(f"\n{df.to_string(index=False)}")


def _generate_plots(results: list[BenchmarkResult], out: Path) -> None:
    """Generate comparison plots for each environment."""
    env_ids = list({r.env_id for r in results})
    for env_id in env_ids:
        env_results = [r for r in results if r.env_id == env_id]

        # Sample efficiency plot (mean ± std across seeds)
        seed_data: dict[str, list[list[float]]] = {}
        for r in env_results:
            seed_data[r.agent_type] = [sr.episode_rewards for sr in r.per_seed]

        if seed_data:
            plot_sample_efficiency(
                seed_data,
                title=f"Sample Efficiency — {env_id}",
                save_path=out / f"{env_id}_sample_efficiency.png",
            )

        # Reward curves (first seed)
        reward_data: dict[str, list[float]] = {}
        for r in env_results:
            if r.per_seed:
                reward_data[r.agent_type] = r.per_seed[0].episode_rewards

        if reward_data:
            plot_trajectories(
                reward_data,
                title=f"Reward Trajectory — {env_id}",
                save_path=out / f"{env_id}_reward_trajectory.png",
            )


def run_generalisation_test(
    env_id: str = "GeneticToggle-v0",
    agent_types: list[str] | None = None,
    n_train_episodes: int = 300,
    n_test_episodes: int = 100,
    n_seeds: int = 5,
    output_dir: str | Path = "results",
    verbose: bool = True,
) -> pd.DataFrame:
    """Train on easy difficulty, test on hard — measures generalisation.

    Returns a DataFrame with train and test rewards per agent per seed.
    """
    agent_types = agent_types or ALL_AGENTS
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for agent_type in agent_types:
        for s in range(n_seeds):
            if verbose:
                print(f"  Generalisation: {env_id}/{agent_type} seed={s}")

            # Train on easy
            train_env = gym.make(env_id, difficulty="easy")
            agent = _make_agent(agent_type, train_env, seed=s)
            train_metrics = agent.train(n_episodes=n_train_episodes, verbose=False)
            train_reward = float(np.mean(train_metrics["episode_rewards"][-50:]))
            train_env.close()

            # Test on hard (using same agent — no further training)
            test_env = gym.make(env_id, difficulty="hard")
            test_rewards: list[float] = []
            for _ in range(n_test_episodes):
                state, _ = test_env.reset(seed=s * 1000 + _)
                ep_reward = 0.0
                terminated = truncated = False
                while not (terminated or truncated):
                    action = agent.act(state)
                    state, reward, terminated, truncated, _ = test_env.step(action)
                    ep_reward += float(reward)
                test_rewards.append(ep_reward)
            test_env.close()

            rows.append({
                "env_id": env_id,
                "agent": agent_type,
                "seed": s,
                "train_reward_mean": train_reward,
                "test_reward_mean": float(np.mean(test_rewards)),
                "test_reward_std": float(np.std(test_rewards)),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out / f"{env_id}_generalisation.csv", index=False)
    if verbose:
        print(f"\n{df.to_string(index=False)}")
    return df
