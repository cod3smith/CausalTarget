"""
Baseline RL agents for comparison — PPO, SAC, and Random.

Wraps ``stable-baselines3`` implementations with a consistent interface
so that benchmarking code can treat all agents uniformly.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


# ────────────────────────────────────────────────────────────────────────── #
#  Random agent                                                              #
# ────────────────────────────────────────────────────────────────────────── #


class RandomAgent:
    """Agent that samples uniformly from the action space."""

    def __init__(self, env: gym.Env, seed: int | None = None) -> None:
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []

    def train(
        self, n_episodes: int = 500, verbose: bool = True, **_kwargs: object
    ) -> dict[str, Any]:
        pbar = tqdm(range(n_episodes), desc="RandomAgent", disable=not verbose)
        total_steps = 0
        for _ in pbar:
            state, _ = self.env.reset(seed=int(self.rng.integers(0, 2**31)))
            ep_reward = 0.0
            ep_len = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action = self.env.action_space.sample()
                state, reward, terminated, truncated, _ = self.env.step(action)
                ep_reward += float(reward)
                ep_len += 1
                total_steps += 1
            self.episode_rewards.append(ep_reward)
            self.episode_lengths.append(ep_len)
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "total_steps": total_steps,
        }

    def act(self, state: NDArray[np.floating]) -> NDArray[np.floating]:
        return self.env.action_space.sample()


# ────────────────────────────────────────────────────────────────────────── #
#  Stable-Baselines3 wrappers                                                #
# ────────────────────────────────────────────────────────────────────────── #


class _SB3Agent:
    """Base wrapper for stable-baselines3 algorithms."""

    _algo_cls: type | None = None

    def __init__(
        self,
        env: gym.Env,
        seed: int | None = None,
        policy: str = "MlpPolicy",
        **sb3_kwargs: object,
    ) -> None:
        self.env = env
        self.seed = seed
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []

        try:
            import stable_baselines3  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "stable-baselines3 is required for PPO/SAC baselines: "
                "pip install stable-baselines3"
            ) from exc

        assert self._algo_cls is not None
        self.model = self._algo_cls(
            policy,
            env,
            seed=seed,
            verbose=0,
            **sb3_kwargs,
        )

    def train(
        self,
        n_episodes: int = 500,
        verbose: bool = True,
        **_kwargs: object,
    ) -> dict[str, Any]:
        """Train for *n_episodes* via SB3's ``learn`` method.

        SB3 operates in total timesteps, so we estimate the budget as
        ``n_episodes × env.spec.max_episode_steps`` (or 500 fallback).
        """
        max_ep = getattr(self.env.spec, "max_episode_steps", None) or 500
        total_timesteps = n_episodes * max_ep

        # Use a callback to record per-episode statistics
        self.model.learn(total_timesteps=total_timesteps, progress_bar=verbose)

        # Evaluate to gather reward traces
        self._evaluate(n_episodes, verbose)
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "total_steps": sum(self.episode_lengths),
        }

    def act(self, state: NDArray[np.floating]) -> NDArray[np.floating]:
        action, _ = self.model.predict(state, deterministic=True)
        return action

    def _evaluate(self, n_episodes: int, verbose: bool) -> None:
        pbar = tqdm(range(n_episodes), desc=self.__class__.__name__ + " eval", disable=not verbose)
        for _ in pbar:
            state, _ = self.env.reset()
            ep_reward = 0.0
            ep_len = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action = self.act(state)
                state, reward, terminated, truncated, _ = self.env.step(action)
                ep_reward += float(reward)
                ep_len += 1
            self.episode_rewards.append(ep_reward)
            self.episode_lengths.append(ep_len)


class PPOAgent(_SB3Agent):
    """Proximal Policy Optimisation (PPO) via stable-baselines3."""

    def __init__(self, env: gym.Env, seed: int | None = None, **kwargs: object) -> None:
        from stable_baselines3 import PPO

        self._algo_cls = PPO  # type: ignore[assignment]
        super().__init__(env, seed=seed, **kwargs)


class SACAgent(_SB3Agent):
    """Soft Actor-Critic (SAC) via stable-baselines3."""

    def __init__(self, env: gym.Env, seed: int | None = None, **kwargs: object) -> None:
        from stable_baselines3 import SAC

        self._algo_cls = SAC  # type: ignore[assignment]
        super().__init__(env, seed=seed, **kwargs)
