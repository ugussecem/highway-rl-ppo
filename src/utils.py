"""Utility helpers: environment factory, reward wrapper, plotting, video I/O."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import gymnasium as gym
import highway_env  # noqa: F401 - side-effect registration of envs
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from config import ENV_CONFIG, ENV_ID, LOG_DIR, REWARD, RewardConfig


# ---------------------------------------------------------------------------
# Reward shaping wrapper
# ---------------------------------------------------------------------------

class ShapedRewardWrapper(gym.Wrapper):
    """Apply the README reward formula on top of highway-env's reward.

    The formula
        R_t = alpha * speed_norm - beta * collision
              - gamma * lane_change_cost + delta * right_lane_bonus
    is computed from the underlying env's info dict so it matches what
    the agent perceives.
    """

    def __init__(self, env: gym.Env, cfg: RewardConfig = REWARD) -> None:
        super().__init__(env)
        self._cfg = cfg
        self._last_lane: int | None = None

    def reset(self, **kwargs):
        self._last_lane = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        # highway-env exposes per-component rewards in info["rewards"].
        rewards = info.get("rewards", {})
        speed_norm = float(rewards.get("high_speed_reward", 0.0))
        collided = float(rewards.get("collision_reward", 0.0))
        right_lane = float(rewards.get("right_lane_reward", 0.0))

        # Lane-change cost: penalize whenever the discrete action is a
        # lane change (actions 0 and 2 in DiscreteMetaAction).
        lane_change_cost = 1.0 if int(action) in (0, 2) else 0.0

        shaped = (
            self._cfg.alpha_speed * speed_norm
            - self._cfg.beta_collision * collided
            - self._cfg.gamma_lane_change * lane_change_cost
            + self._cfg.delta_right_lane * right_lane
        )

        info["shaped_reward"] = shaped
        return obs, shaped, terminated, truncated, info


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(
    seed: int = 0,
    render_mode: str | None = None,
    monitor: bool = True,
    log_subdir: str = "train",
) -> Callable[[], gym.Env]:
    """Return a thunk producing one wrapped env instance.

    Returning a thunk (factory) rather than an env is what SB3's vector
    env constructors expect.
    """

    def _thunk() -> gym.Env:
        env = gym.make(ENV_ID, render_mode=render_mode, config=ENV_CONFIG)
        env = ShapedRewardWrapper(env)
        if monitor:
            log_path = LOG_DIR / log_subdir
            log_path.mkdir(parents=True, exist_ok=True)
            env = Monitor(env, filename=str(log_path / f"monitor_{seed}"))
        env.reset(seed=seed)
        return env

    return _thunk


def make_vec_env(n_envs: int, seed: int) -> SubprocVecEnv | DummyVecEnv:
    """Build a vectorized training env.

    Falls back to ``DummyVecEnv`` for ``n_envs == 1`` to avoid the
    overhead of subprocess workers when debugging.
    """
    thunks = [make_env(seed=seed + i, monitor=True) for i in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(thunks)
    return SubprocVecEnv(thunks, start_method="spawn")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curve(log_dir: Path, out_path: Path) -> None:
    """Read monitor CSVs and produce reward & episode-length plots."""
    monitor_files = sorted(Path(log_dir).rglob("monitor_*.csv"))
    if not monitor_files:
        print(f"[plot] No monitor logs found under {log_dir}")
        return

    frames = []
    for f in monitor_files:
        try:
            df = pd.read_csv(f, skiprows=1)
            frames.append(df)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[plot] Could not read {f}: {exc}")

    if not frames:
        return

    data = pd.concat(frames, ignore_index=True).sort_values("t").reset_index(drop=True)
    data["episode"] = np.arange(1, len(data) + 1)

    # Rolling mean for legibility on a noisy signal.
    window = max(10, len(data) // 40)
    data["reward_smoothed"] = data["r"].rolling(window=window, min_periods=1).mean()
    data["length_smoothed"] = data["l"].rolling(window=window, min_periods=1).mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(data["episode"], data["r"], alpha=0.25, label="raw")
    axes[0].plot(data["episode"], data["reward_smoothed"], linewidth=2.0, label=f"rolling mean (w={window})")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Episode reward")
    axes[0].set_title("Reward vs. Episodes")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")

    axes[1].plot(data["episode"], data["l"], alpha=0.25, label="raw")
    axes[1].plot(data["episode"], data["length_smoothed"], linewidth=2.0, label=f"rolling mean (w={window})")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Episode length (steps)")
    axes[1].set_title("Episode Length vs. Episodes")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="lower right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved training curve to {out_path}")


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def record_episode_frames(
    env: gym.Env,
    policy,
    max_frames: int = 400,
    deterministic: bool = True,
) -> list[np.ndarray]:
    """Roll out one episode and collect rendered RGB frames.

    ``policy`` may be ``None``, in which case actions are sampled
    uniformly at random (used for the untrained baseline video).
    """
    frames: list[np.ndarray] = []
    obs, _ = env.reset()
    for _ in range(max_frames):
        if policy is None:
            action = env.action_space.sample()
        else:
            action, _ = policy.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, _ = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))
        if terminated or truncated:
            break
    return frames


def save_gif(frames: list[np.ndarray], out_path: Path, fps: int = 15) -> None:
    """Save a list of RGB frames as an animated GIF."""
    if not frames:
        print(f"[video] No frames to save for {out_path}")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    print(f"[video] Saved {len(frames)} frames to {out_path}")


def save_mp4(frames: list[np.ndarray], out_path: Path, fps: int = 15) -> None:
    """Save a list of RGB frames as an MP4 (requires imageio-ffmpeg)."""
    if not frames:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps, codec="libx264", quality=8)
    print(f"[video] Saved {len(frames)} frames to {out_path}")
