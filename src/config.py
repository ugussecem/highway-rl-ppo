"""Centralized configuration for the Highway-Env RL project.

All hyperparameters, paths, and environment settings live here so that
the training, evaluation, and video generation scripts stay decoupled
from tuning decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR: Path = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR: Path = ROOT_DIR / "checkpoints"
VIDEO_DIR: Path = ROOT_DIR / "videos"
ASSET_DIR: Path = ROOT_DIR / "assets"
LOG_DIR: Path = ROOT_DIR / "logs"

for _d in (CHECKPOINT_DIR, VIDEO_DIR, ASSET_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

ENV_ID: str = "highway-fast-v0"

ENV_CONFIG: dict = {
    # Observation: relative kinematics of the 5 closest vehicles.
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 5,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
        "normalize": True,
    },
    # Discrete meta-actions: LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER.
    "action": {"type": "DiscreteMetaAction"},
    "lanes_count": 4,
    "vehicles_count": 50,
    "duration": 40,                 # seconds per episode
    "initial_spacing": 2,
    "collision_reward": -1.0,       # raw env penalty (we add our own on top)
    "reward_speed_range": [20, 30],
    "simulation_frequency": 15,
    "policy_frequency": 5,
}


# ---------------------------------------------------------------------------
# Reward shaping coefficients
# ---------------------------------------------------------------------------
#
# R_t = alpha * speed_norm - beta * collision - gamma * lane_change_cost
#       + delta * right_lane_bonus
#
# These are layered on top of highway-env's built-in reward via a wrapper
# so that the formula in the README matches what the agent actually sees.

@dataclass(frozen=True)
class RewardConfig:
    alpha_speed: float = 0.4         # encourage high speed within safe range
    beta_collision: float = 1.0      # strong penalty for crashes
    gamma_lane_change: float = 0.05  # mild penalty to discourage jitter
    delta_right_lane: float = 0.1    # prefer keeping right (driving discipline)


REWARD: RewardConfig = RewardConfig()


# ---------------------------------------------------------------------------
# PPO hyperparameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PPOConfig:
    policy: str = "MlpPolicy"
    learning_rate: float = 5e-4
    n_steps: int = 512
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    # MLP with two hidden layers of 256 units, shared between actor and critic.
    net_arch: tuple[int, ...] = (256, 256)


PPO: PPOConfig = PPOConfig()


# ---------------------------------------------------------------------------
# Training schedule
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainingConfig:
    total_timesteps: int = 200_000
    n_envs: int = 4                 # parallel envs (SubprocVecEnv)
    seed: int = 42
    # Checkpoint timesteps mapping to the three video stages.
    checkpoint_untrained: int = 0
    checkpoint_half: int = 50_000
    checkpoint_full: int = 200_000
    eval_episodes: int = 10
    video_episodes: int = 1
    video_length_frames: int = 400


TRAINING: TrainingConfig = TrainingConfig()


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

CHECKPOINT_PATHS: dict[str, Path] = {
    "untrained": CHECKPOINT_DIR / "ppo_untrained.zip",
    "half": CHECKPOINT_DIR / "ppo_half.zip",
    "full": CHECKPOINT_DIR / "ppo_full.zip",
}
