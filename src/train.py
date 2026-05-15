"""Train a PPO agent on highway-fast-v0.

Three checkpoints are persisted to ``checkpoints/`` to match the three
stages required for the evolution video:

  * ``ppo_untrained.zip`` - the freshly initialized policy (no learning),
  * ``ppo_half.zip``      - policy after ~25% of total_timesteps,
  * ``ppo_full.zip``      - the fully trained policy.

Usage
-----
$ python src/train.py
$ python src/train.py --timesteps 100000     # quick smoke test
$ python src/train.py --n-envs 1             # single-process debugging
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from config import CHECKPOINT_PATHS, LOG_DIR, PPO as PPO_CFG, TRAINING
from model import build_policy_kwargs
from utils import make_vec_env, plot_training_curve


class StageCheckpointCallback(BaseCallback):
    """Save the model exactly once when crossing a target timestep."""

    def __init__(self, target_step: int, save_path: Path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._target = target_step
        self._save_path = save_path
        self._saved = False

    def _on_step(self) -> bool:
        if not self._saved and self.num_timesteps >= self._target:
            self.model.save(str(self._save_path))
            self._saved = True
            if self.verbose:
                print(f"[checkpoint] saved {self._save_path.name} at step {self.num_timesteps}")
        return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO on highway-fast-v0")
    p.add_argument("--timesteps", type=int, default=TRAINING.total_timesteps,
                   help="Total training timesteps.")
    p.add_argument("--n-envs", type=int, default=TRAINING.n_envs,
                   help="Number of parallel environments.")
    p.add_argument("--seed", type=int, default=TRAINING.seed)
    p.add_argument("--skip-plot", action="store_true",
                   help="Skip generating the reward plot at the end.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- 1) Build vectorized env --------------------------------------
    vec_env = make_vec_env(n_envs=args.n_envs, seed=args.seed)

    # --- 2) Build PPO model -------------------------------------------
    policy_kwargs = build_policy_kwargs(PPO_CFG.net_arch)
    model = PPO(
        policy=PPO_CFG.policy,
        env=vec_env,
        learning_rate=PPO_CFG.learning_rate,
        n_steps=PPO_CFG.n_steps,
        batch_size=PPO_CFG.batch_size,
        n_epochs=PPO_CFG.n_epochs,
        gamma=PPO_CFG.gamma,
        gae_lambda=PPO_CFG.gae_lambda,
        clip_range=PPO_CFG.clip_range,
        ent_coef=PPO_CFG.ent_coef,
        vf_coef=PPO_CFG.vf_coef,
        max_grad_norm=PPO_CFG.max_grad_norm,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        tensorboard_log=str(LOG_DIR / "tb"),
        verbose=1,
    )

    # --- 3) Save the untrained baseline immediately -------------------
    model.save(str(CHECKPOINT_PATHS["untrained"]))
    print(f"[checkpoint] saved {CHECKPOINT_PATHS['untrained'].name} (no training yet)")

    # --- 4) Train with a stage checkpoint at the half mark ------------
    half_step = max(1, args.timesteps // 4)  # 25% of total -> 'half' stage
    callback = StageCheckpointCallback(
        target_step=half_step,
        save_path=CHECKPOINT_PATHS["half"],
        verbose=1,
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        progress_bar=True,
    )

    # --- 5) Save the fully trained policy -----------------------------
    model.save(str(CHECKPOINT_PATHS["full"]))
    print(f"[checkpoint] saved {CHECKPOINT_PATHS['full'].name}")

    vec_env.close()

    # --- 6) Plot training curve ---------------------------------------
    if not args.skip_plot:
        plot_training_curve(LOG_DIR / "train", Path("assets") / "reward_plot.png")


if __name__ == "__main__":
    main()
