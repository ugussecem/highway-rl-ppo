"""Evaluate a saved PPO policy on highway-fast-v0.

Reports mean / std of episode reward and length over a fixed number of
deterministic episodes.

Usage
-----
$ python src/evaluate.py                       # evaluates ppo_full.zip
$ python src/evaluate.py --stage half          # evaluates ppo_half.zip
$ python src/evaluate.py --episodes 20
"""

from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO

from config import CHECKPOINT_PATHS, TRAINING
from utils import make_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved PPO policy")
    p.add_argument(
        "--stage",
        choices=["untrained", "half", "full"],
        default="full",
        help="Which checkpoint to evaluate.",
    )
    p.add_argument("--episodes", type=int, default=TRAINING.eval_episodes)
    p.add_argument("--seed", type=int, default=TRAINING.seed + 999)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ckpt = CHECKPOINT_PATHS[args.stage]
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint {ckpt} not found. Run src/train.py first."
        )

    env = make_env(seed=args.seed, monitor=False, render_mode=None)()
    model = PPO.load(str(ckpt))

    rewards: list[float] = []
    lengths: list[int] = []
    crashes = 0

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        total_r, steps, crashed = 0.0, 0, False
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_r += float(reward)
            steps += 1
            if info.get("crashed", False):
                crashed = True
        rewards.append(total_r)
        lengths.append(steps)
        crashes += int(crashed)
        print(f"  ep {ep + 1:2d}: reward={total_r:7.2f}  length={steps:3d}  crashed={crashed}")

    env.close()

    print()
    print(f"Stage: {args.stage}  (checkpoint: {ckpt.name})")
    print(f"Episodes:        {args.episodes}")
    print(f"Reward mean±std: {np.mean(rewards):.3f} ± {np.std(rewards):.3f}")
    print(f"Length mean±std: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
    print(f"Crash rate:      {crashes}/{args.episodes} = {crashes / args.episodes:.0%}")


if __name__ == "__main__":
    main()
