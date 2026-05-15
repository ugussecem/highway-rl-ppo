"""Build the evolution video required by the rubric.

Loads the three saved policies (untrained, half, full) and renders one
episode per stage. Frames are composited side-by-side into a single GIF
saved at ``assets/evolution.gif`` and an MP4 at ``videos/evolution.mp4``.

Usage
-----
$ python src/make_evolution_video.py
$ python src/make_evolution_video.py --no-mp4   # GIF only
"""

from __future__ import annotations

import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO

from config import ASSET_DIR, CHECKPOINT_PATHS, TRAINING, VIDEO_DIR
from utils import make_env, record_episode_frames, save_gif, save_mp4


STAGES: tuple[str, ...] = ("untrained", "half", "full")
STAGE_LABELS: dict[str, str] = {
    "untrained": "Stage 1 - Untrained",
    "half": "Stage 2 - Half-Trained",
    "full": "Stage 3 - Fully Trained",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate the evolution video")
    p.add_argument(
        "--frames",
        type=int,
        default=TRAINING.video_length_frames,
        help="Max frames per stage.",
    )
    p.add_argument("--no-mp4", action="store_true", help="Skip MP4 output.")
    p.add_argument("--seed", type=int, default=TRAINING.seed + 7)
    return p.parse_args()


def _label_frame(frame: np.ndarray, text: str) -> np.ndarray:
    """Overlay a stage label on the top-left of a frame."""
    img = Image.fromarray(frame).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
        )
    except OSError:
        font = ImageFont.load_default()
    pad = 6
    bbox = draw.textbbox((pad, pad), text, font=font)
    draw.rectangle(bbox, fill=(0, 0, 0))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def _pad_or_trim(frames: list[np.ndarray], length: int) -> list[np.ndarray]:
    """Pad with the last frame (or trim) so every stage has equal length."""
    if not frames:
        return []
    if len(frames) >= length:
        return frames[:length]
    pad_frame = frames[-1]
    return frames + [pad_frame] * (length - len(frames))


def _compose_side_by_side(stage_frames: dict[str, list[np.ndarray]]) -> list[np.ndarray]:
    """Stack the three stages horizontally, frame by frame."""
    # Align all stages to the longest run (capped at the per-stage max).
    max_len = max(len(f) for f in stage_frames.values())
    aligned = {k: _pad_or_trim(v, max_len) for k, v in stage_frames.items()}

    composed: list[np.ndarray] = []
    for i in range(max_len):
        row = [aligned[stage][i] for stage in STAGES]
        # Ensure equal heights (highway-env renders consistently, but be safe).
        h = min(f.shape[0] for f in row)
        row = [f[:h] for f in row]
        composed.append(np.concatenate(row, axis=1))
    return composed


def _record_stage(stage: str, max_frames: int, seed: int) -> list[np.ndarray]:
    """Roll out one episode for a given stage and return labeled frames."""
    env = make_env(seed=seed, monitor=False, render_mode="rgb_array")()

    ckpt = CHECKPOINT_PATHS[stage]
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Missing checkpoint for stage '{stage}': {ckpt}\n"
            "Run src/train.py first."
        )
    policy = PPO.load(str(ckpt))

    frames = record_episode_frames(env, policy, max_frames=max_frames, deterministic=True)
    env.close()

    label = STAGE_LABELS[stage]
    return [_label_frame(f, label) for f in frames]


def main() -> None:
    args = parse_args()

    stage_frames: dict[str, list[np.ndarray]] = {}
    for stage in STAGES:
        print(f"[evolution] recording stage: {stage}")
        stage_frames[stage] = _record_stage(stage, args.frames, args.seed)
        print(f"  -> {len(stage_frames[stage])} frames")

    composed = _compose_side_by_side(stage_frames)
    if not composed:
        print("[evolution] No frames captured — aborting.")
        return

    gif_path = ASSET_DIR / "evolution.gif"
    save_gif(composed, gif_path, fps=15)

    if not args.no_mp4:
        mp4_path = VIDEO_DIR / "evolution.mp4"
        try:
            save_mp4(composed, mp4_path, fps=15)
        except Exception as exc:  # pragma: no cover
            print(f"[evolution] MP4 export failed ({exc}); GIF is still available.")


if __name__ == "__main__":
    main()
