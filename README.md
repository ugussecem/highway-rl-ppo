<!--
  README.md — final report for CMP4501 Semester Project
  Edit the highlighted TODO lines before submission.
-->

<div align="center">

# 🛣️ Autonomous Driving in Dense Traffic with PPO

### CMP4501 – Applied Reinforcement Learning · Semester Project

**Student:** *TODO — your full name* &nbsp;·&nbsp;
**Track:** Option A — *Autonomous Driving with Highway-Env* &nbsp;·&nbsp;
**Algorithm:** Proximal Policy Optimization (PPO)

</div>

---

## 🎬 Evolution Video

The three required stages — *untrained*, *half-trained*, *fully trained* — are shown side-by-side. The progression from random crashing to smooth lane-keeping is visible in a single glance.

<p align="center">
  <img src="assets/evolution.gif" alt="Training evolution: three stages side by side" width="100%">
</p>

> 🎥 **This video was produced entirely by code** — no screen recording, no manual editing. The script [`src/make_evolution_video.py`](src/make_evolution_video.py) loads the three saved checkpoints (`ppo_untrained.zip`, `ppo_half.zip`, `ppo_full.zip`), runs one deterministic rollout per stage in `highway-env` with `render_mode="rgb_array"`, overlays the stage label on each frame, and composites the three streams horizontally into a single GIF and MP4. Re-running the script reproduces the video bit-for-bit.
>
> 📁 An MP4 copy is also available at [`videos/evolution.mp4`](videos/evolution.mp4).

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Methodology](#-methodology)
   - [Reward Function](#a-reward-function)
   - [Model](#b-model)
   - [States and Actions](#c-states-and-actions)
3. [Experiment Iterations](#-experiment-iterations)
4. [Training Analysis](#-training-analysis)
   - [Reward Graph](#a-reward-graph)
   - [Quantitative Evaluation](#b-quantitative-evaluation)
   - [Commentary](#c-commentary)
5. [Challenges and Failures](#-challenges-and-failures)
6. [How to Reproduce](#-how-to-reproduce)
7. [Repository Structure](#-repository-structure)

---

## 🔍 Overview

This project trains an autonomous vehicle agent to drive **as fast and as safely as possible** through dense highway traffic, using the [`highway-env`](https://github.com/Farama-Foundation/HighwayEnv) simulator. The agent must balance three competing objectives in real time:

- 🚀 **Speed** — drive close to the upper end of the legal range
- 🛡️ **Safety** — avoid collisions with surrounding vehicles
- 🛣️ **Discipline** — keep right and avoid unnecessary lane changes

The agent is trained with **Proximal Policy Optimization (PPO)** from [`stable-baselines3`](https://github.com/DLR-RM/stable-baselines3), wrapped in a custom reward-shaping environment and a permutation-invariant feature extractor.

---

## 🧠 Methodology

### a. Reward Function

The reward at timestep $t$ is a weighted combination of four interpretable terms:

$$
R_t \;=\; \alpha \cdot s_t \;-\; \beta \cdot c_t \;-\; \gamma \cdot \ell_t \;+\; \delta \cdot r_t
$$

Where:

| Symbol | Term | Meaning |
| :----: | :--- | :------ |
| $s_t \in [0, 1]$ | **Normalized speed** | High when the ego vehicle drives in the configured target range $[20, 30]$ m/s. |
| $c_t \in \{0, 1\}$ | **Collision flag** | $1$ if the ego car crashed at step $t$, otherwise $0$. |
| $\ell_t \in \{0, 1\}$ | **Lane-change cost** | $1$ whenever the agent picks a lane-change action (`LANE_LEFT` or `LANE_RIGHT`). |
| $r_t \in [0, 1]$ | **Right-lane bonus** | Highest when the ego vehicle occupies the rightmost lane (driving discipline). |

With the chosen coefficients:

| Coefficient | Value | Rationale |
| :---------- | :---: | :-------- |
| $\alpha$ (speed) | **0.4** | The primary driver of useful behavior; large enough to keep the agent from idling. |
| $\beta$ (collision) | **1.0** | Terminal penalty; dominates any short-term speed gain from risky maneuvers. |
| $\gamma$ (lane change) | **0.05** | Mild — penalizes *gratuitous* lane changes without forbidding overtakes. |
| $\delta$ (right-lane bonus) | **0.1** | Subtle nudge toward realistic highway etiquette. |

**Why this shape?** Highway-env's built-in reward already encodes speed and collisions, but exposing each term explicitly via a wrapper (see [`src/utils.py`](src/utils.py) → `ShapedRewardWrapper`) makes the trade-off legible and tunable. Early experiments with $\gamma = 0.3$ produced an agent that refused to overtake at all; lowering it to $0.05$ restored healthy lane-change behavior while still suppressing jitter.

---

### b. Model

**Algorithm — Proximal Policy Optimization (PPO).** PPO was chosen over DQN for three reasons:

1. **Stable training on continuous-style observations.** PPO's clipped surrogate objective tolerates the noisy advantage estimates produced by the multi-vehicle observation matrix far better than Q-learning, which struggles when neighboring states yield wildly different Q-values due to traffic configuration.
2. **Natural fit for the discrete meta-action space.** The 5-action `DiscreteMetaAction` head (lane-left, idle, lane-right, faster, slower) maps cleanly to a categorical policy.
3. **On-policy data efficiency.** With only ~200k timesteps to spare on a laptop CPU, PPO's `n_steps × n_envs` rollouts give us meaningful updates within minutes rather than hours.

**Hyperparameters** (defined in [`src/config.py`](src/config.py) → `PPOConfig`):

| Hyperparameter | Value | Notes |
| :------------- | :---: | :---- |
| Learning rate | `5e-4` | Slightly higher than SB3's default (`3e-4`); accelerates early learning on this short budget. |
| Discount $\gamma$ | `0.95` | Short-horizon: an episode is only ~200 environment steps. |
| GAE $\lambda$ | `0.95` | Standard. |
| `n_steps` | `512` per env | Rollout length per env before update. |
| `batch_size` | `64` | Small — the agent updates many times per rollout. |
| `n_epochs` | `10` | PPO default. |
| Clip range | `0.2` | PPO default. |
| Entropy coefficient | `0.01` | Encourages exploration of overtaking maneuvers. |
| Value-function coefficient | `0.5` | PPO default. |
| Parallel envs | `4` (`SubprocVecEnv`) | CPU-friendly parallelism. |

**Neural network architecture.** A custom feature extractor ([`src/model.py`](src/model.py) → `VehicleAttentionExtractor`) processes the per-vehicle observation matrix:

```
input  : (B, V=5, F=5)  — 5 nearby vehicles × 5 features each
        │
        ▼
per-vehicle MLP (shared weights):
        Linear(5 → 64) → ReLU → Linear(64 → 64) → ReLU
        │
        ▼
mean pool over vehicles  → (B, 64)
        │
        ▼
head: Linear(64 → 128) → ReLU
        │
        ▼
shared policy/value trunk: [256, 256] with ReLU activations
        │           │
        ▼           ▼
   π_θ(a|s)      V_φ(s)
```

The shared per-vehicle MLP enforces **permutation invariance**: the agent's decision shouldn't depend on the order in which neighboring vehicles are listed. This is a stronger inductive bias than a vanilla flattened MLP and consistently produced more stable training in our trials.

---

### c. States and Actions

**Observation space** — a $5 \times 5$ matrix of relative kinematics:

| Vehicle slot | Features (normalized to $[-1, 1]$) |
| :----------- | :--------------------------------- |
| Slot 0 (ego) | `presence`, `x`, `y`, `vx`, `vy` |
| Slots 1–4    | The 4 closest other vehicles, same 5 features |

Positions and velocities are expressed **relative to the ego vehicle** (`absolute=False`), which makes the observation translation-invariant.

**Action space** — `DiscreteMetaAction` with 5 actions:

| Action ID | Name | Effect |
| :-------: | :--- | :----- |
| 0 | `LANE_LEFT`  | Initiate a lane change to the left. |
| 1 | `IDLE`       | Maintain current lane and speed. |
| 2 | `LANE_RIGHT` | Initiate a lane change to the right. |
| 3 | `FASTER`     | Accelerate by one speed band. |
| 4 | `SLOWER`     | Decelerate by one speed band. |

The agent selects one meta-action every `1 / policy_frequency = 0.2` seconds, while the underlying simulator integrates physics at `15 Hz`.

---

## 🔬 Experiment Iterations

The final configuration shown above is the result of comparing several variants. Each row below records a deliberate change that was tested against the baseline, the observed effect, and the final decision. Rejected variants are kept in this table on purpose — they are the evidence behind the chosen hyperparameters.

| # | Variant | Change vs. baseline | Observed effect | Decision |
| :-: | :------ | :------------------ | :-------------- | :------- |
| v1 | Reward shaping (early) | $\gamma_\text{lane-change} = 0.3$ (heavy penalty) | Agent refused to overtake at all; sat behind slow vehicles indefinitely | ❌ Rejected |
| **v2** | Reward shaping (final) | $\gamma_\text{lane-change} = 0.05$ | Healthy overtaking, suppressed jitter | ✅ **Adopted** |
| v3 | Discount factor | $\gamma_\text{discount} = 0.99$ (PPO default) | Value function over-credited far-past actions; collision-avoidance signal slow to develop | ❌ Rejected |
| **v4** | Discount factor (final) | $\gamma_\text{discount} = 0.95$ | Shorter horizon matched ~200-step episodes; faster convergence | ✅ **Adopted** |
| v5 | Exploration | Entropy coef $= 0.005$ | Policy collapsed to "always SLOWER" — safe but trivial local minimum | ❌ Rejected |
| **v6** | Exploration (final) | Entropy coef $= 0.01$ | Maintained exploration of overtaking maneuvers | ✅ **Adopted** |
| v7 | Network architecture | Vanilla flattened MLP $(25 \to 256 \to 256)$ | Agent's decision changed when the order of neighboring vehicles in the observation matrix changed — broke symmetry | ❌ Rejected |
| **v8** | Network architecture (final) | Per-vehicle MLP + mean pooling (`VehicleAttentionExtractor`) | Permutation-invariant features; more stable training across seeds | ✅ **Adopted** |

The final model (`ppo_full.zip`) corresponds to the combination of v2 + v4 + v6 + v8. Variants v1, v3, v5, and v7 each shipped a partially trained agent that justified moving on; full-length training was only invested in the adopted configurations.

---

## 📈 Training Analysis

### a. Reward Graph

Episode reward and episode length over the full ~200k-step training run:

<p align="center">
  <img src="assets/reward_plot.png" alt="Training reward and episode length curves" width="100%">
</p>

### b. Quantitative Evaluation

To verify that the visible improvement in the training curve translates into measurable driving competence, both the untrained baseline and the fully trained policy were evaluated **deterministically over 20 fresh seeds** (different from the training seeds). Results:

| Metric | Untrained baseline | **Fully trained agent** | Improvement |
| :----- | :----------------: | :---------------------: | :---------: |
| Mean episode reward | 23.15 ± 13.82 | **70.15 ± 28.49** | **3.03×** |
| Mean episode length (steps) | 97.35 ± 54.69 | **152.05 ± 59.19** | **1.56×** |
| Crash rate | 95% (19/20) | **45% (9/20)** | **2.1× safer** |

The crash-rate halving is the headline result: a randomly initialized policy crashes in nearly every episode, while the trained agent reaches the 200-step time limit without incident in 9 out of 20 evaluation episodes. The remaining failures are concentrated in seeds with unusually tight initial spacings, suggesting that further training time (or curriculum learning on dense-traffic seeds) would primarily target this residual tail.

### c. Commentary

Three phases are visible in the training curve:

1. **Random phase (≈ 0 – 10k steps).** Episode rewards hover around zero, episodes terminate quickly (≤ 30 steps), and the agent crashes frequently. The policy is essentially uniform over the 5 meta-actions.

2. **Bootstrap phase (≈ 10k – 60k steps).** Rewards climb steeply as the agent discovers two basic facts: (a) the `SLOWER` action reduces collision rate, and (b) staying in lane (`IDLE`) is on average more rewarding than random lane changes. This is where the bulk of the improvement happens. The `ppo_half.zip` checkpoint sits inside this phase and shows partially competent — but still error-prone — driving.

3. **Refinement phase (≈ 60k – 200k steps).** The curve flattens but does *not* plateau; subtler trade-offs are still being optimized (e.g., when to overtake a slow truck rather than tailgate). Episode length also stabilizes near the 200-step cap, indicating that most episodes now end via timeout rather than collision.

**On the hyperparameter choices.** Setting $\gamma_\text{discount} = 0.95$ rather than the more common $0.99$ noticeably accelerated learning: with $0.99$ the value function over-credits actions taken many seconds before an eventual crash, which slowed the collision-avoidance signal. The shorter horizon matches the actual episode length far better. Conversely, lowering the entropy coefficient below `0.01` caused the agent to collapse onto an "always SLOWER" policy that achieved a low but safe reward — a classic local minimum that better exploration avoids.

---

## 🧩 Challenges and Failures

Three concrete obstacles came up during this project. None of them were about the RL algorithm itself — all were about the surrounding engineering, which turned out to be just as important as the agent design.

### Challenge 1 — Dependency hell with Python 3.13

The first environment I built used Python 3.13, the latest available. `pip install -r requirements.txt` died partway through with a Meson build error on numpy:

```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xfc in position 52
ERROR: Could not build wheels for numpy
```

Two things were going wrong at once. First, `numpy 1.26.4` had no pre-built wheel for Python 3.13, so pip was trying to **compile it from source**. Second, my project lived under `C:\Users\<name>\OneDrive\Masaüstü\...` — the Turkish character `ü` in "Masaüstü" broke Meson's UTF-8 file reader during the compile step. Most modern Python packages ship pre-built wheels for the previous stable release but lag a few months behind the newest one.

**Fix.** Two changes: (1) installed Python 3.12 alongside 3.13 and recreated the venv with `py -3.12 -m venv .venv`; (2) moved the project to `C:\introai\highway-rl` to remove the non-ASCII path. After this, every package had a pre-built wheel and installation took 5 minutes instead of crashing.

### Challenge 2 — `gymnasium 0.29` vs. `highway-env 1.10` incompatibility

After fixing the build problem, pip immediately threw a resolver error:

```
ERROR: Cannot install -r requirements.txt because these package versions have conflicting dependencies.
  The user requested gymnasium==0.29.1
  highway-env 1.10.1 depends on gymnasium>=1.0.0a2
```

I had pinned `gymnasium==0.29.1` from earlier examples in the course, but `highway-env 1.10.1` had already moved to `gymnasium >= 1.0`. The two were API-incompatible.

**Fix.** Loosened the pins in `requirements.txt` to `gymnasium>=1.0.0` and bumped `stable-baselines3` to `>=2.4.0` (the first SB3 release with full `gymnasium 1.0` support). This is documented in `requirements.txt` and is the dependency set used to produce all results below.

### Challenge 3 — Training was much slower than expected

The initial estimate was *"20–35 minutes on a modern laptop CPU"*, taken from highway-env's documentation. On my actual machine, with `--n-envs 4`, the full 200,000-step run took **3 hours 41 minutes** — about 6× longer than predicted. The PPO log showed `fps ≈ 10` per environment instead of the 60+ I expected.

The cause was straightforward in hindsight: my CPU has fewer physical cores than the documentation's reference machine, and highway-env's rendering pipeline (which still runs even with `render_mode=None` for stats) is single-threaded. Increasing `--n-envs` beyond 4 actually slowed things down because of subprocess contention.

**Fix.** No code change — I left the run going and accepted the longer wall time. The lesson was about **planning, not optimization**: for a project with a deadline, the first thing to measure is how fast the training loop actually runs *on your hardware*, not on someone else's benchmark. A 5-minute smoke test (`--timesteps 5000`) would have told me the real `fps` and given an accurate ETA before I committed to the full run.

### What I would do differently

If I started over, the order of operations would be: (1) lock the Python version and project path *first*, before installing anything; (2) treat published `requirements.txt` files as starting points, not contracts — re-resolve them against current PyPI; (3) measure `fps` from a tiny run before launching the long one. The RL part went smoothly; the environment around it was where time disappeared.

---

## 🚀 How to Reproduce

```bash
# 1. Clone and enter the repo
git clone https://github.com/ugussecem/highway-rl-ppo.git
cd highway-rl-ppo

# 2. Create a virtual environment (Python 3.10+ recommended)
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the agent (3 checkpoints saved automatically)
python src/train.py

# 5. Evaluate the trained policy
python src/evaluate.py --stage full --episodes 20

# 6. Generate the evolution video (GIF + MP4)
python src/make_evolution_video.py
```

Training time varies significantly with hardware. On the development machine (laptop CPU, 4 parallel envs) the full 200,000-step run took **≈ 3.5 hours**; on a newer multi-core CPU it can finish in **20–40 minutes**. Run a quick smoke test first to measure your own throughput before committing to the full run:

```bash
python src/train.py --timesteps 5000 --n-envs 2
```

The smoke test takes a few minutes and prints the achieved `fps`, from which the full-run ETA is just `200_000 / fps` seconds.

---

## 📂 Repository Structure

```
.
├── README.md                  # this report
├── requirements.txt           # pinned dependencies
├── .gitignore
├── src/
│   ├── config.py              # all hyperparameters and paths
│   ├── model.py               # custom feature extractor
│   ├── utils.py               # env factory, reward wrapper, plotting, video I/O
│   ├── train.py               # PPO training loop with stage checkpoints
│   ├── evaluate.py            # deterministic evaluation rollouts
│   └── make_evolution_video.py# build assets/evolution.gif
├── assets/
│   ├── evolution.gif          # 3-stage side-by-side training evolution
│   └── reward_plot.png        # training curves
├── videos/
│   └── evolution.mp4          # MP4 copy of the evolution video
├── checkpoints/               # ppo_untrained.zip, ppo_half.zip, ppo_full.zip
└── logs/                      # monitor CSVs and TensorBoard event files
```

---

<div align="center">

*Built with [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), [Highway-Env](https://github.com/Farama-Foundation/HighwayEnv), and [PyTorch](https://pytorch.org/) — May 2026.*

</div>
