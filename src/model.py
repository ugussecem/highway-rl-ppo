"""Custom neural network components for the Highway-Env PPO agent.

Highway-env returns a flattened (vehicles_count, features) matrix when
configured with the Kinematics observation. PPO's default MlpPolicy
handles this well, but a small custom feature extractor lets us:

  * apply the same per-vehicle linear projection (weight sharing), and
  * keep the architecture explicit and reportable in the README.
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class VehicleAttentionExtractor(BaseFeaturesExtractor):
    """Per-vehicle MLP encoder followed by mean pooling.

    The input is shape (B, V, F) where V is the number of nearby vehicles
    and F the number of per-vehicle features. We embed each vehicle row
    independently with a shared MLP, then pool across the vehicle axis to
    produce a permutation-invariant summary of traffic around the ego car.

    Parameters
    ----------
    observation_space:
        The flattened Box space coming from highway-env.
    features_dim:
        Output dimensionality consumed by the PPO policy head.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 128) -> None:
        super().__init__(observation_space, features_dim)

        # highway-env flattens the (V, F) observation to (V * F,). We need
        # the original shape to apply the per-vehicle MLP.
        if len(observation_space.shape) == 2:
            self._n_vehicles, self._n_features = observation_space.shape
        else:
            # Fallback: assume 5 vehicles x 5 features (default config).
            total = int(observation_space.shape[0])
            self._n_vehicles = 5
            self._n_features = total // self._n_vehicles

        hidden = 64
        self._per_vehicle = nn.Sequential(
            nn.Linear(self._n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self._head = nn.Sequential(
            nn.Linear(hidden, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Re-introduce the (V, F) axis if the obs was flattened by SB3.
        if observations.ndim == 2:
            observations = observations.view(-1, self._n_vehicles, self._n_features)

        # (B, V, F) -> (B, V, hidden) via shared MLP.
        embedded = self._per_vehicle(observations)

        # Mean pool across vehicles -> (B, hidden).
        pooled = embedded.mean(dim=1)

        return self._head(pooled)


def build_policy_kwargs(net_arch: tuple[int, ...]) -> dict:
    """Assemble policy_kwargs for stable-baselines3 PPO.

    Returns
    -------
    dict
        Kwargs accepted by ``PPO(..., policy_kwargs=...)``.
    """
    return {
        "features_extractor_class": VehicleAttentionExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
        "net_arch": list(net_arch),
        "activation_fn": nn.ReLU,
    }
