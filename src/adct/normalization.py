"""Mean/std normalization compatible with the experimental LeRobot pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class NormalizationStats:
    state_mean: tuple[float, ...]
    state_std: tuple[float, ...]
    action_mean: tuple[float, ...]
    action_std: tuple[float, ...]

    @classmethod
    def from_numpy(
        cls,
        *,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
    ) -> NormalizationStats:
        return cls(
            state_mean=tuple(float(value) for value in state_mean),
            state_std=tuple(float(value) for value in state_std),
            action_mean=tuple(float(value) for value in action_mean),
            action_std=tuple(float(value) for value in action_std),
        )

    def _tensor(self, values: tuple[float, ...], reference: Tensor) -> Tensor:
        return torch.as_tensor(values, dtype=reference.dtype, device=reference.device)

    def normalize_state(self, state: Tensor) -> Tensor:
        return (state - self._tensor(self.state_mean, state)) / self._tensor(
            self.state_std,
            state,
        )

    def normalize_action(self, action: Tensor) -> Tensor:
        return (action - self._tensor(self.action_mean, action)) / self._tensor(
            self.action_std,
            action,
        )

    def unnormalize_action(self, action: Tensor) -> Tensor:
        return action * self._tensor(self.action_std, action) + self._tensor(
            self.action_mean,
            action,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> NormalizationStats:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            state_mean=tuple(data["state_mean"]),
            state_std=tuple(data["state_std"]),
            action_mean=tuple(data["action_mean"]),
            action_std=tuple(data["action_std"]),
        )

