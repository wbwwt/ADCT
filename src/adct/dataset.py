"""Prepared, detector-free training dataset for ADCT."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from adct.config import ADCTModelConfig
from adct.normalization import NormalizationStats

REQUIRED_ARRAYS = {
    "states",
    "actions",
    "action_is_pad",
    "labels",
    "boxes",
    "scores",
    "detection_mask",
    "state_mean",
    "state_std",
    "action_mean",
    "action_std",
}


class PreparedADCTDataset(Dataset[dict[str, Tensor]]):
    """Load semantic primitives and action chunks stored in one ``.npz`` file."""

    def __init__(self, path: str | Path, model_config: ADCTModelConfig) -> None:
        self.path = Path(path)
        archive = np.load(self.path, allow_pickle=False)
        missing = REQUIRED_ARRAYS - set(archive.files)
        if missing:
            raise ValueError(f"{self.path} is missing arrays: {sorted(missing)}")
        self.arrays = {name: archive[name] for name in REQUIRED_ARRAYS}
        self.stats = NormalizationStats.from_numpy(
            state_mean=self.arrays["state_mean"],
            state_std=self.arrays["state_std"],
            action_mean=self.arrays["action_mean"],
            action_std=self.arrays["action_std"],
        )
        self._validate(model_config)

    def _validate(self, config: ADCTModelConfig) -> None:
        count = self.arrays["states"].shape[0]
        expected = {
            "states": (count, config.state_dim),
            "actions": (count, config.chunk_size, config.action_dim),
            "action_is_pad": (count, config.chunk_size),
            "labels": (count, config.max_detections),
            "boxes": (count, config.max_detections, 4),
            "scores": (count, config.max_detections),
            "detection_mask": (count, config.max_detections),
            "state_mean": (config.state_dim,),
            "state_std": (config.state_dim,),
            "action_mean": (config.action_dim,),
            "action_std": (config.action_dim,),
        }
        for name, shape in expected.items():
            if self.arrays[name].shape != shape:
                raise ValueError(
                    f"{name} has shape {self.arrays[name].shape}; expected {shape}."
                )

    def __len__(self) -> int:
        return self.arrays["states"].shape[0]

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        state = self.stats.normalize_state(
            torch.as_tensor(self.arrays["states"][index], dtype=torch.float32)
        )
        actions = self.stats.normalize_action(
            torch.as_tensor(self.arrays["actions"][index], dtype=torch.float32)
        )
        action_is_pad = torch.as_tensor(
            self.arrays["action_is_pad"][index],
            dtype=torch.bool,
        )
        actions[action_is_pad] = 0.0
        return {
            "state": state,
            "actions": actions,
            "action_is_pad": action_is_pad,
            "labels": torch.as_tensor(self.arrays["labels"][index], dtype=torch.long),
            "boxes": torch.as_tensor(self.arrays["boxes"][index], dtype=torch.float32),
            "scores": torch.as_tensor(self.arrays["scores"][index], dtype=torch.float32),
            "detection_mask": torch.as_tensor(
                self.arrays["detection_mask"][index],
                dtype=torch.bool,
            ),
        }

