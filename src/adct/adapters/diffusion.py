"""Experience-tree conditioning hook for Diffusion Policy implementations."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from adct.features import DetectionFeatureEncoder


class ExperienceTreeDiffusionConditioner(nn.Module):
    """Build a compact global-conditioning vector for a diffusion policy.

    The returned vector can replace or augment an RGB encoder's global
    conditioning. This module intentionally does not depend on a specific
    Diffusion Policy implementation.
    """

    def __init__(self, state_dim: int, num_classes: int, feature_dim: int = 128) -> None:
        super().__init__()
        if state_dim < 1:
            raise ValueError("state_dim must be positive.")
        self.state_dim = state_dim
        self.detection_encoder = DetectionFeatureEncoder(num_classes, feature_dim)

    @property
    def output_dim(self) -> int:
        return self.state_dim + self.detection_encoder.label_projection.out_features * 2

    def forward(
        self,
        state: Tensor,
        labels: Tensor,
        boxes: Tensor,
        detection_mask: Tensor,
    ) -> Tensor:
        encoded = self.detection_encoder(labels, boxes)
        mask = detection_mask.to(dtype=encoded.dtype).unsqueeze(-1)
        pooled = (encoded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return torch.cat([state, pooled], dim=-1)

