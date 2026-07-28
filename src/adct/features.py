"""Balanced semantic and spatial detection-feature encoding."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def normalize_boxes_xyxy(
    boxes: Tensor,
    *,
    image_width: int | float,
    image_height: int | float,
) -> Tensor:
    """Map pixel-space ``xyxy`` boxes to the interval ``[-1, 1]``."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")
    normalized = boxes.to(dtype=torch.float32).clone()
    normalized[..., (0, 2)] = normalized[..., (0, 2)] / float(image_width) * 2.0 - 1.0
    normalized[..., (1, 3)] = normalized[..., (1, 3)] / float(image_height) * 2.0 - 1.0
    return normalized


class DetectionFeatureEncoder(nn.Module):
    """Encode class one-hot vectors and boxes separately, then concatenate."""

    def __init__(self, num_classes: int, dim_model: int) -> None:
        super().__init__()
        if num_classes < 1 or dim_model < 2 or dim_model % 2:
            raise ValueError(
                "num_classes must be positive and dim_model must be positive and even."
            )
        half = dim_model // 2
        self.num_classes = num_classes
        self.label_projection = nn.Linear(num_classes, half)
        self.box_projection = nn.Linear(4, half)

    def forward(self, labels: Tensor, boxes: Tensor) -> Tensor:
        if labels.shape != boxes.shape[:-1]:
            raise ValueError(
                f"labels shape {labels.shape} must match boxes prefix {boxes.shape[:-1]}."
            )
        if boxes.shape[-1] != 4:
            raise ValueError("boxes must have four coordinates.")
        if torch.any(labels < 0) or torch.any(labels >= self.num_classes):
            raise ValueError(f"labels must be in [0, {self.num_classes - 1}].")
        one_hot = F.one_hot(labels.to(torch.long), num_classes=self.num_classes).to(boxes.dtype)
        return torch.cat(
            [self.label_projection(one_hot), self.box_projection(boxes)],
            dim=-1,
        )

