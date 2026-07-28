"""CPU-only smoke example for the ADCT runtime."""

from __future__ import annotations

import torch

from adct.config import (
    ADCTModelConfig,
    DynamicHorizonConfig,
    ExperienceTreeConfig,
    TargetRegionConfig,
)
from adct.experience_tree import ExperienceTree
from adct.model import ADCT
from adct.runtime import ADCTRuntime
from adct.types import Detection


class MockDetector:
    def predict(self, images: torch.Tensor) -> list[list[Detection]]:
        return [[Detection(label=0, box=(10, 10, 20, 20), score=0.95)]]


tree_config = ExperienceTreeConfig(
    target_region=TargetRegionConfig(50, 50, 5, 5),
)
tree = ExperienceTree(tree_config).fit(
    [
        [
            [Detection(0, (10, 10, 20, 20), 0.95)],
            [Detection(0, (48, 48, 52, 52), 0.95)],
        ]
    ]
)
model = ADCT(
    ADCTModelConfig(
        state_dim=6,
        action_dim=6,
        num_classes=1,
        chunk_size=4,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=8,
    )
).eval()
runtime = ADCTRuntime(
    model,
    MockDetector(),
    tree,
    DynamicHorizonConfig(chunk_size=4, confidence_threshold=0.93, scaling_factor=4),
    device="cpu",
)

image = torch.zeros(3, 64, 64)
robot_state = torch.zeros(6)
print(runtime.step(image, robot_state))

