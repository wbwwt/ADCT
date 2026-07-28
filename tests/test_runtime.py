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


class FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, images: torch.Tensor) -> list[list[Detection]]:
        self.calls += 1
        return [[Detection(label=0, box=(10, 10, 20, 20), score=0.5)]]


def test_runtime_replans_after_dynamic_prefix() -> None:
    tree_config = ExperienceTreeConfig(
        target_region=TargetRegionConfig(50, 50, 5, 5),
        min_confidence=0.3,
    )
    tree = ExperienceTree(tree_config).fit(
        [
            [
                [Detection(0, (10, 10, 20, 20), 0.9)],
                [Detection(0, (48, 48, 52, 52), 0.9)],
            ]
        ]
    )
    model = ADCT(
        ADCTModelConfig(
            state_dim=2,
            action_dim=2,
            num_classes=1,
            chunk_size=4,
            dim_model=16,
            n_heads=4,
            dim_feedforward=32,
            n_encoder_layers=1,
            n_decoder_layers=1,
            n_vae_encoder_layers=1,
            latent_dim=4,
            dropout=0,
        )
    )
    detector = FakeDetector()
    runtime = ADCTRuntime(
        model,
        detector,
        tree,
        DynamicHorizonConfig(
            chunk_size=4,
            confidence_threshold=0.9,
            scaling_factor=4,
        ),
        device="cpu",
    )
    image = torch.zeros(3, 64, 64)
    state = torch.zeros(2)

    runtime.step(image, state)
    runtime.step(image, state)
    assert detector.calls == 1
    runtime.step(image, state)
    assert detector.calls == 2

