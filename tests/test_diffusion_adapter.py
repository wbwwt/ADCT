import torch

from adct.adapters import ExperienceTreeDiffusionConditioner


def test_diffusion_conditioner_output_shape() -> None:
    module = ExperienceTreeDiffusionConditioner(
        state_dim=6,
        num_classes=3,
        feature_dim=32,
    )
    result = module(
        state=torch.zeros(2, 6),
        labels=torch.tensor([[0], [1]]),
        boxes=torch.zeros(2, 1, 4),
        detection_mask=torch.ones(2, 1, dtype=torch.bool),
    )
    assert result.shape == (2, 38)
    assert module.output_dim == 38

