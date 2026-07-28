import torch

from adct.normalization import NormalizationStats


def test_normalization_round_trip() -> None:
    stats = NormalizationStats(
        state_mean=(1.0, 2.0),
        state_std=(2.0, 4.0),
        action_mean=(3.0, 4.0),
        action_std=(0.5, 2.0),
    )
    state = torch.tensor([[3.0, 6.0]])
    action = torch.tensor([[4.0, 8.0]])

    assert torch.allclose(stats.normalize_state(state), torch.ones_like(state))
    assert torch.allclose(
        stats.unnormalize_action(stats.normalize_action(action)),
        action,
    )
