import torch

from adct.config import ADCTModelConfig
from adct.model import ADCT


def small_config() -> ADCTModelConfig:
    return ADCTModelConfig(
        state_dim=6,
        action_dim=6,
        num_classes=3,
        chunk_size=4,
        max_detections=1,
        dim_model=32,
        n_heads=4,
        dim_feedforward=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_vae_encoder_layers=1,
        latent_dim=8,
        dropout=0.0,
    )


def test_training_forward_and_loss_backward() -> None:
    model = ADCT(small_config())
    state = torch.randn(2, 6)
    labels = torch.tensor([[0], [2]])
    boxes = torch.rand(2, 1, 4) * 2 - 1
    actions = torch.randn(2, 4, 6)
    is_pad = torch.tensor([[False, False, False, False], [False, False, True, True]])

    output = model(
        state,
        labels,
        boxes,
        actions=actions,
        action_is_pad=is_pad,
    )
    assert output.actions.shape == actions.shape
    loss, metrics = model.compute_loss(output, actions, is_pad)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(metrics) == {"loss", "reconstruction_loss", "kl_loss"}


def test_inference_uses_zero_latent() -> None:
    model = ADCT(small_config()).eval()
    with torch.inference_mode():
        output = model(
            state=torch.zeros(1, 6),
            labels=torch.zeros(1, 1, dtype=torch.long),
            boxes=torch.zeros(1, 1, 4),
        )
    assert output.actions.shape == (1, 4, 6)
    assert output.mean is None
    assert output.log_variance is None

