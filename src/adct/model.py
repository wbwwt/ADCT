"""Image-free ACT policy conditioned on experience-tree detection primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from adct.config import ADCTModelConfig
from adct.features import DetectionFeatureEncoder
from adct.normalization import NormalizationStats


def _sinusoidal_encoding(length: int, dimension: int) -> Tensor:
    if dimension % 2:
        raise ValueError("Sinusoidal encoding requires an even dimension.")
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, dimension, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / dimension)
    )
    encoding = torch.zeros(length, dimension, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies)
    return encoding


@dataclass
class ADCTOutput:
    actions: Tensor
    mean: Tensor | None
    log_variance: Tensor | None


class ADCT(nn.Module):
    """Action and Detection Chunking with Transformers.

    RGB image embeddings are deliberately absent. A frozen detector and an
    experience tree produce labels and normalized boxes before this model is
    called.
    """

    def __init__(self, config: ADCTModelConfig) -> None:
        super().__init__()
        self.config = config
        self.normalization_stats: NormalizationStats | None = None

        self.detection_encoder = DetectionFeatureEncoder(
            num_classes=config.num_classes,
            dim_model=config.dim_model,
        )
        self.state_projection = nn.Linear(config.state_dim, config.dim_model)
        self.latent_projection = nn.Linear(config.latent_dim, config.dim_model)
        self.encoder_positions = nn.Embedding(
            2 + config.max_detections,
            config.dim_model,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.policy_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_encoder_layers,
            norm=nn.LayerNorm(config.dim_model),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.dim_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.policy_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.n_decoder_layers,
            norm=nn.LayerNorm(config.dim_model),
        )
        self.action_queries = nn.Embedding(config.chunk_size, config.dim_model)
        self.action_head = nn.Linear(config.dim_model, config.action_dim)

        # CVAE style encoder: [CLS, robot state, action chunk].
        self.vae_cls = nn.Parameter(torch.empty(1, 1, config.dim_model))
        self.vae_state_projection = nn.Linear(config.state_dim, config.dim_model)
        self.vae_action_projection = nn.Linear(config.action_dim, config.dim_model)
        vae_layer = nn.TransformerEncoderLayer(
            d_model=config.dim_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.vae_encoder = nn.TransformerEncoder(
            vae_layer,
            num_layers=config.n_vae_encoder_layers,
            norm=nn.LayerNorm(config.dim_model),
        )
        self.latent_distribution = nn.Linear(config.dim_model, 2 * config.latent_dim)
        self.register_buffer(
            "vae_positions",
            _sinusoidal_encoding(config.chunk_size + 2, config.dim_model).unsqueeze(0),
            persistent=True,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.vae_cls, std=0.02)
        nn.init.normal_(self.action_queries.weight, std=0.02)
        nn.init.normal_(self.encoder_positions.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _encode_latent(
        self,
        state: Tensor,
        actions: Tensor | None,
        action_is_pad: Tensor | None,
    ) -> tuple[Tensor, Tensor | None, Tensor | None]:
        batch_size = state.shape[0]
        if actions is None:
            latent = torch.zeros(
                batch_size,
                self.config.latent_dim,
                dtype=state.dtype,
                device=state.device,
            )
            return latent, None, None

        expected_shape = (batch_size, self.config.chunk_size, self.config.action_dim)
        if tuple(actions.shape) != expected_shape:
            raise ValueError(
                f"Expected actions with shape {expected_shape}, got {tuple(actions.shape)}."
            )
        if action_is_pad is None:
            action_is_pad = torch.zeros(
                batch_size,
                self.config.chunk_size,
                dtype=torch.bool,
                device=state.device,
            )
        if tuple(action_is_pad.shape) != expected_shape[:2]:
            raise ValueError(
                f"Expected action_is_pad with shape {expected_shape[:2]}, "
                f"got {tuple(action_is_pad.shape)}."
            )

        cls_token = self.vae_cls.expand(batch_size, -1, -1)
        state_token = self.vae_state_projection(state).unsqueeze(1)
        action_tokens = self.vae_action_projection(actions)
        tokens = torch.cat([cls_token, state_token, action_tokens], dim=1)
        tokens = tokens + self.vae_positions.to(dtype=tokens.dtype)

        prefix_mask = torch.zeros(batch_size, 2, dtype=torch.bool, device=state.device)
        padding_mask = torch.cat([prefix_mask, action_is_pad.to(torch.bool)], dim=1)
        encoded = self.vae_encoder(tokens, src_key_padding_mask=padding_mask)
        parameters = self.latent_distribution(encoded[:, 0])
        mean, log_variance = parameters.chunk(2, dim=-1)
        std = torch.exp(0.5 * log_variance)
        latent = mean + std * torch.randn_like(std)
        return latent, mean, log_variance

    def forward(
        self,
        state: Tensor,
        labels: Tensor,
        boxes: Tensor,
        detection_mask: Tensor | None = None,
        *,
        actions: Tensor | None = None,
        action_is_pad: Tensor | None = None,
    ) -> ADCTOutput:
        if state.ndim != 2 or state.shape[-1] != self.config.state_dim:
            raise ValueError(
                f"state must have shape (batch, {self.config.state_dim}), got {tuple(state.shape)}."
            )
        batch_size = state.shape[0]
        expected_labels = (batch_size, self.config.max_detections)
        expected_boxes = (*expected_labels, 4)
        if tuple(labels.shape) != expected_labels:
            raise ValueError(f"Expected labels shape {expected_labels}, got {tuple(labels.shape)}.")
        if tuple(boxes.shape) != expected_boxes:
            raise ValueError(f"Expected boxes shape {expected_boxes}, got {tuple(boxes.shape)}.")
        if detection_mask is None:
            detection_mask = torch.ones(expected_labels, dtype=torch.bool, device=state.device)
        if tuple(detection_mask.shape) != expected_labels:
            raise ValueError(
                f"Expected detection_mask shape {expected_labels}, "
                f"got {tuple(detection_mask.shape)}."
            )

        latent, mean, log_variance = self._encode_latent(state, actions, action_is_pad)
        latent_token = self.latent_projection(latent).unsqueeze(1)
        state_token = self.state_projection(state).unsqueeze(1)
        detection_tokens = self.detection_encoder(labels, boxes)
        encoder_tokens = torch.cat([latent_token, state_token, detection_tokens], dim=1)
        positions = self.encoder_positions.weight.unsqueeze(0)
        encoder_tokens = encoder_tokens + positions

        prefix_mask = torch.zeros(batch_size, 2, dtype=torch.bool, device=state.device)
        encoder_padding_mask = torch.cat([prefix_mask, ~detection_mask.to(torch.bool)], dim=1)
        memory = self.policy_encoder(
            encoder_tokens,
            src_key_padding_mask=encoder_padding_mask,
        )

        queries = self.action_queries.weight.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.policy_decoder(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=encoder_padding_mask,
        )
        return ADCTOutput(
            actions=self.action_head(decoded),
            mean=mean,
            log_variance=log_variance,
        )

    def compute_loss(
        self,
        output: ADCTOutput,
        target_actions: Tensor,
        action_is_pad: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if output.mean is None or output.log_variance is None:
            raise ValueError("CVAE loss requires a training forward pass with target actions.")
        if target_actions.shape != output.actions.shape:
            raise ValueError("Target and predicted action chunks must have the same shape.")
        if action_is_pad is None:
            action_is_pad = torch.zeros(
                target_actions.shape[:2],
                dtype=torch.bool,
                device=target_actions.device,
            )

        valid = (~action_is_pad.to(torch.bool)).unsqueeze(-1).expand_as(target_actions)
        squared_error = F.mse_loss(output.actions, target_actions, reduction="none")
        reconstruction = squared_error.masked_select(valid).mean()
        kl = -0.5 * (
            1.0 + output.log_variance - output.mean.square() - output.log_variance.exp()
        ).sum(dim=-1).mean()
        total = reconstruction + self.config.kl_weight * kl
        return total, {
            "loss": total.detach(),
            "reconstruction_loss": reconstruction.detach(),
            "kl_loss": kl.detach(),
        }

    def parameter_counts(self) -> dict[str, int]:
        return {
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "trainable": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        }

