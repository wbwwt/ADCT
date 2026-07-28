"""Frozen RT-DETR inference adapter used by ADCT."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Protocol

import torch
import torchvision.transforms.functional as vision_f
from torch import Tensor, nn

from adct.types import Detection


class Detector(Protocol):
    def predict(self, images: Tensor) -> list[list[Detection]]:
        """Predict one detection list for each ``BCHW`` input image."""


def _load_torch_checkpoint(path: Path, device: torch.device) -> dict:
    # RT-DETR checkpoints contain nested Python dictionaries, so the restricted
    # weights-only loader cannot deserialize the official format.
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch 2.2 compatibility.
        return torch.load(path, map_location=device)


class RTDETRDetector(nn.Module):
    """Inference-only wrapper around the vendored official RT-DETR code."""

    def __init__(
        self,
        config_path: str | Path,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda",
        input_size: int = 640,
        score_threshold: float = 0.30,
        top_k: int = 6,
        use_amp: bool = True,
    ) -> None:
        super().__init__()
        if input_size < 1 or top_k < 1:
            raise ValueError("input_size and top_k must be positive.")
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be in [0, 1].")

        try:
            from RT_DETR.src.core import YAMLConfig
        except ImportError as error:
            raise ImportError(
                "The RT_DETR package is unavailable. Install this repository in editable mode "
                "or restore src/RT_DETR from the official RT-DETR source."
            ) from error

        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for RT-DETR but is not available.")
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.top_k = top_k
        self.use_amp = use_amp and self.device.type == "cuda"

        config_path = Path(config_path).expanduser().resolve()
        checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"RT-DETR config not found: {config_path}")
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"RT-DETR checkpoint not found: {checkpoint_path}")

        cfg = YAMLConfig(str(config_path), resume=str(checkpoint_path))
        checkpoint = _load_torch_checkpoint(checkpoint_path, self.device)
        state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
        cfg.model.load_state_dict(state)
        self.model = cfg.model.deploy().to(self.device)
        self.postprocessor = cfg.postprocessor.deploy().to(self.device)

        self.model.eval()
        self.postprocessor.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def predict(self, images: Tensor) -> list[list[Detection]]:
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (B, 3, H, W) or (3, H, W).")

        images = images.to(device=self.device, dtype=torch.float32)
        batch_size, _, height, width = images.shape
        original_sizes = torch.tensor(
            [[width, height]] * batch_size,
            dtype=torch.float32,
            device=self.device,
        )
        resized = vision_f.resize(
            images,
            [self.input_size, self.input_size],
            antialias=True,
        )
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.use_amp
            else nullcontext()
        )
        with autocast:
            outputs = self.model(resized)
            labels, boxes, scores = self.postprocessor(outputs, original_sizes)

        results: list[list[Detection]] = []
        for image_labels, image_boxes, image_scores in zip(
            labels,
            boxes,
            scores,
            strict=True,
        ):
            kept: list[Detection] = []
            for label, box, score in zip(
                image_labels[: self.top_k],
                image_boxes[: self.top_k],
                image_scores[: self.top_k],
                strict=True,
            ):
                confidence = float(score)
                if confidence < self.score_threshold:
                    continue
                kept.append(
                    Detection(
                        label=int(label),
                        box=tuple(float(value) for value in box),
                        score=confidence,
                    )
                )
            results.append(kept)
        return results

    def forward(self, images: Tensor) -> Sequence[Sequence[Detection]]:
        return self.predict(images)

