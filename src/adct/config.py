"""Typed configuration objects used by the ADCT reference implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TargetRegionConfig:
    """Axis-aligned tolerance region around the placement target, in pixels."""

    center_x: float
    center_y: float
    tolerance_x: float = 25.0
    tolerance_y: float = 25.0

    def __post_init__(self) -> None:
        if self.tolerance_x <= 0 or self.tolerance_y <= 0:
            raise ValueError("Target-region tolerances must be positive.")

    def contains(self, center_x: float, center_y: float) -> bool:
        return (
            abs(center_x - self.center_x) < self.tolerance_x
            and abs(center_y - self.center_y) < self.tolerance_y
        )


@dataclass(frozen=True)
class ExperienceTreeConfig:
    target_region: TargetRegionConfig
    min_confidence: float = 0.30

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1].")


@dataclass(frozen=True)
class DynamicHorizonConfig:
    chunk_size: int = 50
    confidence_threshold: float = 0.93
    scaling_factor: float = 50.0
    min_horizon: int = 1

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1].")
        if self.scaling_factor < 0:
            raise ValueError("scaling_factor must be non-negative.")
        if not 1 <= self.min_horizon <= self.chunk_size:
            raise ValueError("min_horizon must be in [1, chunk_size].")


@dataclass(frozen=True)
class ADCTModelConfig:
    state_dim: int
    action_dim: int
    num_classes: int
    chunk_size: int = 50
    max_detections: int = 1
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    n_vae_encoder_layers: int = 4
    latent_dim: int = 32
    dropout: float = 0.1
    kl_weight: float = 1.0

    def __post_init__(self) -> None:
        positive = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "num_classes": self.num_classes,
            "chunk_size": self.chunk_size,
            "max_detections": self.max_detections,
            "dim_model": self.dim_model,
            "n_heads": self.n_heads,
            "dim_feedforward": self.dim_feedforward,
            "n_encoder_layers": self.n_encoder_layers,
            "n_decoder_layers": self.n_decoder_layers,
            "n_vae_encoder_layers": self.n_vae_encoder_layers,
            "latent_dim": self.latent_dim,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}.")
        if self.dim_model % self.n_heads != 0:
            raise ValueError("dim_model must be divisible by n_heads.")
        if self.dim_model % 2 != 0:
            raise ValueError("dim_model must be even for balanced label/box encoding.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.kl_weight < 0:
            raise ValueError("kl_weight must be non-negative.")


@dataclass(frozen=True)
class DetectorConfig:
    config: str
    checkpoint: str
    device: str = "cuda"
    input_size: int = 640
    score_threshold: float = 0.30
    top_k: int = 6
    use_amp: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    epochs: int = 200
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda"


@dataclass(frozen=True)
class ExperimentConfig:
    model: ADCTModelConfig
    experience_tree: ExperienceTreeConfig
    dynamic_horizon: DynamicHorizonConfig
    detector: DetectorConfig | None = None
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        tree_data = dict(data["experience_tree"])
        tree_data["target_region"] = TargetRegionConfig(**tree_data["target_region"])
        return cls(
            model=ADCTModelConfig(**data["model"]),
            experience_tree=ExperienceTreeConfig(**tree_data),
            dynamic_horizon=DynamicHorizonConfig(**data["dynamic_horizon"]),
            detector=DetectorConfig(**data["detector"]) if data.get("detector") else None,
            training=TrainingConfig(**data.get("training", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a mapping in {path}.")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_project_path(path: str | Path, project_root: str | Path) -> Path:
    """Resolve configuration paths without depending on the process working directory."""

    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else Path(project_root) / candidate

