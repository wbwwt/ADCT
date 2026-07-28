"""Shared, dependency-light data types."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Detection:
    """One detector primitive using an ``xyxy`` pixel-space bounding box."""

    label: int
    box: tuple[float, float, float, float]
    score: float

    def __post_init__(self) -> None:
        if self.label < 0:
            raise ValueError("Detection labels must be non-negative.")
        if len(self.box) != 4:
            raise ValueError("A detection box must contain four coordinates.")
        x1, y1, x2, y2 = self.box
        if x2 < x1 or y2 < y1:
            raise ValueError(f"Invalid xyxy box: {self.box}.")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Detection scores must be in [0, 1].")

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return (0.5 * (x1 + x2), 0.5 * (y1 + y2))

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "box": list(self.box), "score": self.score}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Detection:
        box: Sequence[float] = data["box"]
        return cls(
            label=int(data["label"]),
            box=tuple(float(value) for value in box),
            score=float(data["score"]),
        )


@dataclass(frozen=True)
class ManifestFrame:
    episode_index: int
    frame_index: int
    state: tuple[float, ...]
    action: tuple[float, ...]
    detections: tuple[Detection, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManifestFrame:
        return cls(
            episode_index=int(data["episode_index"]),
            frame_index=int(data["frame_index"]),
            state=tuple(float(value) for value in data["state"]),
            action=tuple(float(value) for value in data["action"]),
            detections=tuple(Detection.from_dict(item) for item in data["detections"]),
        )

