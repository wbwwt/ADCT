"""Confidence-aware action-chunk execution from Equation (3) of the paper."""

from __future__ import annotations

import math
from collections import deque

from torch import Tensor

from adct.config import DynamicHorizonConfig


def execution_horizon(confidence: float, config: DynamicHorizonConfig) -> int:
    """Return how many actions to execute before the next perception update."""

    confidence = min(1.0, max(0.0, float(confidence)))
    if confidence >= config.confidence_threshold:
        return config.chunk_size
    horizon = math.floor(config.chunk_size - config.scaling_factor * (1.0 - confidence))
    return min(config.chunk_size, max(config.min_horizon, horizon))


class ActionChunkBuffer:
    """Queue the confidence-selected prefix of one predicted action chunk."""

    def __init__(self, config: DynamicHorizonConfig) -> None:
        self.config = config
        self._queue: deque[Tensor] = deque()
        self.last_horizon = 0
        self.last_confidence: float | None = None

    def __len__(self) -> int:
        return len(self._queue)

    def clear(self) -> None:
        self._queue.clear()
        self.last_horizon = 0
        self.last_confidence = None

    def load(self, actions: Tensor, confidence: float) -> int:
        if actions.ndim == 3:
            if actions.shape[0] != 1:
                raise ValueError("Runtime buffering currently supports batch size 1.")
            actions = actions[0]
        if actions.ndim != 2:
            raise ValueError("Expected actions with shape (chunk, action_dim).")
        if actions.shape[0] < self.config.chunk_size:
            raise ValueError(
                f"Expected at least {self.config.chunk_size} actions, got {actions.shape[0]}."
            )
        self.clear()
        self.last_confidence = float(confidence)
        self.last_horizon = execution_horizon(confidence, self.config)
        self._queue.extend(action for action in actions[: self.last_horizon])
        return self.last_horizon

    def pop(self) -> Tensor:
        if not self._queue:
            raise IndexError("The action chunk buffer is empty.")
        return self._queue.popleft()

