"""Closed-loop ADCT runtime with experience-tree gating and dynamic stepping."""

from __future__ import annotations

import torch
from torch import Tensor

from adct.config import DynamicHorizonConfig
from adct.detector import Detector
from adct.dynamic_horizon import ActionChunkBuffer
from adct.experience_tree import ExperienceState, ExperienceTree
from adct.features import normalize_boxes_xyxy
from adct.model import ADCT
from adct.types import Detection


class NoTargetError(RuntimeError):
    """Raised when the experience tree cannot find a valid visible target."""


class ADCTRuntime:
    """Execute one action at a time while re-planning at confidence-aware intervals."""

    def __init__(
        self,
        model: ADCT,
        detector: Detector,
        experience_tree: ExperienceTree,
        horizon_config: DynamicHorizonConfig,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        if model.config.chunk_size != horizon_config.chunk_size:
            raise ValueError("Model and dynamic-horizon chunk sizes must match.")
        if model.config.max_detections != 1:
            raise ValueError("The reference runtime expects one experience-tree target.")
        self.model = model
        self.detector = detector
        self.experience_tree = experience_tree
        self.device = torch.device(device or next(model.parameters()).device)
        self.model.to(self.device).eval()
        self.state = ExperienceState()
        self.actions = ActionChunkBuffer(horizon_config)
        self.last_selected: Detection | None = None
        self.last_detections: list[Detection] = []

    def reset(self) -> None:
        self.state.reset()
        self.actions.clear()
        self.last_selected = None
        self.last_detections = []

    @torch.inference_mode()
    def _plan(self, image: Tensor, robot_state: Tensor) -> None:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("image must have shape (3, H, W).")
        if robot_state.numel() != self.model.config.state_dim:
            raise ValueError(
                f"Expected {self.model.config.state_dim} robot-state values, "
                f"got {robot_state.numel()}."
            )

        detections_batch = self.detector.predict(image.unsqueeze(0))
        if len(detections_batch) != 1:
            raise RuntimeError("Detector returned an unexpected batch size.")
        self.last_detections = list(detections_batch[0])
        self.experience_tree.update_completed(self.last_detections, self.state)
        selected = self.experience_tree.select_target(self.last_detections, self.state)
        if selected is None:
            raise NoTargetError(
                "No visible, unplaced target matched the learned experience tree."
            )
        self.last_selected = selected

        _, image_height, image_width = image.shape
        labels = torch.tensor([[selected.label]], dtype=torch.long, device=self.device)
        boxes = torch.tensor([[selected.box]], dtype=torch.float32, device=self.device)
        boxes = normalize_boxes_xyxy(
            boxes,
            image_width=image_width,
            image_height=image_height,
        )
        detection_mask = torch.ones((1, 1), dtype=torch.bool, device=self.device)
        state = robot_state.reshape(1, -1).to(device=self.device, dtype=torch.float32)
        if self.model.normalization_stats is not None:
            state = self.model.normalization_stats.normalize_state(state)
        output = self.model(
            state=state,
            labels=labels,
            boxes=boxes,
            detection_mask=detection_mask,
        )
        action_chunk = output.actions
        if self.model.normalization_stats is not None:
            action_chunk = self.model.normalization_stats.unnormalize_action(action_chunk)
        self.actions.load(action_chunk, selected.score)

    @torch.inference_mode()
    def step(self, image: Tensor, robot_state: Tensor) -> Tensor:
        """Return the next action, re-running perception when the queue is empty."""

        if len(self.actions) == 0:
            self._plan(image, robot_state)
        return self.actions.pop()

