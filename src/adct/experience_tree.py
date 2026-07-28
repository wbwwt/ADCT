"""Semantic experience-tree construction and target selection.

The tree is learned from the ordered first entries of object categories into
the target region. During inference it acts as a semantic gate: only the
highest-priority unplaced object is forwarded to the action policy.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adct.config import ExperienceTreeConfig, TargetRegionConfig
from adct.types import Detection


@dataclass
class _Node:
    label: int | None
    visits: int = 0
    children: dict[int, _Node] = field(default_factory=dict)

    def add_path(self, labels: Sequence[int]) -> None:
        self.visits += 1
        node = self
        for label in labels:
            node = node.children.setdefault(label, _Node(label=label))
            node.visits += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "visits": self.visits,
            "children": {
                str(label): child.to_dict()
                for label, child in sorted(self.children.items())
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _Node:
        node = cls(label=data["label"], visits=int(data["visits"]))
        node.children = {
            int(label): cls.from_dict(child)
            for label, child in data.get("children", {}).items()
        }
        return node


@dataclass
class ExperienceState:
    """Episode-local progress through a learned experience tree."""

    path: list[int] = field(default_factory=list)
    completed_labels: set[int] = field(default_factory=set)
    active_label: int | None = None

    def reset(self) -> None:
        self.path.clear()
        self.completed_labels.clear()
        self.active_label = None


class ExperienceTree:
    def __init__(self, config: ExperienceTreeConfig) -> None:
        self.config = config
        self.root = _Node(label=None)
        self.sequence_counts: Counter[tuple[int, ...]] = Counter()

    @property
    def target_region(self) -> TargetRegionConfig:
        return self.config.target_region

    def _inside_target(self, detection: Detection) -> bool:
        return self.target_region.contains(*detection.center)

    def extract_sequence(self, frames: Sequence[Sequence[Detection]]) -> list[int]:
        """Extract first-entry labels from one demonstration trajectory."""

        entered: set[int] = set()
        sequence: list[int] = []
        for detections in frames:
            best_by_label: dict[int, Detection] = {}
            for detection in detections:
                if detection.score < self.config.min_confidence:
                    continue
                previous = best_by_label.get(detection.label)
                if previous is None or detection.score > previous.score:
                    best_by_label[detection.label] = detection

            newly_entered = [
                detection
                for detection in best_by_label.values()
                if detection.label not in entered and self._inside_target(detection)
            ]
            # Multiple simultaneous entries are resolved deterministically by
            # confidence and then label. In normal demonstrations this is rare.
            newly_entered.sort(key=lambda item: (-item.score, item.label))
            for detection in newly_entered:
                entered.add(detection.label)
                sequence.append(detection.label)
        return sequence

    def fit(self, episodes: Iterable[Sequence[Sequence[Detection]]]) -> ExperienceTree:
        count = 0
        for frames in episodes:
            sequence = self.extract_sequence(frames)
            if not sequence:
                continue
            self.root.add_path(sequence)
            self.sequence_counts[tuple(sequence)] += 1
            count += 1
        if count == 0:
            raise ValueError(
                "No object entered the configured target region in any demonstration."
            )
        return self

    def _node_for_path(self, path: Sequence[int]) -> _Node:
        node = self.root
        for label in path:
            child = node.children.get(label)
            if child is None:
                return self.root
            node = child
        return node

    def _global_priority(self) -> dict[int, int]:
        priorities: Counter[int] = Counter()

        def visit(node: _Node) -> None:
            for label, child in node.children.items():
                priorities[label] += child.visits
                visit(child)

        visit(self.root)
        return dict(priorities)

    def update_completed(
        self,
        detections: Sequence[Detection],
        state: ExperienceState,
    ) -> list[int]:
        """Advance state for tree labels currently observed inside the target."""

        changed: list[int] = []
        inside = {
            detection.label
            for detection in detections
            if detection.score >= self.config.min_confidence
            and self._inside_target(detection)
        }
        current = self._node_for_path(state.path)
        expected = set(current.children)

        # Prefer the expected next branch. Fall back to any tree label already
        # in the target, which correctly initializes partially completed scenes.
        ordered = sorted(
            inside - state.completed_labels,
            key=lambda label: (
                label not in expected,
                -current.children.get(label, _Node(label)).visits,
                label,
            ),
        )
        tree_labels = set(self._global_priority())
        for label in ordered:
            if label not in tree_labels:
                continue
            state.completed_labels.add(label)
            state.path.append(label)
            if state.active_label == label:
                state.active_label = None
            changed.append(label)
        return changed

    def select_target(
        self,
        detections: Sequence[Detection],
        state: ExperienceState,
    ) -> Detection | None:
        """Select the highest-priority visible and unplaced detection."""

        best_by_label: dict[int, Detection] = {}
        for detection in detections:
            if (
                detection.score < self.config.min_confidence
                or detection.label in state.completed_labels
                or self._inside_target(detection)
            ):
                continue
            previous = best_by_label.get(detection.label)
            if previous is None or detection.score > previous.score:
                best_by_label[detection.label] = detection

        if not best_by_label:
            state.active_label = None
            return None

        node = self._node_for_path(state.path)
        global_priority = self._global_priority()

        def rank(detection: Detection) -> tuple[int, int, float, int]:
            child = node.children.get(detection.label)
            return (
                1 if child is not None else 0,
                child.visits if child is not None else global_priority.get(detection.label, 0),
                detection.score,
                -detection.label,
            )

        selected = max(best_by_label.values(), key=rank)
        state.active_label = selected.label
        return selected

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "config": {
                "min_confidence": self.config.min_confidence,
                "target_region": {
                    "center_x": self.target_region.center_x,
                    "center_y": self.target_region.center_y,
                    "tolerance_x": self.target_region.tolerance_x,
                    "tolerance_y": self.target_region.tolerance_y,
                },
            },
            "root": self.root.to_dict(),
            "sequence_counts": {
                ",".join(str(label) for label in sequence): count
                for sequence, count in sorted(self.sequence_counts.items())
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperienceTree:
        if data.get("format_version") != 1:
            raise ValueError(f"Unsupported experience-tree format: {data.get('format_version')}")
        config_data = data["config"]
        config = ExperienceTreeConfig(
            target_region=TargetRegionConfig(**config_data["target_region"]),
            min_confidence=float(config_data["min_confidence"]),
        )
        tree = cls(config)
        tree.root = _Node.from_dict(data["root"])
        tree.sequence_counts = Counter(
            {
                tuple(int(label) for label in sequence.split(",") if label): int(count)
                for sequence, count in data.get("sequence_counts", {}).items()
            }
        )
        return tree

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> ExperienceTree:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

