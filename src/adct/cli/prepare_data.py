from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from adct.cli.common import read_manifest
from adct.config import ExperimentConfig
from adct.experience_tree import ExperienceState, ExperienceTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a JSONL demonstration manifest into an ADCT .npz dataset."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--tree-output",
        default=None,
        help="Experience-tree JSON path (default: next to output dataset).",
    )
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    return parser.parse_args()


def _normalized_box(
    box: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> np.ndarray:
    values = np.asarray(box, dtype=np.float32).copy()
    values[[0, 2]] = values[[0, 2]] / image_width * 2.0 - 1.0
    values[[1, 3]] = values[[1, 3]] / image_height * 2.0 - 1.0
    return values


def main() -> None:
    args = parse_args()
    if args.image_width < 1 or args.image_height < 1:
        raise ValueError("Image dimensions must be positive.")
    config = ExperimentConfig.from_yaml(args.config)
    episodes = read_manifest(args.manifest)
    model_config = config.model

    detection_episodes = [
        [frame.detections for frame in frames] for frames in episodes.values()
    ]
    tree = ExperienceTree(config.experience_tree).fit(detection_episodes)

    states: list[np.ndarray] = []
    action_chunks: list[np.ndarray] = []
    action_padding: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    boxes: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    detection_masks: list[np.ndarray] = []
    episode_indices: list[int] = []
    frame_indices: list[int] = []

    for episode_index, frames in episodes.items():
        if not frames:
            continue
        episode_actions = np.asarray([frame.action for frame in frames], dtype=np.float32)
        if episode_actions.ndim != 2 or episode_actions.shape[1] != model_config.action_dim:
            raise ValueError(
                f"Episode {episode_index} actions must have dimension "
                f"{model_config.action_dim}; got {episode_actions.shape}."
            )
        state = ExperienceState()
        for offset, frame in enumerate(frames):
            frame_state = np.asarray(frame.state, dtype=np.float32)
            if frame_state.shape != (model_config.state_dim,):
                raise ValueError(
                    f"Episode {episode_index}, frame {frame.frame_index}: state must have "
                    f"shape ({model_config.state_dim},), got {frame_state.shape}."
                )

            tree.update_completed(frame.detections, state)
            selected = tree.select_target(frame.detections, state)
            if selected is None:
                continue

            end = min(len(frames), offset + model_config.chunk_size)
            valid_count = end - offset
            chunk = np.zeros(
                (model_config.chunk_size, model_config.action_dim),
                dtype=np.float32,
            )
            chunk[:valid_count] = episode_actions[offset:end]
            is_pad = np.ones(model_config.chunk_size, dtype=np.bool_)
            is_pad[:valid_count] = False

            sample_labels = np.zeros(model_config.max_detections, dtype=np.int64)
            sample_boxes = np.zeros((model_config.max_detections, 4), dtype=np.float32)
            sample_scores = np.zeros(model_config.max_detections, dtype=np.float32)
            sample_mask = np.zeros(model_config.max_detections, dtype=np.bool_)
            sample_labels[0] = selected.label
            sample_boxes[0] = _normalized_box(
                selected.box,
                args.image_width,
                args.image_height,
            )
            sample_scores[0] = selected.score
            sample_mask[0] = True

            states.append(frame_state)
            action_chunks.append(chunk)
            action_padding.append(is_pad)
            labels.append(sample_labels)
            boxes.append(sample_boxes)
            scores.append(sample_scores)
            detection_masks.append(sample_mask)
            episode_indices.append(episode_index)
            frame_indices.append(frame.frame_index)

    if not states:
        raise ValueError(
            "No trainable frames remained after experience-tree filtering. "
            "Check the target region, labels, and detector confidence threshold."
        )

    states_array = np.stack(states)
    actions_array = np.stack(action_chunks)
    action_padding_array = np.stack(action_padding)
    valid_actions = actions_array[~action_padding_array]
    state_mean = states_array.mean(axis=0, dtype=np.float64).astype(np.float32)
    state_std = states_array.std(axis=0, dtype=np.float64).astype(np.float32)
    action_mean = valid_actions.mean(axis=0, dtype=np.float64).astype(np.float32)
    action_std = valid_actions.std(axis=0, dtype=np.float64).astype(np.float32)
    state_std = np.maximum(state_std, 1e-6)
    action_std = np.maximum(action_std, 1e-6)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        states=states_array,
        actions=actions_array,
        action_is_pad=action_padding_array,
        labels=np.stack(labels),
        boxes=np.stack(boxes),
        scores=np.stack(scores),
        detection_mask=np.stack(detection_masks),
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        episode_index=np.asarray(episode_indices, dtype=np.int64),
        frame_index=np.asarray(frame_indices, dtype=np.int64),
    )
    tree_output = Path(args.tree_output) if args.tree_output else output.with_name(
        "experience_tree.json"
    )
    tree.save(tree_output)
    print(f"Prepared {len(states)} samples in {output}")
    print(f"Saved experience tree to {tree_output}")


if __name__ == "__main__":
    main()

