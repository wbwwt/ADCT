"""Export a LeRobot dataset to ADCT's detector-primitive JSONL format.

This example targets the LeRobot API used by the original experiment
workspace (commit 5e947380). Install that LeRobot checkout and this ADCT
package in the same environment before running.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from adct.config import ExperimentConfig, resolve_project_path
from adct.detector import RTDETRDetector


def scalar_int(value) -> int:
    return int(value.item()) if hasattr(value, "item") else int(value)


def as_chw_float(image) -> torch.Tensor:
    tensor = torch.as_tensor(image)
    if tensor.ndim != 3:
        raise ValueError(f"Expected a three-dimensional image, got {tuple(tensor.shape)}.")
    if tensor.shape[0] != 3 and tensor.shape[-1] == 3:
        tensor = tensor.permute(2, 0, 1)
    if tensor.shape[0] != 3:
        raise ValueError(f"Could not infer image channels for shape {tuple(tensor.shape)}.")
    tensor = tensor.to(torch.float32)
    if tensor.max() > 1:
        tensor = tensor / 255.0
    return tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--image-key", default="observation.images.depth")
    parser.add_argument("--state-key", default="observation.state")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--config", default="configs/adct_so100.yaml")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise SystemExit(
            "LeRobot is not installed. Install the experiment-compatible LeRobot checkout first."
        ) from error

    cfg = ExperimentConfig.from_yaml(args.config)
    if cfg.detector is None:
        raise ValueError("The experiment configuration has no detector section.")
    detector_cfg = cfg.detector
    detector = RTDETRDetector(
        resolve_project_path(detector_cfg.config, args.project_root),
        resolve_project_path(detector_cfg.checkpoint, args.project_root),
        device=detector_cfg.device,
        input_size=detector_cfg.input_size,
        score_threshold=detector_cfg.score_threshold,
        top_k=detector_cfg.top_k,
        use_amp=detector_cfg.use_amp,
    )

    dataset_kwargs = {}
    if args.dataset_root is not None:
        dataset_kwargs["root"] = Path(args.dataset_root)
    dataset = LeRobotDataset(args.repo_id, **dataset_kwargs)
    limit = len(dataset) if args.max_frames is None else min(len(dataset), args.max_frames)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for index in tqdm(range(limit), desc="detecting"):
            frame = dataset[index]
            image = as_chw_float(frame[args.image_key])
            detections = detector.predict(image)[0]
            row = {
                "episode_index": scalar_int(frame["episode_index"]),
                "frame_index": scalar_int(frame["frame_index"]),
                "state": torch.as_tensor(frame[args.state_key]).tolist(),
                "action": torch.as_tensor(frame[args.action_key]).tolist(),
                "detections": [detection.to_dict() for detection in detections],
            }
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"Wrote {limit} frames to {output}")


if __name__ == "__main__":
    main()

