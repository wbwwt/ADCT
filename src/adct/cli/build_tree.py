from __future__ import annotations

import argparse

from adct.cli.common import read_manifest
from adct.config import ExperimentConfig
from adct.experience_tree import ExperienceTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an ADCT experience tree from a detection manifest."
    )
    parser.add_argument("--manifest", required=True, help="Input JSONL demonstration manifest.")
    parser.add_argument("--config", required=True, help="Experiment YAML configuration.")
    parser.add_argument("--output", required=True, help="Output experience_tree.json.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    episodes = read_manifest(args.manifest)
    tree = ExperienceTree(config.experience_tree).fit(
        [[frame.detections for frame in frames] for frames in episodes.values()]
    )
    tree.save(args.output)
    print(f"Saved {len(tree.sequence_counts)} unique experience path(s) to {args.output}")


if __name__ == "__main__":
    main()

