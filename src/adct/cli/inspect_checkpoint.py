from __future__ import annotations

import argparse
import json
from pathlib import Path

from adct.checkpoint import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an ADCT safetensors checkpoint.")
    parser.add_argument("checkpoint_dir")
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint_dir)
    print(json.dumps(model.parameter_counts(), indent=2))
    print((Path(args.checkpoint_dir) / "model_config.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

