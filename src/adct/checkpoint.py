"""Safe checkpoint serialization for ADCT."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from safetensors.torch import load_file, save_file

from adct.config import ADCTModelConfig
from adct.model import ADCT
from adct.normalization import NormalizationStats


def save_checkpoint(
    model: ADCT,
    output_dir: str | Path,
    *,
    normalization_stats: NormalizationStats | None = None,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    save_file(model.state_dict(), destination / "model.safetensors")
    (destination / "model_config.json").write_text(
        json.dumps(asdict(model.config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if normalization_stats is not None:
        normalization_stats.save(destination / "normalization.json")


def load_checkpoint(
    checkpoint_dir: str | Path,
    *,
    device: str = "cpu",
) -> ADCT:
    source = Path(checkpoint_dir)
    config = ADCTModelConfig(
        **json.loads((source / "model_config.json").read_text(encoding="utf-8"))
    )
    model = ADCT(config)
    model.load_state_dict(load_file(source / "model.safetensors", device=device))
    normalization_path = source / "normalization.json"
    if normalization_path.exists():
        model.normalization_stats = NormalizationStats.load(normalization_path)
    return model.to(device)

