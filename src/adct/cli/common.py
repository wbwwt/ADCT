from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from adct.types import ManifestFrame


def read_manifest(path: str | Path) -> dict[int, list[ManifestFrame]]:
    episodes: dict[int, list[ManifestFrame]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                frame = ManifestFrame.from_dict(json.loads(line))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid manifest line {line_number}: {error}") from error
            episodes[frame.episode_index].append(frame)
    if not episodes:
        raise ValueError(f"No frames found in {path}.")
    for frames in episodes.values():
        frames.sort(key=lambda item: item.frame_index)
        indices = [frame.frame_index for frame in frames]
        if len(indices) != len(set(indices)):
            raise ValueError("Duplicate frame indices found inside an episode.")
    return dict(sorted(episodes.items()))

