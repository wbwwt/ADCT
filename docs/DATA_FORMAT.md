# ADCT data format

## Demonstration manifest

The portable input format is newline-delimited JSON (`.jsonl`). Each non-empty
line describes one synchronized robot frame:

```json
{
  "episode_index": 3,
  "frame_index": 27,
  "state": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
  "action": [0.01, 0.11, 0.21, 0.31, 0.41, 0.51],
  "detections": [
    {
      "label": 2,
      "box": [125.0, 72.0, 181.0, 149.0],
      "score": 0.96
    }
  ]
}
```

Fields:

- `episode_index`: non-negative integer episode identifier.
- `frame_index`: unique integer within an episode.
- `state`: current robot state, matching `model.state_dim`.
- `action`: demonstrated action, matching `model.action_dim`.
- `detections`: zero or more detector primitives.
- `label`: zero-based class index matching the detector label map.
- `box`: pixel-space `xyxy` coordinates.
- `score`: confidence in `[0, 1]`.

Frames may appear in any order in the file; the loader groups by episode and
sorts by `frame_index`.

## Experience-tree construction

For each episode, the builder records a class when its highest-confidence
detection first enters the configured target region:

```text
abs(center_x - target_x) < tolerance_x
and
abs(center_y - target_y) < tolerance_y
```

The resulting paths and visit counts are stored in a versioned JSON file.
This makes target-selection behavior inspectable and editable without
retraining the policy.

## Prepared training archive

`adct-prepare-data` writes a compressed NumPy archive with:

| Array | Shape | Type |
|---|---|---|
| `states` | `(N, state_dim)` | `float32` |
| `actions` | `(N, chunk_size, action_dim)` | `float32` |
| `action_is_pad` | `(N, chunk_size)` | `bool` |
| `labels` | `(N, max_detections)` | `int64` |
| `boxes` | `(N, max_detections, 4)` | `float32` |
| `scores` | `(N, max_detections)` | `float32` |
| `detection_mask` | `(N, max_detections)` | `bool` |
| `state_mean`, `state_std` | `(state_dim,)` | `float32` |
| `action_mean`, `action_std` | `(action_dim,)` | `float32` |
| `episode_index` | `(N,)` | `int64` |
| `frame_index` | `(N,)` | `int64` |

Prepared boxes are normalized to `[-1, 1]`. Confidence is stored for auditing
and runtime stepping but is not an input to the action transformer.

The converter computes state statistics over all retained frames and action
statistics over non-padding actions. The dataset normalizes both during
training. Checkpoints store these values in `normalization.json`; the runtime
normalizes robot states and converts predicted actions back to robot units.

## Train/evaluation separation

Do not fit the experience tree on evaluation roll-outs. Construct it only from
training demonstrations, then reuse the frozen JSON tree during evaluation.
If scene-specific trees are compared, record the tree file hash with every
result.

