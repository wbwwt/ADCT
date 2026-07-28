# Mapping from the experiment workspace to paper variants

The original experiment directory accumulated one Python file per ablation.
The final paper method was implemented by `DETR_ACT07` with
`use_solution=Second`. The public package replaces these duplicated files with
composable modules and configuration switches.

| Historical class | Paper interpretation |
|---|---|
| `DETR_ACT` | RGB + detection at VAE and policy transformer (`VT`) |
| `DETR_ACT02` | RGB + detection at policy transformer (`T`) |
| `DETR_ACT03` | RGB + detection at VAE (`V`) |
| `DETR_ACT04` | Detection-only input |
| `DETR_ACT05` | Detection-only with repeated labels (`RL`) |
| `DETR_ACT06` | Detection-only with separate label/box encoding (`SE`) |
| `DETR_ACTS01` | ET + RGB + detection (`VT`) |
| `DETR_ACTS02` | ET + RGB + detection (`T`) |
| `DETR_ACTS03` | ET + RGB + detection (`V`) |
| `DETR_ACTS04` | ET + detection-only |
| `DETR_ACTS05` | ET + detection-only repeated labels |
| `DETR_ACT07`, solution `Second` | **ADCT** |
| `et_detr_dp.py` | ET + Diffusion Policy |

## Cleanup decisions

- Detector construction was removed from the transformer forward pass and
  placed in `RTDETRDetector`.
- Personal paths and unconditional `.to("cuda")` calls were replaced by typed
  configuration.
- The detector is explicitly frozen and excluded from action-policy training.
- The experience tree is a serializable prefix tree rather than an implicit
  label-sort branch.
- Dynamic stepping is implemented in `ActionChunkBuffer`; it was not connected
  to the historical action queue.
- Debug visualization, sleep calls, commented experiments, duplicate
  transformer classes, and generated test binaries are not part of the
  algorithm package.
- The original `lerobot_my` workspace remains untouched for experiment
  provenance.

