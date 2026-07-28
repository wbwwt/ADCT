# Reproducing the paper experiments

This document separates parameters reported in the paper from values that
must be calibrated for a new robot setup.

## Hardware

| Component | Paper setup |
|---|---|
| Robot | Low-cost SO100, 6-DoF |
| Camera | One top-view Intel RealSense D415 |
| Training GPU | NVIDIA RTX 3090, 24 GB |
| Main inference GPU | NVIDIA RTX 4060, 8 GB |
| Additional inference GPU | NVIDIA RTX 3060 |

The paper reports 31 ms inference latency on the RTX 4060 and 45 ms on the RTX
3060. Measure latency after warm-up and synchronize CUDA around timed regions.

## Optimization and policy settings

| Parameter | Value |
|---|---:|
| Batch size | 16 |
| Learning rate | `1e-5` |
| Training duration | Approximately 200 epochs, until convergence |
| Action chunk size, \(K\) | 50 |
| Confidence threshold, \(c_\mathrm{th}\) | 0.93 |
| Dynamic-step scale, \(r\) | 50 |
| Spatial tolerance, \(\Delta_x,\Delta_y\) | 25 pixels |
| Main roll-outs per result | 100 |
| Roll-outs per ablation | 30 |

These defaults are recorded in `configs/adct_so100.yaml`. The target-region
center is camera/fixture specific and is therefore intentionally not fixed by
the paper configuration.

## Datasets

- **Dataset A (high visual similarity):** 60 demonstrations over two scenes.
  The scenes share the same background and differ mainly in object layout;
  reported feature KL divergence is approximately 0.5.
- **Dataset B (low visual similarity):** 90 demonstrations over three visually
  distinct scenes; reported feature KL divergence is approximately 1.5.
- Each dataset is evaluated both as independent one-scene subsets (`OScreen`)
  and a mixed multi-scene subset (`MScreen`).

The low-shot experiment retrains on reduced subsets of Dataset B scenes c and
d. The reported result remains above 90% with eight demonstrations.

## Pipeline

1. Calibrate the top-view camera and target-region center.
2. Fine-tune RT-DETR-R18 for the task categories.
3. Export detector primitives for every demonstration frame.
4. Build the experience tree from first target-region entries.
5. Prepare semantic target features and 50-step action chunks.
6. Train ADCT for approximately 200 epochs.
7. Evaluate with a fresh runtime state for every roll-out.
8. Report grasping success over 100 roll-outs; use 30 only for ablations.

Commands:

```bash
adct-build-tree \
  --manifest data/demonstrations.jsonl \
  --config configs/adct_so100.yaml \
  --output data/experience_tree.json

adct-prepare-data \
  --manifest data/demonstrations.jsonl \
  --config configs/adct_so100.yaml \
  --output data/adct_train.npz \
  --image-width 640 \
  --image-height 480

adct-train \
  --config configs/adct_so100.yaml \
  --dataset data/adct_train.npz \
  --output-dir outputs/adct
```

## Ablations

The cleaned modules expose the important paper ablations:

| Paper variant | Public-code setting |
|---|---|
| ADCT | Experience tree + separate label/box encoding + dynamic horizon |
| ADCT NSA | Set `dynamic_horizon.scaling_factor: 0` |
| DetOnly SE | Bypass `ExperienceTree.select_target`; retain separate encoding |
| DetOnly RL | Replace the separate encoder with repeated scalar labels |
| ET+DP | Use `ExperienceTreeDiffusionConditioner` as DP global conditioning |

The historical one-file variants and their exact Table II mapping are listed
in [EXPERIMENT_CODE_MAP.md](EXPERIMENT_CODE_MAP.md).

## Parameter accounting

The cleaned ADCT policy contains approximately 40.15M trainable parameters
with the paper configuration. The frozen six-class RT-DETR-R18 model contains
approximately 21.86M parameters, for approximately 62.01M total. The paper
reports 40.5M trainable and 62.4M total parameters for the original experiment
workspace. The small difference comes from removing unused experimental
projections and image-backbone objects that were constructed but not used in
the final forward pass.

For scientific comparisons, report both trainable parameters and parameters
that actually participate in inference.

## Artifacts required for exact numerical reproduction

Exact success-rate reproduction additionally requires:

- the original detector checkpoint and label map;
- ADCT policy checkpoints or the complete training demonstrations;
- Dataset A/B scene definitions and train/evaluation splits;
- the success/failure annotation protocol;
- robot calibration and target-region center;
- random seeds used for each training run.

The source repository does not invent these values. Add their public URLs to
the artifact table in the README once uploaded.

