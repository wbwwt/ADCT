# Robot deployment guide

This guide covers the control-loop boundary. It does not replace the safety
documentation for the robot arm.

## Camera and geometry

1. Mount the camera in a stable top-view pose.
2. Record the image resolution used by the detector.
3. Measure the pixel center of the placement container.
4. Set `target_region.center_x` and `center_y`.
5. Start with the paper tolerance of 25 pixels and verify that a correctly
   placed object's center enters the region.
6. Confirm that detector boxes use `xyxy`, not `xywh`.

The runtime normalizes boxes using the actual image width and height. It does
not use the historical hard-coded `640 x 480` calculation.

## Episode lifecycle

Call `runtime.reset()`:

- before the first episode;
- after a success or failure;
- after moving objects manually;
- after changing the scene.

Failing to reset carries completed labels and queued actions into the next
episode.

## Control loop

```python
runtime.reset()
while not episode_done:
    image = camera.read_tensor()       # float32, (3, H, W)
    state = robot.read_state_tensor()  # float32, (state_dim,)
    try:
        action = runtime.step(image, state)
    except NoTargetError:
        robot.stop()
        break
    robot.send_action(action)
```

The detector and policy are re-run only when the current confidence-selected
action prefix is exhausted. At confidence above `0.93`, all 50 actions are
executed. Lower confidence produces a shorter prefix and therefore an earlier
perception update.

## Safety checklist

- Test with motors disabled or in simulation first.
- Enforce joint-position, velocity, current, and workspace limits outside
  ADCT.
- Add an independent emergency stop.
- Stop on missing detections, stale camera frames, NaNs, or communication
  timeouts.
- Verify class-label order against the experience tree.
- Begin with a low robot speed and an empty workspace.
- Do not treat detector confidence as a calibrated collision-risk estimate.

## Timing

Warm up both networks before measuring. For CUDA:

```python
torch.cuda.synchronize()
start = time.perf_counter()
action = runtime.step(image, state)
torch.cuda.synchronize()
elapsed_ms = (time.perf_counter() - start) * 1000
```

Report detector, policy, and end-to-end loop latency separately.

