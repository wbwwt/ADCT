from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from adct.checkpoint import save_checkpoint
from adct.config import ExperimentConfig
from adct.dataset import PreparedADCTDataset
from adct.model import ADCT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the image-free ADCT policy.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    experiment = ExperimentConfig.from_yaml(args.config)
    training = experiment.training
    epochs = args.epochs if args.epochs is not None else training.epochs
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    device = torch.device(args.device or training.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    _set_seed(training.seed)

    dataset = PreparedADCTDataset(args.dataset, experiment.model)
    loader = DataLoader(
        dataset,
        batch_size=training.batch_size,
        shuffle=True,
        num_workers=training.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=training.num_workers > 0,
    )
    model = ADCT(experiment.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    counts = model.parameter_counts()
    print(f"Parameters: {counts['trainable']:,} trainable / {counts['total']:,} total")

    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        model.train()
        totals = {"loss": 0.0, "reconstruction_loss": 0.0, "kl_loss": 0.0}
        sample_count = 0
        progress = tqdm(loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for batch in progress:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            output = model(
                state=batch["state"],
                labels=batch["labels"],
                boxes=batch["boxes"],
                detection_mask=batch["detection_mask"],
                actions=batch["actions"],
                action_is_pad=batch["action_is_pad"],
            )
            loss, metrics = model.compute_loss(
                output,
                batch["actions"],
                batch["action_is_pad"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            batch_size = batch["state"].shape[0]
            sample_count += batch_size
            for name in totals:
                totals[name] += float(metrics[name]) * batch_size
            progress.set_postfix(loss=f"{float(loss):.4f}")

        row: dict[str, float | int] = {
            "epoch": epoch,
            "seconds": time.perf_counter() - started,
            **{name: value / sample_count for name, value in totals.items()},
        }
        history.append(row)
        print(
            f"epoch={epoch:04d} loss={row['loss']:.6f} "
            f"recon={row['reconstruction_loss']:.6f} kl={row['kl_loss']:.6f}"
        )

    model.eval()
    save_checkpoint(model, output_dir, normalization_stats=dataset.stats)
    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "experiment_config.json").write_text(
        json.dumps(experiment.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved checkpoint to {output_dir}")


if __name__ == "__main__":
    main()

