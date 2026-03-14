from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from fastslow.data import TaskSpec, make_batch
from fastslow.models import build_model, count_parameters, default_model_config
from fastslow.tracking import build_tracker


PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "train_pairs": 32,
        "eval_pairs": [32, 64],
        "steps": 120,
        "batch_size": 24,
        "eval_batches": 8,
        "eval_interval": 20,
        "lr": 3e-4,
        "weight_decay": 0.05,
        "warmup_steps": 20,
        "grad_clip": 1.0,
        "log_interval": 10,
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 8,
        "d_ff": 512,
        "d_slow": 48,
        "slow_update_gap": 4,
        "dropout": 0.1,
    },
    "default": {
        "train_pairs": 96,
        "eval_pairs": [96, 128, 160],
        "steps": 2400,
        "batch_size": 32,
        "eval_batches": 32,
        "eval_interval": 200,
        "lr": 2.5e-4,
        "weight_decay": 0.05,
        "warmup_steps": 150,
        "grad_clip": 1.0,
        "log_interval": 20,
        "d_model": 192,
        "n_heads": 6,
        "n_layers": 12,
        "d_ff": 768,
        "d_slow": 64,
        "slow_update_gap": 4,
        "dropout": 0.1,
    },
}


@dataclass
class TrainingConfig:
    variant: str
    preset: str
    run_name: str
    seed: int
    train_pairs: int
    eval_pairs: list[int]
    steps: int
    batch_size: int
    eval_batches: int
    eval_interval: int
    lr: float
    weight_decay: float
    warmup_steps: int
    grad_clip: float
    log_interval: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    d_slow: int
    slow_update_gap: int
    dropout: float
    artifacts_dir: Path
    device: str


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train FastSlow experiment variants.")
    parser.add_argument("--variant", choices=["baseline", "fastslow", "fastslow_every_layer", "widened_baseline"], required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="default")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--artifacts-dir", default=None)
    parser.add_argument("--device", default="auto")

    for key, value in PRESETS["default"].items():
        arg = "--" + key.replace("_", "-")
        if isinstance(value, list):
            parser.add_argument(arg, type=int, nargs="+")
        elif isinstance(value, int):
            parser.add_argument(arg, type=int)
        else:
            parser.add_argument(arg, type=float)

    args = parser.parse_args()
    preset = PRESETS[args.preset].copy()
    for key in preset:
        value = getattr(args, key)
        if value is not None:
            preset[key] = value

    artifacts_dir = Path(args.artifacts_dir or f"artifacts/{args.run_name}")
    return TrainingConfig(
        variant=args.variant,
        preset=args.preset,
        run_name=args.run_name,
        seed=args.seed,
        artifacts_dir=artifacts_dir,
        device=args.device,
        **preset,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(raw: str) -> torch.device:
    if raw == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(raw)


def compute_masked_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, float]:
    vocab = logits.shape[-1]
    flat_loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1), reduction="none")
    flat_mask = mask.reshape(-1).float()
    loss = (flat_loss * flat_mask).sum() / flat_mask.sum().clamp_min(1.0)
    predictions = logits.argmax(dim=-1)
    accuracy = predictions[mask].eq(targets[mask]).float().mean().item()
    return loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    spec: TaskSpec,
    device: torch.device,
    eval_pairs: list[int],
    batch_size: int,
    eval_batches: int,
) -> dict[str, float]:
    model.eval()
    metrics: dict[str, float] = {}
    for num_pairs in eval_pairs:
        losses = []
        accuracies = []
        for _ in range(eval_batches):
            inputs, targets, mask = make_batch(batch_size, num_pairs, spec, device)
            logits = model(inputs)
            loss, accuracy = compute_masked_loss(logits, targets, mask)
            losses.append(loss.item())
            accuracies.append(accuracy)
        metrics[f"eval_loss_pairs_{num_pairs}"] = float(np.mean(losses))
        metrics[f"eval_acc_pairs_{num_pairs}"] = float(np.mean(accuracies))
    model.train()
    return metrics


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def append_history(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def learning_rate(step: int, base_lr: float, warmup_steps: int, total_steps: int) -> float:
    if step <= warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.1 * base_lr + 0.9 * base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(config: TrainingConfig) -> None:
    set_seed(config.seed)
    device = choose_device(config.device)
    spec = TaskSpec()
    max_pairs = max([config.train_pairs, *config.eval_pairs])
    max_seq_len = max_pairs * 3 + 2
    base_model_config = default_model_config(spec, max_seq_len)
    base_model_config = replace(
        base_model_config,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        d_slow=config.d_slow,
        slow_update_gap=config.slow_update_gap,
        dropout=config.dropout,
    )
    model, resolved_model_config = build_model(config.variant, base_model_config)
    model.to(device)

    artifacts_dir = config.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    tracker = build_tracker(artifacts_dir)
    tracker.phase("train")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    history_path = artifacts_dir / "history.jsonl"
    checkpoint_path = artifacts_dir / "best.pt"
    summary_path = artifacts_dir / "summary.json"

    resolved = {
        **asdict(config),
        "artifacts_dir": str(config.artifacts_dir),
        "tracker_mode": tracker.mode,
        "device_resolved": str(device),
        "model_params": count_parameters(model),
        "resolved_model_config": resolved_model_config.__dict__,
    }
    write_json(artifacts_dir / "config.json", resolved)

    best_metric = -1.0
    last_eval: dict[str, float] = {}

    progress = tqdm(range(1, config.steps + 1), desc=f"{config.variant}:{config.run_name}")
    try:
        for step in progress:
            lr = learning_rate(step, config.lr, config.warmup_steps, config.steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            inputs, targets, mask = make_batch(config.batch_size, config.train_pairs, spec, device)
            logits = model(inputs)
            loss, accuracy = compute_masked_loss(logits, targets, mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            train_metrics = {
                "train_loss": float(loss.item()),
                "train_acc": float(accuracy),
                "lr": float(lr),
                "grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
            }
            if step % max(1, config.log_interval) == 0 or step == 1:
                tracker.log(step, train_metrics)
                append_history(history_path, {"step": step, "phase": "train", **train_metrics})

            if step % max(1, config.eval_interval) == 0 or step == config.steps:
                tracker.phase("eval")
                eval_metrics = evaluate(
                    model=model,
                    spec=spec,
                    device=device,
                    eval_pairs=config.eval_pairs,
                    batch_size=config.batch_size,
                    eval_batches=config.eval_batches,
                )
                tracker.log(step, eval_metrics)
                append_history(history_path, {"step": step, "phase": "eval", **eval_metrics})
                tracker.phase("train")
                last_eval = eval_metrics

                longest_key = f"eval_acc_pairs_{max(config.eval_pairs)}"
                score = eval_metrics[longest_key]
                if score >= best_metric:
                    best_metric = score
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "config": resolved,
                            "best_metric": best_metric,
                        },
                        checkpoint_path,
                    )

                progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{accuracy:.3f}", best=f"{best_metric:.3f}")

        summary = {
            "variant": config.variant,
            "run_name": config.run_name,
            "seed": config.seed,
            "tracker_mode": tracker.mode,
            "device": str(device),
            "model_params": count_parameters(model),
            "resolved_model_config": resolved_model_config.__dict__,
            "train_pairs": config.train_pairs,
            "eval_pairs": config.eval_pairs,
            "best_long_context_accuracy": best_metric,
            "final_eval": last_eval,
        }
        write_json(summary_path, summary)
        tracker.complete(summary=f"best_long_context_accuracy={best_metric:.4f}")
    except Exception as exc:
        tracker.fail(summary=str(exc))
        raise


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
