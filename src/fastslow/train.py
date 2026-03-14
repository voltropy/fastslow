from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm

from fastslow.data import TaskSpec, make_batch
from fastslow.models import build_model, count_parameters, default_model_config
from fastslow.tracking import NullTracker, build_tracker, validate_tracking_backend


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
    tracking_backend: str
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


@dataclass(frozen=True)
class RuntimeContext:
    device: torch.device
    rank: int
    local_rank: int
    world_size: int
    distributed: bool

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train FastSlow experiment variants.")
    parser.add_argument("--variant", choices=["baseline", "fastslow", "fastslow_every_layer", "widened_baseline"], required=True)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="default")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tracking-backend", choices=["auto", "volta", "jsonl"], default="auto")
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
        tracking_backend=args.tracking_backend,
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


def init_runtime(raw_device: str) -> RuntimeContext:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if raw_device not in {"auto", "cuda"} and not raw_device.startswith("cuda"):
            raise ValueError("Distributed execution requires CUDA; use --device auto or --device cuda.")
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed execution requested but CUDA is unavailable.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device("cuda", local_rank)
    else:
        device = choose_device(raw_device)

    return RuntimeContext(
        device=device,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        distributed=distributed,
    )


def cleanup_runtime(runtime: RuntimeContext) -> None:
    if runtime.distributed and dist.is_initialized():
        dist.destroy_process_group()


def reduce_metrics(metrics: dict[str, float], runtime: RuntimeContext) -> dict[str, float]:
    if not runtime.distributed:
        return metrics

    ordered_keys = sorted(metrics)
    tensor = torch.tensor([metrics[key] for key in ordered_keys], device=runtime.device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= runtime.world_size
    return {key: float(value) for key, value in zip(ordered_keys, tensor.tolist())}


def maybe_barrier(runtime: RuntimeContext) -> None:
    if runtime.distributed:
        dist.barrier()


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
    runtime: RuntimeContext,
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
            inputs, targets, mask = make_batch(batch_size, num_pairs, spec, runtime.device)
            logits = model(inputs)
            loss, accuracy = compute_masked_loss(logits, targets, mask)
            reduced = reduce_metrics(
                {"eval_loss": float(loss.item()), "eval_acc": float(accuracy)},
                runtime,
            )
            losses.append(reduced["eval_loss"])
            accuracies.append(reduced["eval_acc"])
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
    runtime = init_runtime(config.device)
    try:
        validate_tracking_backend(config.tracking_backend)
        set_seed(config.seed + runtime.rank)
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
        model.to(runtime.device)
        if runtime.distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[runtime.local_rank],
                output_device=runtime.local_rank,
            )
        model_for_artifacts = model.module if isinstance(model, DistributedDataParallel) else model

        artifacts_dir = config.artifacts_dir
        if runtime.is_main_process:
            artifacts_dir.mkdir(parents=True, exist_ok=True)
        maybe_barrier(runtime)

        tracker = build_tracker(artifacts_dir, config.tracking_backend) if runtime.is_main_process else NullTracker()
        tracker.phase("train")

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        history_path = artifacts_dir / "history.jsonl"
        checkpoint_path = artifacts_dir / "best.pt"
        summary_path = artifacts_dir / "summary.json"

        resolved = {
            **asdict(config),
            "artifacts_dir": str(config.artifacts_dir),
            "tracker_mode": tracker.mode,
            "device_resolved": str(runtime.device),
            "distributed": runtime.distributed,
            "world_size": runtime.world_size,
            "per_rank_batch_size": config.batch_size,
            "global_batch_size": config.batch_size * runtime.world_size,
            "model_params": count_parameters(model_for_artifacts),
            "resolved_model_config": resolved_model_config.__dict__,
        }
        if runtime.is_main_process:
            write_json(artifacts_dir / "config.json", resolved)

        best_metric = -1.0
        last_eval: dict[str, float] = {}

        iterator = range(1, config.steps + 1)
        progress = tqdm(iterator, desc=f"{config.variant}:{config.run_name}") if runtime.is_main_process else iterator
        for step in progress:
            lr = learning_rate(step, config.lr, config.warmup_steps, config.steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            inputs, targets, mask = make_batch(config.batch_size, config.train_pairs, spec, runtime.device)
            logits = model(inputs)
            loss, accuracy = compute_masked_loss(logits, targets, mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            train_metrics = reduce_metrics(
                {
                    "train_loss": float(loss.item()),
                    "train_acc": float(accuracy),
                    "lr": float(lr),
                    "grad_norm": float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm),
                },
                runtime,
            )
            if runtime.is_main_process and (step % max(1, config.log_interval) == 0 or step == 1):
                tracker.log(step, train_metrics)
                append_history(history_path, {"step": step, "phase": "train", **train_metrics})

            if step % max(1, config.eval_interval) == 0 or step == config.steps:
                maybe_barrier(runtime)
                if runtime.is_main_process:
                    tracker.phase("eval")
                eval_metrics = evaluate(
                    model=model,
                    spec=spec,
                    runtime=runtime,
                    eval_pairs=config.eval_pairs,
                    batch_size=config.batch_size,
                    eval_batches=config.eval_batches,
                )
                if runtime.is_main_process:
                    tracker.log(step, eval_metrics)
                    append_history(history_path, {"step": step, "phase": "eval", **eval_metrics})
                last_eval = eval_metrics

                longest_key = f"eval_acc_pairs_{max(config.eval_pairs)}"
                score = eval_metrics[longest_key]
                if runtime.is_main_process and score >= best_metric:
                    best_metric = score
                    tracker.phase("checkpoint")
                    tracker.log(step, {"checkpoint_best_long_context_accuracy": float(score)})
                    append_history(
                        history_path,
                        {
                            "step": step,
                            "phase": "checkpoint",
                            "checkpoint_best_long_context_accuracy": float(score),
                        },
                    )
                    torch.save(
                        {
                            "model_state": model_for_artifacts.state_dict(),
                            "config": resolved,
                            "best_metric": best_metric,
                        },
                        checkpoint_path,
                    )
                if runtime.is_main_process:
                    tracker.phase("train")

                if runtime.is_main_process:
                    progress.set_postfix(
                        loss=f"{train_metrics['train_loss']:.4f}",
                        acc=f"{train_metrics['train_acc']:.3f}",
                        best=f"{best_metric:.3f}",
                    )
                maybe_barrier(runtime)

        if runtime.is_main_process:
            summary = {
                "variant": config.variant,
                "run_name": config.run_name,
                "seed": config.seed,
                "tracker_mode": tracker.mode,
                "device": str(runtime.device),
                "distributed": runtime.distributed,
                "world_size": runtime.world_size,
                "per_rank_batch_size": config.batch_size,
                "global_batch_size": config.batch_size * runtime.world_size,
                "model_params": count_parameters(model_for_artifacts),
                "resolved_model_config": resolved_model_config.__dict__,
                "train_pairs": config.train_pairs,
                "eval_pairs": config.eval_pairs,
                "best_long_context_accuracy": best_metric,
                "final_eval": last_eval,
            }
            write_json(summary_path, summary)
            tracker.complete(summary=f"best_long_context_accuracy={best_metric:.4f}")
    except Exception as exc:
        if "tracker" in locals() and runtime.is_main_process:
            tracker.fail(summary=str(exc))
        raise
    finally:
        cleanup_runtime(runtime)


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
