from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TaskSpec:
    key_vocab_size: int = 256
    value_vocab_size: int = 256

    @property
    def pair_token(self) -> int:
        return 1

    @property
    def query_token(self) -> int:
        return 2

    @property
    def key_offset(self) -> int:
        return 3

    @property
    def value_offset(self) -> int:
        return self.key_offset + self.key_vocab_size

    @property
    def vocab_size(self) -> int:
        return self.value_offset + self.value_vocab_size


def _sample_unique_keys(batch_size: int, num_pairs: int, key_vocab_size: int, device: torch.device) -> torch.Tensor:
    if num_pairs > key_vocab_size:
        raise ValueError(f"num_pairs={num_pairs} exceeds key vocabulary {key_vocab_size}")
    keys = torch.stack(
        [torch.randperm(key_vocab_size, device=device)[:num_pairs] for _ in range(batch_size)],
        dim=0,
    )
    return keys


def make_batch(
    batch_size: int,
    num_pairs: int,
    spec: TaskSpec,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a causal-LM batch whose only supervised token is the final answer."""

    keys = _sample_unique_keys(batch_size, num_pairs, spec.key_vocab_size, device=device)
    values = torch.randint(0, spec.value_vocab_size, (batch_size, num_pairs), device=device)

    sequence = torch.empty(batch_size, num_pairs * 3 + 3, dtype=torch.long, device=device)
    sequence[:, 0::3][:, :num_pairs] = spec.pair_token
    sequence[:, 1::3][:, :num_pairs] = keys + spec.key_offset
    sequence[:, 2::3][:, :num_pairs] = values + spec.value_offset

    query_index_limit = max(1, num_pairs // 4)
    query_positions = torch.randint(0, query_index_limit, (batch_size,), device=device)
    batch_indices = torch.arange(batch_size, device=device)

    sequence[:, -3] = spec.query_token
    sequence[:, -2] = keys[batch_indices, query_positions] + spec.key_offset
    sequence[:, -1] = values[batch_indices, query_positions] + spec.value_offset

    inputs = sequence[:, :-1].contiguous()
    targets = sequence[:, 1:].contiguous()
    loss_mask = torch.zeros_like(targets, dtype=torch.bool)
    loss_mask[:, -1] = True
    return inputs, targets, loss_mask
