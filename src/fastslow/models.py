from __future__ import annotations

import math
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from fastslow.data import TaskSpec


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 12
    d_ff: int = 768
    dropout: float = 0.1
    max_seq_len: int = 1024
    d_slow: int = 64
    slow_update_gap: int = 4
    slow_block_count: int | None = None


def _sinusoidal_positions(length: int, width: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / max(width, 2))
    )
    pe = torch.zeros(length, width, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class CausalSelfAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError(f"width={width} must be divisible by heads={heads}")
        self.heads = heads
        self.head_dim = width // heads
        self.dropout = dropout
        self.qkv = nn.Linear(width, width * 3)
        self.out = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, width = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, width)
        return self.out(attn)


class Mlp(nn.Module):
    def __init__(self, width: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int, ff_width: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = CausalSelfAttention(width, heads, dropout)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = Mlp(width, ff_width, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class StandardTransformerLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
                for _ in range(config.n_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq = tokens.shape
        if seq > self.config.max_seq_len:
            raise ValueError(f"sequence length {seq} exceeds max_seq_len={self.config.max_seq_len}")
        positions = _sinusoidal_positions(seq, self.config.d_model, tokens.device)
        x = self.token_embedding(tokens) + positions.unsqueeze(0)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_ln(x)
        return self.lm_head(x)


class FastSlowTransformerLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.d_slow <= 0:
            raise ValueError("d_slow must be positive for fast/slow models")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.main_blocks = nn.ModuleList(
            [
                TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
                for _ in range(config.n_layers)
            ]
        )
        self.init_slow = nn.Linear(config.d_model, config.d_slow)
        self.slow_to_main = nn.Linear(config.d_slow, config.d_model)
        self.main_to_slow = nn.Linear(config.d_model, config.d_slow)

        slow_heads = max(1, min(config.n_heads, config.d_slow))
        while config.d_slow % slow_heads != 0:
            slow_heads -= 1
        slow_block_count = config.slow_block_count or math.ceil(config.n_layers / config.slow_update_gap)
        slow_ff = max(config.d_slow * 4, config.d_ff * config.d_slow // config.d_model)
        self.slow_blocks = nn.ModuleList(
            [
                TransformerBlock(config.d_slow, slow_heads, slow_ff, config.dropout)
                for _ in range(slow_block_count)
            ]
        )
        self.final_ln = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, seq = tokens.shape
        if seq > self.config.max_seq_len:
            raise ValueError(f"sequence length {seq} exceeds max_seq_len={self.config.max_seq_len}")
        positions = _sinusoidal_positions(seq, self.config.d_model, tokens.device)
        main = self.token_embedding(tokens) + positions.unsqueeze(0)
        main = self.dropout(main)
        slow = self.init_slow(main)
        update_count = 0

        for layer_idx, block in enumerate(self.main_blocks):
            main = block(main + self.slow_to_main(slow))
            if layer_idx % self.config.slow_update_gap == 0:
                slow_block = self.slow_blocks[update_count % len(self.slow_blocks)]
                slow = slow_block(slow + self.main_to_slow(main))
                update_count += 1

        main = self.final_ln(main)
        return self.lm_head(main)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def make_standard_config(base: ModelConfig, width: int) -> ModelConfig:
    scale = width / base.d_model
    ff_width = max(base.n_heads, int(round(base.d_ff * scale)))
    ff_width = max(ff_width, width * 2)
    return replace(base, d_model=width, d_ff=ff_width)


def match_widened_width(
    target_params: int,
    base_config: ModelConfig,
    step_multiple: int,
) -> ModelConfig:
    best_config = base_config
    best_gap = abs(count_parameters(StandardTransformerLM(base_config)) - target_params)
    for width in range(base_config.d_model, base_config.d_model * 3 + step_multiple, step_multiple):
        candidate = make_standard_config(base_config, width)
        if candidate.d_model % candidate.n_heads != 0:
            continue
        params = count_parameters(StandardTransformerLM(candidate))
        gap = abs(params - target_params)
        if gap < best_gap:
            best_gap = gap
            best_config = candidate
    return best_config


def build_model(variant: str, config: ModelConfig) -> tuple[nn.Module, ModelConfig]:
    if variant == "baseline":
        return StandardTransformerLM(config), config
    if variant == "fastslow":
        return FastSlowTransformerLM(config), config
    if variant == "fastslow_every_layer":
        slow_count = math.ceil(config.n_layers / max(config.slow_update_gap, 1))
        adjusted = replace(config, slow_update_gap=1, slow_block_count=slow_count)
        return FastSlowTransformerLM(adjusted), adjusted
    if variant == "widened_baseline":
        target_params = count_parameters(FastSlowTransformerLM(config))
        widened = match_widened_width(target_params, make_standard_config(config, config.d_model), config.n_heads)
        return StandardTransformerLM(widened), widened
    raise ValueError(f"unknown variant: {variant}")


def default_model_config(spec: TaskSpec, max_seq_len: int) -> ModelConfig:
    return ModelConfig(vocab_size=spec.vocab_size, max_seq_len=max_seq_len)
