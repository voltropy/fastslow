import torch

from fastslow.data import TaskSpec
from fastslow.models import build_model, count_parameters, default_model_config


def test_fastslow_forward_shape() -> None:
    spec = TaskSpec()
    config = default_model_config(spec, max_seq_len=128)
    model, _ = build_model("fastslow", config)
    tokens = torch.randint(0, spec.vocab_size, (2, 32))
    logits = model(tokens)
    assert logits.shape == (2, 32, spec.vocab_size)


def test_widened_baseline_matches_fastslow_order_of_magnitude() -> None:
    spec = TaskSpec()
    config = default_model_config(spec, max_seq_len=128)
    fastslow, fastslow_config = build_model("fastslow", config)
    widened, widened_config = build_model("widened_baseline", config)

    fastslow_params = count_parameters(fastslow)
    widened_params = count_parameters(widened)

    assert fastslow_config.d_model == config.d_model
    assert widened_config.d_model >= config.d_model
    assert abs(fastslow_params - widened_params) / fastslow_params < 0.25
