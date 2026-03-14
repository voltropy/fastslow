import torch

from fastslow.data import TaskSpec, make_batch


def test_make_batch_marks_only_final_target() -> None:
    spec = TaskSpec()
    inputs, targets, mask = make_batch(batch_size=4, num_pairs=8, spec=spec, device=torch.device("cpu"))

    assert inputs.shape == targets.shape == mask.shape
    assert mask.sum().item() == 4
    assert torch.all(mask[:, -1])
    assert not torch.any(mask[:, :-1])
