"""Exercise training adapters and EMA on small, fully specified examples."""

import numpy as np
import pytest
import torch
from torch import nn

from praxis.train_eff import build_parser, validate_args
from praxis.training_utils import EMA, ActSet, SealFlow


def test_ema_averages_parameters_and_copies_discrete_buffers():
    model = nn.BatchNorm1d(2)
    ema = EMA(model, decay=0.75)
    with torch.no_grad():
        model.weight.fill_(5)
        model.running_mean.fill_(8)
        model.num_batches_tracked.fill_(7)
    # The shadow owns its storage and does not follow model mutations.
    torch.testing.assert_close(ema.shadow["weight"], torch.ones(2))
    ema.update(model)
    torch.testing.assert_close(ema.shadow["weight"], torch.full((2,), 2.0))
    torch.testing.assert_close(ema.shadow["running_mean"], torch.full((2,), 2.0))
    assert ema.shadow["num_batches_tracked"].item() == 7
    model.load_state_dict(ema.shadow)
    torch.testing.assert_close(model.weight, torch.full((2,), 2.0))


def test_demo_adapter_preserves_normalized_targets_and_subtask():
    sample = {
        "imgs": torch.zeros(2, 3, 2, 2),
        "state": torch.arange(8).float(),
        "task": torch.tensor(3),
        "action": torch.full((8, 7), -0.5),
        "sub": torch.tensor(1),
        "irrelevant": "metadata",
    }
    stats = (np.zeros(7), np.ones(7), np.zeros(8), np.ones(8))
    adapter = ActSet([sample], None, stats)
    assert len(adapter) == 1
    result = adapter[0]
    assert set(result) == {"imgs", "state", "task", "action", "sub"}
    assert result["action"] is sample["action"]
    assert result["state"] is sample["state"]
    assert result["sub"].item() == 1


@pytest.mark.parametrize("with_subtask", [False, True])
def test_corrective_adapter_normalizes_raw_chunks_and_camera_layout(tmp_path, with_subtask):
    # Two cameras with distinguishable channels detect misplaced camera/channel axes.
    images = np.empty((1, 2, 2, 2, 3), dtype=np.uint8)
    images[:, 0] = [0, 128, 255]
    images[:, 1] = [255, 0, 64]
    images.tofile(tmp_path / "imgs.u8")
    raw_actions = np.arange(8 * 7, dtype=np.float32).reshape(1, 8, 7)
    raw_states = np.arange(8, dtype=np.float32).reshape(1, 8)
    np.savez(
        tmp_path / "idx.npz",
        state=raw_states,
        act=raw_actions,
        task=np.array([4]),
        **({"sub": np.array([2])} if with_subtask else {}),
    )
    stats = (np.ones(7), np.full(7, 2), np.full(8, 2), np.full(8, 4))
    dataset = SealFlow(tmp_path, 1, 2, stats)
    result = dataset[0]
    assert result["imgs"].shape == (2, 3, 2, 2)
    torch.testing.assert_close(result["imgs"][0, :, 0, 0], torch.tensor([0, 128 / 255, 1]))
    torch.testing.assert_close(result["imgs"][1, :, 1, 1], torch.tensor([1, 0, 64 / 255]))
    torch.testing.assert_close(result["action"], torch.tensor((raw_actions[0] - 1) / 2))
    torch.testing.assert_close(result["state"], torch.tensor((raw_states[0] - 2) / 4))
    assert result["task"].dtype == torch.long
    assert result["task"].item() == 4
    assert result["sub"].item() == (2 if with_subtask else -1)


@pytest.mark.parametrize(
    "args",
    [
        ["--bs", "0"],
        ["--workers", "-1"],
        ["--rsi_lo", "0.9", "--rsi_hi", "0.2"],
        ["--ema", "nan"],
        ["--dropout", "1"],
        ["--out", "../escape"],
        ["--device", "cpu"],
        ["--det", "2"],
        ["--rounds", "1", "--workers", "0"],
    ],
)
def test_invalid_cli_settings_fail_before_loading_training_dependencies(args):
    parser = build_parser()
    with pytest.raises(SystemExit) as exc:
        validate_args(parser.parse_args(args), parser)
    assert exc.value.code == 2
