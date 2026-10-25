"""Check analytic labels independently of the simulator runtime."""

from types import SimpleNamespace

import numpy as np
import pytest

from praxis import seal


def test_no_correction_preserves_feedforward_without_simulator(monkeypatch):
    def forbidden():
        pytest.fail("no-correction labels must not import simulator transforms")

    monkeypatch.setattr(seal, "_transforms", forbidden)
    states = np.zeros((2, 7))
    actions = np.array([[0.1, 0.2, 0.3, -0.2, 0.4, 0.5, 2.0], [0.2] * 7])
    result = seal.getback_action(
        np.ones(3), np.array([0, 0, 0, 1]), states, actions, 0, no_correct=True
    )
    np.testing.assert_allclose(result, [0.1, 0.2, 0.3, -0.2, 0.4, 0.5, 1.0])
    np.testing.assert_equal(actions[0, 6], 2.0)
    assert result.dtype == np.float32


@pytest.mark.parametrize("horizon", [None, 4, 8])
def test_cap_preserves_position_direction(monkeypatch, horizon):
    # Isolate translation: both orientations are the identity. Quaternion composition
    # remains delegated to robosuite in production; this stub supplies only that identity.
    monkeypatch.setattr(
        seal,
        "_transforms",
        lambda: SimpleNamespace(axisangle2quat=lambda _: np.array([0, 0, 0, 1])),
    )
    monkeypatch.setattr(seal, "rot_err", lambda *_: np.zeros(3))
    states = np.zeros((1, 7))
    states[0, :3] = [0.3, -0.15, 0.075]
    actions = np.zeros((1, 7))
    result = seal.getback_action(
        np.zeros(3), np.array([0, 0, 0, 1]), states, actions, 0, cap=0.8, horizon=horizon
    )
    np.testing.assert_allclose(result[:3], [0.8, -0.4, 0.2])
    assert np.max(np.abs(result[:6])) <= 0.8 + 1e-7


def test_alignment_does_not_cross_gripper_transition(monkeypatch):
    states = np.zeros((3, 7))
    states[:, 0] = [0, 1, 2]
    actions = np.zeros((3, 7))
    actions[:, 6] = [-1, 1, 1]
    result = seal.getback_action(
        np.array([2, 0, 0]), None, states, actions, 0, align=True, no_correct=True
    )
    assert result[6] == -1


def test_rotation_error_uses_simulator_transforms():
    pytest.importorskip("robosuite", reason="rotation integration requires optional robosuite")
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(seal.rot_err(identity, identity), np.zeros(3), atol=1e-7)
