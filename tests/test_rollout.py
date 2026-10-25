"""Exercise the rollout process protocol without simulator dependencies."""

import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

from praxis import rollout


def _protocol_worker(conn, jobs, *args):
    try:
        conn.send(("obs", {"value": 1}, 2, 3, 4))
        command, actions = conn.recv()
        assert command == "act"
        assert np.max(actions) == 1
        conn.send(("chain", ({"value": 1}, np.zeros((1, 7))), 2, 3, 4))
        conn.send(("episode_end", True, 2, 3, 5))
        conn.send(("worker_done", None, -1, -1, 0))
    finally:
        conn.close()


def _crashing_worker(conn, *args):
    os._exit(7)


def _silent_worker(conn, *args):
    time.sleep(30)


def _assert_clean(exposure):
    assert all(c.closed for c in exposure.conns)
    assert all(not p.is_alive() for p in exposure.procs)
    assert all(p.exitcode is not None for p in exposure.procs)
    exposure.close()


def test_protocol_callbacks_clipping_and_cleanup(monkeypatch):
    monkeypatch.setattr(rollout, "_worker", _protocol_worker)
    exposure = rollout.SelfExposure([("job",)], n_workers=1, stuck_s=10)
    samples, episodes, chains = [], [], []
    exposure.on_episode_end = lambda *args: episodes.append(args)
    exposure.on_chain = lambda *args: chains.append(args)

    def actions(obs, tasks, meta):
        assert obs == [{"value": 1}]
        assert tasks == [2]
        assert meta == [(3, 4)]
        return np.full((1, 8, 7), 2.0)

    exposure.run(actions, lambda *args: samples.append(args), pass_meta=True)
    assert samples == [({"value": 1}, 2, 3, 4)]
    assert episodes == [(True, 2, 3, 5)]
    assert len(chains) == 1
    _assert_clean(exposure)


def test_crash_is_an_error_and_reaps_worker(monkeypatch):
    monkeypatch.setattr(rollout, "_worker", _crashing_worker)
    exposure = rollout.SelfExposure([("job",)], n_workers=1, stuck_s=10)
    with pytest.raises(RuntimeError, match="worker 0.*(disconnected|exited)"):
        exposure.run(lambda *_: None, lambda *_: None)
    _assert_clean(exposure)


def test_timeout_is_an_error_and_reaps_worker(monkeypatch):
    monkeypatch.setattr(rollout, "_worker", _silent_worker)
    exposure = rollout.SelfExposure([("job",)], n_workers=1, stuck_s=0.1)
    with pytest.raises(TimeoutError, match="worker 0 inactive"):
        exposure.run(lambda *_: None, lambda *_: None)
    _assert_clean(exposure)


def test_callback_failure_cleans_up(monkeypatch):
    monkeypatch.setattr(rollout, "_worker", _protocol_worker)
    exposure = rollout.SelfExposure([("job",)], n_workers=1, stuck_s=10)

    def fail(*_):
        raise ValueError("policy failed")

    with pytest.raises(ValueError, match="policy failed"):
        exposure.run(fail, lambda *_: None)
    _assert_clean(exposure)


def test_empty_jobs_complete_without_starting_workers():
    exposure = rollout.SelfExposure([])
    exposure.run(lambda *_: pytest.fail("unexpected policy call"), lambda *_: None)
    _assert_clean(exposure)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_workers": 0},
        {"n_workers": 1.5},
        {"exec_steps": 0},
        {"max_steps": -1},
        {"res": 0},
        {"temporal_stride": 0},
        {"stuck_s": 0},
        {"stuck_s": np.inf},
        {"ensemble_n": 0},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        rollout.SelfExposure([], **kwargs)


def test_init_state_load_is_explicit_and_does_not_replace_torch_load(monkeypatch):
    import torch

    calls = []

    def load(path, **kwargs):
        calls.append((path, kwargs))
        return [1]

    monkeypatch.setattr(torch, "load", load)
    task = SimpleNamespace(problem_folder="suite", init_states_file="task.pruned_init")
    assert rollout._load_init_states(task, lambda _: "/assets") == [1]
    assert calls == [("/assets/suite/task.pruned_init", {"weights_only": False})]
    assert torch.load is load


def test_worker_closes_environment_and_pipe_on_simulator_failure(monkeypatch):
    import sys
    from types import ModuleType

    class Environment:
        closed = False

        def reset(self):
            raise RuntimeError("reset failed")

        def close(self):
            self.closed = True

    env = Environment()
    conn = SimpleNamespace(closed=False)
    conn.close = lambda: setattr(conn, "closed", True)
    task = SimpleNamespace(problem_folder="suite", bddl_file="task.bddl")
    benchmark = SimpleNamespace(
        get_benchmark_dict=lambda: {"suite": lambda: SimpleNamespace(get_task=lambda _: task)}
    )
    package = ModuleType("libero")
    api = ModuleType("libero.libero")
    api.benchmark = benchmark
    api.get_libero_path = lambda _: "/assets"
    envs = ModuleType("libero.libero.envs")
    envs.OffScreenRenderEnv = lambda **_: env
    monkeypatch.setitem(sys.modules, "libero", package)
    monkeypatch.setitem(sys.modules, "libero.libero", api)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", envs)
    monkeypatch.setattr(rollout, "_load_init_states", lambda *_: [np.zeros(1)])
    with pytest.raises(RuntimeError, match="reset failed"):
        rollout._worker(conn, [("suite", 0, 0, 0, 0)], 128, 8, 400)
    assert env.closed
    assert conn.closed
