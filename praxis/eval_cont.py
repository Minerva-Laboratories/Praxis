"""Evaluate task-conditioned policies in parallel LIBERO simulators."""

from __future__ import annotations

import argparse
import math
import numpy as np

from .config import DEFAULT_STATS, DEFAULT_TASK_MAP, load_task_map

MAXS = {"libero_spatial": 400, "libero_object": 400, "libero_goal": 400, "libero_10": 600}


def _positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def _temporal_weight(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return number


def evaluate(
    ckpt,
    episodes=10,
    workers=4,
    steps=5,
    stats=DEFAULT_STATS,
    deterministic=False,
    exec_steps=None,
    temporal_m=None,
    temporal_stride=1,
    ensemble_n=None,
    wrist_res=0,
    global_res=128,
    oracle_phase=False,
    init_offset=0,
    device="cuda",
    task_map=DEFAULT_TASK_MAP,
):
    """Return suite success, task wins/counts and per-task results.

    Model and simulator imports are deferred so the command's help and argument
    validation work without a simulator installation.
    """
    for name, value in (
        ("episodes", episodes),
        ("workers", workers),
        ("steps", steps),
        ("temporal_stride", temporal_stride),
        ("global_res", global_res),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("exec_steps", exec_steps), ("ensemble_n", ensemble_n)):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("wrist_res", wrist_res), ("init_offset", init_offset)):
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if temporal_m is not None and (not math.isfinite(temporal_m) or temporal_m < 0):
        raise ValueError("temporal_m must be finite and nonnegative")
    tmap = load_task_map(task_map)
    unsupported = {suite for suite, _ in tmap.values()} - MAXS.keys()
    if unsupported:
        raise ValueError(f"Unsupported LIBERO suites in task map: {sorted(unsupported)}")
    try:
        import torch
        from robosuite.utils.transform_utils import quat2axisangle
        from .cont_infer import ContInfer
        from .rollout import SelfExposure
    except ImportError as exc:
        raise RuntimeError(
            "Evaluation requires PyTorch, LIBERO and robosuite; install the simulator dependencies"
        ) from exc

    torch.manual_seed(0)
    pol = ContInfer(ckpt, steps=steps, stats=stats, deterministic=deterministic, dev=device)
    if max(tmap) >= pol.n_tasks:
        raise ValueError("Task map contains task IDs outside the checkpoint's task vocabulary")
    if exec_steps is not None and exec_steps > pol.chunk:
        raise ValueError("exec_steps cannot exceed the checkpoint's action chunk")
    if ensemble_n is not None and ensemble_n > pol.chunk:
        raise ValueError("ensemble_n cannot exceed the checkpoint's action chunk")
    if oracle_phase and type(pol.m).__name__ != "ContSubtaskEff":
        raise ValueError("oracle_phase requires a subtask_eff checkpoint")
    orders = np.load("data/subtask_labels.npz")["order"] if oracle_phase else None
    if wrist_res > 0:
        if not hasattr(pol.m, "net") or not hasattr(pol.m.net, "cam_res"):
            raise ValueError("This checkpoint does not support a per-camera resolution override")
        pol.m.net.cam_res = [global_res, wrist_res]
        pol.res = wrist_res
    ex = exec_steps or pol.chunk
    if temporal_m is not None and ensemble_n is None:
        ex = 1
    items = sorted(tmap.items())
    jobs = [
        (suite, tid, ep, task, ep + init_offset, MAXS[suite])
        for task, (suite, tid) in items
        for ep in range(episodes)
    ]
    use_hist = pol.hist > 1 or pol.vhist > 1

    def act_fn(observations, tasks, meta=None):
        images = np.stack([np.stack([obs["img"], obs["wrist"]]) for obs in observations])
        states = np.stack(
            [
                np.concatenate([obs["pos"], quat2axisangle(obs["quat"]), obs["grip"]])
                for obs in observations
            ]
        )
        phases = [int(obs.get("phase", -1)) for obs in observations] if oracle_phase else None
        return pol.act(images, states, tasks, meta if use_hist else None, phases)

    wins, runs = {}, {}
    task_suites = {task: suite for task, (suite, _) in items}

    def on_end(success, task, demo, step):
        wins[task] = wins.get(task, 0) + int(success)
        runs[task] = runs.get(task, 0) + 1
        completed = sum(runs.values())
        if completed % 40 == 0:
            per_suite = {}
            for task_id, count in runs.items():
                per_suite.setdefault(task_suites[task_id], []).append(wins[task_id] / count)
            mean = np.mean([np.mean(values) for values in per_suite.values()])
            details = " ".join(
                f"{suite}:{np.mean(values):.2f}" for suite, values in sorted(per_suite.items())
            )
            print(
                f"  [progress {completed}/{len(jobs)}] running MEAN {mean:.3f}  {details}",
                flush=True,
            )

    exposure = SelfExposure(
        jobs,
        n_workers=workers,
        res=pol.res,
        exec_steps=ex,
        max_steps=max(MAXS[suite] for _, (suite, _) in items),
        temporal_m=temporal_m,
        temporal_stride=temporal_stride,
        ensemble_n=ensemble_n,
        orders=orders,
    )
    try:
        exposure.on_episode_end = on_end
        exposure.run(act_fn, lambda *args: None, pass_meta=use_hist)
    finally:
        exposure.close()
    if any(runs.get(task, 0) != episodes for task, _ in items):
        raise RuntimeError("Evaluation ended before all requested episodes completed")
    per, per_task = {}, {}
    for task, (suite, tid) in items:
        rate = wins[task] / runs[task]
        per.setdefault(suite, []).append(rate)
        per_task[task] = (suite, tid, rate, runs[task])
    return {suite: float(np.mean(values)) for suite, values in per.items()}, wins, runs, per_task


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, help="checkpoint basename or explicit path")
    parser.add_argument("--device", default="cuda", help="PyTorch device, e.g. cpu, cuda, cuda:0")
    parser.add_argument(
        "--stats", default=str(DEFAULT_STATS), help="normalization basename or explicit path"
    )
    parser.add_argument(
        "--task-map",
        dest="task_map",
        default=str(DEFAULT_TASK_MAP),
        help="task map basename or explicit path",
    )
    parser.add_argument("--episodes", type=_positive, default=10)
    parser.add_argument("--workers", type=_positive, default=4)
    parser.add_argument("--steps", type=_positive, default=5)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--exec_steps", type=_positive, default=None)
    parser.add_argument("--temporal_m", type=_temporal_weight, default=None)
    parser.add_argument("--temporal_stride", type=_positive, default=1)
    parser.add_argument("--ensemble_n", type=_positive, default=None)
    parser.add_argument("--pertask", choices=sorted(MAXS), default=None)
    parser.add_argument("--wrist_res", type=_nonnegative, default=0)
    parser.add_argument("--global_res", type=_positive, default=128)
    parser.add_argument("--oracle_phase", action="store_true")
    parser.add_argument("--init_offset", type=_nonnegative, default=0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    options = vars(args).copy()
    pertask = options.pop("pertask")
    try:
        per, wins, runs, per_task = evaluate(**options)
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Evaluation failed: {exc}") from exc
    print(f"\n### {args.ckpt}")
    for suite, rate in sorted(per.items()):
        print(f"  {suite:16s} {rate:.3f}")
    print(
        f"  {'MEAN':16s} {np.mean(list(per.values())):.3f}   "
        f"({sum(wins.values())}/{sum(runs.values())} episodes)"
    )
    if pertask:
        print(f"--- per-task [{pertask}] ---")
        for _, (suite, tid, rate, count) in sorted(per_task.items()):
            if suite == pertask:
                print(f"  task {tid:2d}  {rate:.2f}  ({round(rate * count)}/{count})")


if __name__ == "__main__":
    main()
