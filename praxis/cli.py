"""Lightweight command dispatch; importing the CLI does not initialize PyTorch."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path


def doctor(argv=None):
    """Report local prerequisites without launching a simulator or loading weights."""
    parser = argparse.ArgumentParser(description=doctor.__doc__)
    parser.add_argument("--json", action="store_true", help="print a machine-readable report")
    args = parser.parse_args(argv)
    root = Path(
        os.environ.get("PRAXIS_CHECKPOINT_DIR", Path(__file__).resolve().parents[1] / "checkpoints")
    )
    modules = ("torch", "numpy", "PIL", "robosuite", "libero")
    dependencies = {name: importlib.util.find_spec(name) is not None for name in modules}
    assets = {
        name: (root / name).is_file()
        for name in (
            "praxis-regression.pt",
            "praxis-generative.pt",
            "normalization.npy",
            "task_map.json",
        )
    }
    versions = {}
    for name in ("torch", "numpy", "Pillow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    report = {
        "checkpoint_directory": str(root),
        "dependencies": dependencies,
        "assets": assets,
        "versions": versions,
        "note": "Presence check only; CUDA, EGL and simulator execution are not tested.",
    }
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Checkpoint directory: {root}")
        for group in (dependencies, assets):
            for name, present in group.items():
                print(f"  {'OK     ' if present else 'MISSING'} {name}")
        print(report["note"])
        if not dependencies["libero"] or not dependencies["robosuite"]:
            print(
                "Install LIBERO and its simulator dependencies before running evaluation or training."
            )
    return 0 if all(dependencies.values()) and all(assets.values()) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="praxis", description="Compact vision-action learning with self-exposure."
    )
    subcommands = parser.add_subparsers(dest="command", metavar="{evaluate,train,doctor}")
    subcommands.add_parser("evaluate", add_help=False, help="evaluate a checkpoint on LIBERO")
    subcommands.add_parser("train", add_help=False, help="train with behavior cloning and SEAL")
    subcommands.add_parser("doctor", add_help=False, help="check dependencies and bundled assets")
    args, remaining = parser.parse_known_args(argv)
    if args.command is None:
        if remaining:
            parser.error("unrecognized arguments: " + " ".join(remaining))
        parser.print_help()
        return 0
    if args.command == "doctor":
        return doctor(remaining)
    module = "praxis.eval_cont" if args.command == "evaluate" else "praxis.train_eff"
    return importlib.import_module(module).main(remaining)
