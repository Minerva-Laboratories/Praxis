# Setup, inference and training

## Install

Clone the repository and install it in an isolated environment. Install a CPU or CUDA
PyTorch build appropriate for your machine, then install PRAXIS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
praxis --help
```

The package supports Python 3.10+. The CPU test suite is checked on Python 3.12.
`requirements.txt` is an alternative installation entry point for the same package.

The editable install keeps bundled assets in the checkout's `checkpoints/` directory.
A built wheel contains code, not weights. For an installed wheel or a separate model
directory, set `PRAXIS_CHECKPOINT_DIR` to the absolute directory holding the checkpoints,
`normalization.npy` and `task_map.json`, or pass explicit asset paths.

```bash
praxis doctor
praxis doctor --json
```

The doctor checks that dependencies and assets exist. It does not launch a simulator
or certify CUDA/EGL readiness. A missing simulator dependency is expected if you only
want CPU policy inference.

## Policy inference

```python
import numpy as np
from praxis.inference import PraxisPolicy

policy = PraxisPolicy("praxis-regression.pt", dev="cpu", deterministic=True)
images = np.zeros((1, 2, 128, 128, 3), dtype=np.uint8)
state = policy.s_mean[None, :].copy()
actions = policy.act(images, state, np.array([0]))
assert actions.shape == (1, 8, 7)
```

This example supplies synthetic observations solely to exercise the API. Use real camera
observations and raw proprioceptive state for inference on an environment. The wrapper
normalizes state and de-normalizes actions. Inputs are two RGB images per item, eight
state values and an integer task ID. Outputs are clipped normalized controller actions.
Use `dev="cuda"` for a GPU. BF16 autocast is used when supported; other paths use FP32.

The original `ContInfer` API and `python -m praxis.eval_cont` entry point remain available.
For temporal checkpoints, supply `(episode, step)` pairs as `metas` and reset episode memory
with `policy.reset()`, or `policy.reset(task=..., demo=...)` for one episode.

## LIBERO evaluation

Install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO#installation), including its
robosuite/MuJoCo dependencies, task assets and initial-state files. Configure LIBERO's paths
and EGL rendering for your system. If using a source checkout, add its package parent to
`PYTHONPATH`. No packed training demonstrations are needed to evaluate the included policies.

```bash
export MUJOCO_GL=egl
praxis evaluate --ckpt checkpoints/praxis-regression.pt \
  --episodes 10 --workers 4 --deterministic

praxis evaluate --ckpt checkpoints/praxis-generative.pt \
  --episodes 10 --workers 4 --deterministic
```

The CLI defaults to CUDA; add `--device cpu` to use a CPU. `--deterministic` selects prior-mean
decoding for the generative head. It does not guarantee bitwise reproducibility of the simulator
or training process. `--episodes 1` shortens evaluation but still visits every task in the task map.

`--stats` accepts a bundled basename or an explicit relative/absolute path. `--task-map` selects
a mapping of model task IDs to LIBERO suite/task indices. Use `praxis evaluate --help` for all
options. Report simulator and dependency versions with any new benchmark; this release does not
claim a pinned reproduction environment. [Paper and checkpoint protocols](../README.md#results)
explain which evaluation produced each reported number.

## Train

Training requires a CUDA GPU and a [compatible demonstration pack](data-format.md).
The original dataset packer and training demonstrations are not distributed here.
Downloading raw LIBERO data does not by itself produce the required packed format.

The original pipeline entry points are preserved below. These commands illustrate the
stages; they are not a complete reconstruction of the corrected paper's aggregate.

```bash
praxis train --pack data/libero_merged --n_tasks 113 \
  --opt nadamw --lr 3e-4 --bc_epochs 8 \
  --rounds 4 --episodes 8 --seal_epochs 2 --rsi --out pretrain

praxis train --pack data/libero_merged --n_tasks 113 \
  --task_lo 0 --task_hi 39 --init pretrain_seal_r4 \
  --bc_epochs 8 --rounds 4 --episodes 8 --seal_epochs 2 --rsi --out ft
```

Use normalization statistics matching your data. `--checkpoint-dir` controls default asset
lookup and checkpoint output; `--out` is a filename prefix. `--init` accepts a checkpoint
stem or path. Use a different `--seal_dir` for each experiment whose self-exposure data must
be retained. The historical seed/deterministic options do not eliminate all sources of
asynchronous rollout variation.

## Development

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
ruff check praxis tests
ruff format --check praxis tests
python -m build --wheel
```

Tests use synthetic data and subprocesses; they do not run full LIBERO benchmarks or train
policies. Real checkpoint smoke inference is checked separately on CPU. Optional tests that
require robosuite are skipped when it is not installed.

Model checkpoints use PyTorch's restricted `weights_only=True` loader. The original NumPy
normalization file and LIBERO initial-state assets use legacy pickle-capable formats; load
those assets only from trusted sources.
