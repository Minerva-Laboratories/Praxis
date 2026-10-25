# Method and architecture

PRAXIS controls a known repertoire of manipulation tasks using task identifiers.
Two RGB views and proprioception describe the current observation. Task-prefix tokens,
a state token and action queries pass through a compact cross-attention trunk. The
regression head decodes the action queries with temporal convolutions.

## Self-exposure with aligned labels

SEAL initializes a simulator rollout from an anchor on a recorded demonstration.
The learner executes one observation interval. Each visited state is paired with the
aligned demonstration action, optionally augmented with an analytic recovery term:

```text
label = aligned demonstration action + optional controller-inverse recovery term
```

The action-space inverse maps commanded end-effector deltas to normalized actions.
It is not a learned dynamics model and does not guarantee that the robot reaches the
reference pose in one step. Rotation error uses quaternion composition. Translation
and rotation groups are capped together within each group to preserve direction.

`praxis/seal.py` implements label construction. `praxis/rollout.py` owns simulator
workers and delivers observations, trajectory metadata and completion callbacks.
`praxis/train_eff.py` mixes demonstrations and self-exposure data through successive rounds.

The corrected paper calls SEAL **Self-Exposure with Aligned Labels**. Older source
uses “Analytic Labels” for the corrected variant. Both variants are available through
the training options. The paper's results and qualifications live in the [README](../README.md#results).

## Two policy heads

- **Regression:** produces a complete action chunk in one forward pass.
- **Conditional generative:** learns a conditional prior over the latent action
  representation. Deterministic inference uses the prior mean; stochastic inference
  samples it. Its recognition encoder is only needed during training.

The public API is `PraxisPolicy` in `praxis.inference`. Model class names and state-dictionary
keys are retained for compatibility with the original weights. The temporal variants
keep buffers by task and episode; call `reset()` at an episode boundary when driving the
API yourself, and provide `(episode, step)` metadata for temporal checkpoints.

## Package boundaries

| Layer | Modules |
|:---|:---|
| Commands and environment check | `cli`, `eval_cont`, `train_eff` |
| Inference and validated assets | `inference`, `cont_infer`, `config` |
| Policies and representations | `policy*`, `vision*`, `rope`, `temporal` |
| Training data and shared helpers | `data`, `training_utils`, `augment`, `utils` |
| Self-exposure and correction | `rollout`, `seal` |
| Optimizer and optional adaptation | `musgd`, `lora` |

Simulator imports are delayed until the operation needs them. CLI help works without
LIBERO. CPU policy inference works without a renderer. Rollout failures raise errors
and close their worker processes instead of silently returning an incomplete evaluation.

## Provenance

The source was cloned from [`alexhergomz/Praxis`](https://github.com/alexhergomz/Praxis)
at commit `f4e463840f243625a4ed7cbec395d594008e1efb`. This refactor preserves learned
weights, controller constants, policy state-dictionary names and training equations.
It removes historical command scripts, results indexes, logs and manuscript drafts
from the working tree. Numerical findings are presented in the README.
