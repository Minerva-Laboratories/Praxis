# Checkpoints

- `praxis-regression.pt`: original regression policy snapshot.
- `praxis-generative.pt`: original conditional generative policy snapshot.
- `normalization.npy`: action/state means and standard deviations.
- `task_map.json`: model task ID to LIBERO suite and task index.

The weight files were renamed for clarity; their bytes are unchanged from the upstream
release. See the [README](../README.md#results) for the distinction between these snapshots
and the corrected paper's aggregate results. The three-run paper checkpoint set and
Arena G1 weights are not included.

The default path is this directory when using an editable installation. Set
`PRAXIS_CHECKPOINT_DIR` to use a different asset directory. Weight files are excluded
from the built Python wheel.
