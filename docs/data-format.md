# Packed demonstration format

`praxis.data.LiberoChunks` reads a directory containing `index.npz` and `blob.u8`.
The original conversion script and demonstrations are not included. This document describes
the reader contract so a compatible pack can be prepared without guessing its layout.

## Files and arrays

`blob.u8` is the concatenation of the JPEG byte strings for both cameras. It is opened as
a read-only `uint8` memory map. Offsets and byte lengths locate each frame.

| Array in `index.npz` | Shape | Meaning |
|:---|:---|:---|
| `img_off`, `img_len` | `(N,)` | Offset and byte length of each agent-camera JPEG |
| `wr_off`, `wr_len` | `(N,)` | Offset and byte length of each wrist-camera JPEG |
| `state` | `(N, 8)` | Position (3), axis-angle orientation (3), gripper joint positions (2) |
| `act` | `(N, 7)` | End-effector translation delta (3), rotation delta (3), gripper (1) |
| `task` | `(N,)` | Integer model task ID |
| `ep_start`, `ep_len` | `(E,)` | First row and length of each episode |

Episodes must stay within the flattened arrays and preserve temporal order. Camera frames
must align with the state and action at the same row. Use the same image orientation as
the policy observations; the rollout wrapper flips the LIBERO camera images vertically.
Task IDs must agree with the checkpoint embedding and the evaluation task map.

The reader preserves the upstream chunk sampling rule: a length-H chunk must leave one
trailing frame in the episode. State and visual history are padded at episode boundaries
with the earliest in-episode observation. Images are resized to the configured resolution,
converted to RGB and normalized to `[0, 1]`.

## Normalization and output

Statistics are ordered `(action_mean, action_std, state_mean, state_std)` with lengths
`7, 7, 8, 8`. Standard deviations must be finite and strictly positive. The bundled file
is a legacy NumPy object array. Do not reuse it for data with different state/action semantics.

Each dataset item has `imgs`, `state`, `action`, `task` and `sub` keys. The optional
subtask label is `-1` when absent. Inference expects **raw** state values and applies
normalization internally; dataset items are already normalized for training.

## Self-exposure buffers

`praxis.training_utils.SealFlow` reads `idx.npz` (`state`, `act`, `task`, optional `sub`)
and `imgs.u8`, a raw image memory map of shape `(samples, 2, resolution, resolution, 3)`.
Training writes these buffers as temporary data. Keep experiment buffers outside version
control and use distinct directories when they need to be retained.
