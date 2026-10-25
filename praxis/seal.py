"""Self-Exposure with Analytic Labels (SEAL) for LIBERO policies.

At a visited state s', the analytic target adds the controller-space inverse of
reference drift to the demonstration action:

    a* = a_ref[k] + B^+ (s_ref[k] (-) s')

For robosuite OSC_POSE, action 1.0 commands 0.05 m translation or 0.5 rad
rotation, so B = diag([0.05]*3 + [0.5]*3). These are commanded deltas;
achieved motion must not be substituted into the action-space inverse.

Rotation error uses q_ref * inverse(q_cur), avoiding axis-angle subtraction's
singularity near pi. Optional monotonic path alignment separates trajectory
progress from off-path drift while retaining gripper phases.
"""

from __future__ import annotations
import numpy as np

# ---- exact action-space inverse from the OSC controller spec ----
SCALE = np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])  # output_max: action 1.0 -> this delta
BINV = 1.0 / SCALE  # B^+  (diag, exact -- no free gain)
ETA = 0.250  # OSC achieved/commanded displacement ratio used by the horizon parameterization.
# Closure per step is g*ETA, so N-step recovery uses g = 1/(ETA*N).


def _transforms():
    """Import simulator transforms only when a rotation label is needed."""
    from robosuite.utils import transform_utils

    return transform_utils


def rot_err(q_ref, q_cur):
    """proper rotation error as an axis-angle vector (safe at |aa| ~ pi)."""
    T = _transforms()
    return T.quat2axisangle(T.quat_multiply(q_ref, T.quat_inverse(q_cur)))


def align_k(s_prime_pos, demo_pos, k_prev, window=20):
    """monotonic nearest projection of the visited eef position onto the demo path."""
    hi = min(len(demo_pos), k_prev + window)
    seg = demo_pos[k_prev:hi]
    if len(seg) == 0:
        return len(demo_pos) - 1
    return k_prev + int(np.argmin(np.linalg.norm(seg - s_prime_pos, axis=1)))


def getback_action(
    pos_cur,
    quat_cur,
    demo_state,
    demo_act,
    k,
    cap=1.0,
    exact_rot=False,
    no_correct=False,
    horizon=None,
    align=False,
    align_window=20,
):
    """Label one visited state using the reference action and inverse drift correction.

    By default k indexes demonstration time. With align=True, project forward
    onto the nearest reference position unless that crosses a gripper transition.

    exact_rot composes the demonstration rotation delta with the reference
    orientation before measuring error from quat_cur. The default adds the
    feed-forward and error as axis-angle vectors, a small-angle approximation.
    Translation composes linearly in both cases.

    Scale each translation/rotation group together when its largest component
    exceeds cap, preserving its direction. With horizon=None, cap the combined
    command. Otherwise scale correction by 1/(ETA*horizon), cap it first, and
    use the remaining budget for feed-forward. Copy and clip the gripper action.
    no_correct returns reference feed-forward with only gripper clipping.
    """
    L = len(demo_act)
    k = int(min(max(k, 0), L - 1))
    if align:
        # Project onto the path so schedule differences do not become positional corrections.
        k_new = int(
            min(
                max(align_k(np.asarray(pos_cur), demo_state[:, :3], k, window=align_window), 0),
                L - 1,
            )
        )
        # Position paths can self-intersect across grasp phases. Keep the time index if
        # projection crosses a discrete gripper transition.
        if k_new != k:
            lo, hi = (k, k_new) if k < k_new else (k_new, k)
            if np.any(np.sign(demo_act[lo : hi + 1, 6]) != np.sign(demo_act[lo, 6])):
                k_new = k
        k = k_new
    s_ref = demo_state[k]
    e_pos = s_ref[:3] - pos_cur
    a = demo_act[k].copy()
    if no_correct:
        # Exposure-only ablation: retain reference feed-forward without drift correction.
        a[6] = np.clip(a[6], -1.0, 1.0)
        return a.astype(np.float32)
    ff = a[:6].copy()  # the demo's own feed-forward, kept separate
    c_pos = BINV[:3] * e_pos  # desired one-step correction
    T = _transforms()
    if exact_rot:  # compose rotations as quaternions (exact)
        q_ref = T.axisangle2quat(s_ref[3:6])
        q_next = T.quat_multiply(
            T.axisangle2quat(ff[3:6] * SCALE[3:6]), q_ref
        )  # demo's next orientation
        c_rot = (
            BINV[3:6] * rot_err(q_next, quat_cur) - ff[3:6]
        )  # exact_rot yields the WHOLE command, so the
    else:  # correction is what it adds beyond feed-forward
        c_rot = BINV[3:6] * rot_err(T.axisangle2quat(s_ref[3:6]), quat_cur)
    if horizon is None:
        # Cap the combined feed-forward and correction, preserving each group's direction.
        a[:3] = ff[:3] + c_pos
        a[3:6] = ff[3:6] + c_rot
        for sl in (slice(0, 3), slice(3, 6)):  # direction-preserving cap per group (not per-dim)
            m = np.abs(a[sl]).max()
            if m > cap:
                a[sl] = a[sl] * (cap / m)
    else:
        # Return over `horizon` steps. Correction takes priority on the action budget;
        # the feed-forward uses the remainder so it cannot displace the recovery command.
        g = 1.0 / (ETA * float(horizon))
        for sl, c in ((slice(0, 3), c_pos * g), (slice(3, 6), c_rot * g)):
            m = np.abs(c).max()  # correction first, capped to the box itself
            if m > cap:
                c = c * (cap / max(m, 1e-12))
            budget = max(cap - np.abs(c).max(), 0.0)  # feed-forward gets the remainder
            f = ff[sl]
            mf = np.abs(f).max()
            if mf > budget:
                f = f * (budget / max(mf, 1e-12))
            a[sl] = f + c
    a[6] = np.clip(a[6], -1.0, 1.0)  # gripper is a binary decision -> plain clip
    return a.astype(np.float32)


def corrective_chunk(pos_cur, quat_cur, demo_state, demo_act, k, H):
    """Compatibility label with correction only on the chunk's first action.

    a*[0] = a_ref[k] + B^+ (s_ref[k] (-) s'); remaining actions follow the demo.
    """
    L = len(demo_act)
    k = min(k, L - 1)
    s_ref = demo_state[k]
    e_pos = s_ref[:3] - pos_cur
    T = _transforms()
    q_ref = T.axisangle2quat(s_ref[3:6])
    e_rot = rot_err(q_ref, quat_cur)
    a0 = demo_act[k].copy()
    a0[:3] = a0[:3] + BINV[:3] * e_pos
    a0[3:6] = a0[3:6] + BINV[3:6] * e_rot
    # Remaining steps retain the demonstration actions.
    rest = demo_act[k + 1 : k + H]
    if len(rest) < H - 1:  # pad at episode end with the last action
        pad = np.repeat(demo_act[-1:], H - 1 - len(rest), axis=0)
        rest = np.concatenate([rest, pad], 0) if len(rest) else pad
    chunk = np.concatenate([a0[None], rest], 0)[:H]
    return np.clip(chunk, -1.0, 1.0).astype(np.float32)
