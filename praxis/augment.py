"""Visual augmentations for a from-scratch encoder.

A policy built on a frozen pretrained backbone inherits whatever invariances that backbone happened to learn.
Training the encoder ourselves removes that inheritance -- which is why texture and sensor-noise perturbations
hurt us -- but it also means the invariances can be chosen and installed directly, at training cost only.

Each function takes (B, cams, 3, H, W) in [0,1] and augments each (sample, camera) INDEPENDENTLY unless noted.
Independence across cameras matters: sharing the perturbation lets the policy keep relying on whichever view is
cleanest, while decorrelating them forces both views to be individually usable.

What these can and cannot buy: pixel-space augmentation covers appearance (texture, light, noise) and image
geometry (translation, rotation, scale). It cannot synthesise the arm in a configuration it never occupied, so
robot-initial-state robustness is not addressable here -- that requires visiting those states, which is what
widening the SEAL anchor distribution does.
"""

import torch


def apr(x, p=0.5):
    """Amplitude-Phase Recombination (Chen et al., ICCV 2021).

    Convolutional encoders are biased toward texture. The amplitude spectrum of an image carries most of its
    texture/style; the phase spectrum carries structure. Recombining the phase of one image with the amplitude
    of another keeps the geometry while replacing the surface appearance, so the encoder is pushed toward phase.
    Aimed squarely at the background-texture perturbation.
    """
    B, C2 = x.shape[0], x.shape[1]
    flat = x.reshape(-1, *x.shape[2:])  # (B*cams,3,H,W)
    n = flat.shape[0]
    take = torch.rand(n, device=x.device) < p
    if not take.any():
        return x
    f = torch.fft.rfft2(flat.float(), dim=(-2, -1))
    amp, pha = f.abs(), f.angle()
    perm = torch.randperm(n, device=x.device)  # donor of the amplitude (texture)
    lam = torch.rand(n, 1, 1, 1, device=x.device)  # partial swap -> a spectrum of strengths
    amp_mix = torch.where(take.view(-1, 1, 1, 1), lam * amp[perm] + (1 - lam) * amp, amp)
    out = torch.fft.irfft2(torch.polar(amp_mix, pha), s=flat.shape[-2:], dim=(-2, -1))
    return out.clamp(0, 1).reshape_as(x).to(x.dtype)


def cutout(x, n_holes=2, max_frac=0.25, p=0.5):
    """Random rectangular erasing, independent per (sample, camera).

    Occludes part of one view so the policy cannot depend on a single fixed region of a single camera; the
    other view has to carry the step. Holes are filled with the per-image mean rather than zero, so the
    statistics the encoder sees are not shifted by the mask itself.
    """
    B, C2, C, H, W = x.shape
    out = x.clone()
    flat = out.reshape(-1, C, H, W)
    n = flat.shape[0]
    fill = flat.mean(dim=(-2, -1), keepdim=True)
    for _ in range(n_holes):
        take = torch.rand(n, device=x.device) < p
        hh = (torch.rand(n, device=x.device) * max_frac * H).long().clamp(min=1)
        ww = (torch.rand(n, device=x.device) * max_frac * W).long().clamp(min=1)
        y0 = (torch.rand(n, device=x.device) * (H - hh).clamp(min=1)).long()
        x0 = (torch.rand(n, device=x.device) * (W - ww).clamp(min=1)).long()
        ys = torch.arange(H, device=x.device).view(1, H, 1)
        xs = torch.arange(W, device=x.device).view(1, 1, W)
        m = (
            (ys >= y0.view(-1, 1, 1))
            & (ys < (y0 + hh).view(-1, 1, 1))
            & (xs >= x0.view(-1, 1, 1))
            & (xs < (x0 + ww).view(-1, 1, 1))
            & take.view(-1, 1, 1)
        )
        flat = torch.where(m.unsqueeze(1), fill.expand_as(flat), flat)
    return flat.reshape_as(x)


def state_bias(st, sigma=1.0, dims=3):
    """Translate normalized position coordinates while retaining delta-action labels.

    The coordinate origin changes; relative displacement and the action target stay
    unchanged. Images are not transformed. sigma is measured in normalized state
    units. Rotation/scaling are not covered because their action targets would also
    need to transform."""
    b = torch.zeros_like(st)
    b[..., :dims] = torch.randn_like(st[..., :dims]) * sigma
    return st + b


def state_equivariant(st, act, gain, lo, hi, sigma=0.02, scale=0.03, dims=3):
    """EQUIVARIANT state augmentation: offset the observed state and correct the label in closed form.

    `state_bias` above teaches invariance -- perturb the input, keep the label, the policy learns to ignore the
    channel. This instead teaches the policy how the answer *changes*, which expands the state distribution
    rather than blurring it.

    The correction is exact rather than approximate. The corrective label is

        a* = a_ref + B^+ (s_ref - s'),

    so offsetting the observed state by b gives, with no simulation and no re-labelling pass,

        a**  =  a_ref + B^+ (s_ref - s' - b)  =  a* - B^+ b.

    Only the FIRST action of the chunk is corrected: under the deadbeat property one step absorbs the whole
    offset, after which the arm is back on the reference and the remaining actions are unchanged.

    Why this is more than augmentation: the image is held fixed while the state varies, so the only way to fit
    the varying label is through the state channel. The policy is therefore driven toward d a / d s = -B^+ --
    exactly the Jacobian the method's deadbeat argument requires. It installs that gain analytically.

    Its limit is equally clear. Real drift moves the image as well as the state; here the pixels do not move,
    so this teaches the state-to-action gain but none of the visual correlates of being off-track. It is the
    linearisation the theory already assumes, not a replacement for on-policy exposure.

    `lo`/`hi` are the normalised bounds of the raw action box, used to keep every generated sample
    unsaturated. `gain` is B^+ * s_std / a_std over the position dims: the action change, in normalised action units, per
    unit of normalised state change (~7-14 here, since B is small). `sigma` is the additive spread and `scale`
    the multiplicative one; both are inverted exactly, so neither teaches the policy anything false.
    """
    # A general affine perturbation of the state, s -> k*s + b, scaling about the normalised mean so the
    # transform does not depend on where the coordinate origin happens to sit. Both parts are admissible for
    # the same reason: the label is AFFINE in s', so the change in the label depends only on how far the state
    # moved,
    #       delta_a = -B^+ delta_s,
    # and never on s_ref. Shift is the k = 1 case. Scaling additionally varies the offset in proportion to the
    # state itself, which spreads the induced drift over magnitudes rather than sampling one scale.
    # The identity delta_a = -B^+ delta_s holds only while the corrected action stays inside the actuation
    # box. Beyond it the real label saturates -- getback_action caps the correction, direction-preserving --
    # and a linear extrapolation would command an action the arm cannot execute. Training on saturated
    # corrective labels is precisely what destroyed the deep-rollout variant, so rather than clip the label
    # afterwards (which would teach the saturated value) we SHRINK the perturbation per sample until the
    # induced correction fits. The relation then remains exact for every sample generated.
    g = gain.to(act.device).view(1, dims)
    B = st.shape[0]
    k = 1.0 + (torch.rand(B, 1, device=st.device) * 2 - 1) * scale  # multiplicative, about the mean
    b = torch.randn(B, dims, device=st.device) * sigma  # additive
    s0 = st[..., :dims]
    ds = (k - 1.0) * s0 + b  # proposed normalised state change

    a0 = act[:, 0, :dims]
    da = -ds * g
    room = torch.where(
        da >= 0, (hi - a0).clamp(min=0), (a0 - lo).clamp(min=0)
    )  # headroom in each direction
    need = da.abs()
    f = torch.where(need > room, room / need.clamp(min=1e-8), torch.ones_like(need))
    f = f.min(dim=-1, keepdim=True).values.clamp(0.0, 1.0)  # one factor per sample, keeps direction
    ds = ds * f

    st = st.clone()
    st[..., :dims] = s0 + ds
    act = act.clone()
    act[:, 0, :dims] = a0 - ds * g
    return st, act


def affine(x, max_rot=8.0, max_scale=0.10, max_shear=0.05, p=0.5):
    """Small per-(sample, camera) rotation / scale / shear.

    A warp is not a true change of viewpoint -- it induces no parallax -- but it does break the assumption that
    a given object always projects to the same image location and orientation, which is the part of viewpoint
    sensitivity that is cheap to attack. Kept small deliberately: large warps produce images the camera could
    never generate and cost more than they buy.
    """
    B, C2, C, H, W = x.shape
    flat = x.reshape(-1, C, H, W)
    n = flat.shape[0]
    take = (torch.rand(n, device=x.device) < p).float()
    r = (torch.rand(n, device=x.device) * 2 - 1) * (max_rot * 3.14159265 / 180.0) * take
    s = 1.0 + (torch.rand(n, device=x.device) * 2 - 1) * max_scale * take
    sh = (torch.rand(n, device=x.device) * 2 - 1) * max_shear * take
    cos, sin = torch.cos(r) / s, torch.sin(r) / s
    theta = torch.zeros(n, 2, 3, device=x.device, dtype=torch.float32)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin + sh
    theta[:, 1, 0] = sin + sh
    theta[:, 1, 1] = cos
    grid = torch.nn.functional.affine_grid(theta, flat.shape, align_corners=False)
    out = torch.nn.functional.grid_sample(
        flat.float(), grid, mode="bilinear", padding_mode="border", align_corners=False
    )
    return out.clamp(0, 1).reshape_as(x).to(x.dtype)
