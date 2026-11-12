"""ODE policy:  da/dt = f(c, s, a).

`a` is the reference the arm should be at (on LIBERO: the end-effector pose, 3 position +
rotation vector relative to CANONICAL_QUAT), `c` the condition encoded from the camera
frames and the task embedding, `s` the measured proprioception. Instead of emitting a
chunk of actions, the policy emits a vector field over `a` and the chunk is whatever that
field integrates to, starting from where the previous chunk ended.

Training never regresses f against a velocity label -- differencing the demo would hand
it the demo's jitter. It integrates f from the current pose over the next n_action_steps
control steps and matches the PATH against the demonstrated one (Chen et al. 2018). The
solver is a fixed-step Euler (or RK2 / RK4) backpropagated through directly rather than
the adjoint: f is a small head on a per-chunk condition, so the unrolled graph is a few
MB, and plain backprop gives the exact gradient where the adjoint re-integrates backwards
and adds its own error. c and s are held fixed over the chunk, as at execution.

Start perturbation: on a fraction of samples a0 is offset by a random amount and the
target becomes demo + offset * rho^k, i.e. "return to the demonstration within a few
steps". That trains the field around the demo, not only along it.

side_head=K > 0 adds the optional diversity head as a plug-in: the field is unchanged
(it outputs the mean velocity, and every one of its parameters gets exactly the same
gradient with or without the head -- the head reads detached features and fits a
detached error). At the chunk start ONE small head outputs an amplitude (a rate per step
and axis) and a K-component Gaussian mixture over a direction eps. The drawn offset is
applied on the velocity, da/dt = mu + rate * eps (side_mode="velocity", default), or on
the position, path = mean path + accumulated amplitude * eps ("position"). With eps = 0
the policy is exactly the plain ODE.

The gripper is near-binary and does not belong in an ODE; a separate head predicts it for
each step of the chunk from the condition alone (one logit per gripper and step).
grip_ode=True is the ablation that integrates it as one more ODE state instead.

Batch `action` layout (see ode_policy/libero/build_ode_zarr.py): [pose(6), gripper(1)]
per step; in general [pose(pose_dim), gripper(n_grippers)].
"""
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ode_policy.policies.base_image_policy import BaseImagePolicy
from ode_policy.utils.normalizer import LinearNormalizer
from ode_policy.utils.pytorch_util import dict_apply
from ode_policy.utils.vision.multi_image_obs_encoder import MultiImageObsEncoder

# keep in sync with ode_policy/libero/pose.py
CANONICAL_QUAT = (0.99959104, -0.00150985, -0.02851977, 0.00144439)
POSE_DIM = 6


def _quat_mul(q, r):
    """xyzw quaternion product q * r."""
    x1, y1, z1, w1 = q.unbind(-1)
    x2, y2, z2, w2 = r.unbind(-1)
    return torch.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dim=-1)


def pose_signal_torch(agent_pos: torch.Tensor) -> torch.Tensor:
    """(..., 8) agent_pos -> (..., 6) pose signal, matching ode_policy/libero/pose.py."""
    pos, quat = agent_pos[..., :3], agent_pos[..., 3:7]
    quat = quat / quat.norm(dim=-1, keepdim=True)
    ref = torch.tensor(CANONICAL_QUAT, dtype=quat.dtype, device=quat.device)
    ref_inv = ref * torch.tensor([-1, -1, -1, 1], dtype=quat.dtype, device=quat.device)
    rel = _quat_mul(quat, ref_inv.expand_as(quat))
    rel = torch.where(rel[..., 3:4] < 0, -rel, rel)
    v, w = rel[..., :3], rel[..., 3:4]
    n = v.norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(n, w)
    # angle / n -> 2 / w as n -> 0
    k = torch.where(n > 1e-8, angle / n.clamp_min(1e-12), 2.0 / w)
    return torch.cat([pos, v * k], dim=-1)


class _FiLMLayer(nn.Module):
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.film = nn.Linear(cond_dim, 2 * dim)

    def forward(self, h, c=None, gb=None):
        g, b = gb if gb is not None else self.film(c).chunk(2, dim=-1)
        return h + F.silu(self.lin(h) * (1 + g) + b)


class VectorField(nn.Module):
    """ODE-MLP: f(a, s | c), a FiLM-modulated residual MLP. Autonomous, no time input --
    the field depends only on where the reference is and what the policy currently sees."""

    def __init__(self, cond_dim, state_dim, hidden_dim, num_layers, out_dim=POSE_DIM,
                 pose_dim=POSE_DIM):
        super().__init__()
        self.inp = nn.Linear(pose_dim + state_dim, hidden_dim)
        self.layers = nn.ModuleList([_FiLMLayer(hidden_dim, cond_dim) for _ in range(num_layers)])
        self.out = nn.Linear(hidden_dim, out_dim)
        self.feat_dim = hidden_dim
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def precompute(self, c):
        """FiLM scale/shift depend on c only; compute them once per integration."""
        return [layer.film(c).chunk(2, dim=-1) for layer in self.layers]

    def forward(self, a, s, c, film=None, return_h=False):
        h = self.inp(torch.cat([a, s], dim=-1))
        for i, layer in enumerate(self.layers):
            h = layer(h, c, None if film is None else film[i])
        return (self.out(h), h) if return_h else self.out(h)


class DiTVectorField(nn.Module):
    """ODE-DiT: the vector field built from fm_dit's DiT blocks (AdaLNBlock, Peebles & Xie 2023).

    [a, s] is projected to n_tokens tokens; every block is an attention + MLP block whose
    LayerNorm scale/shift/gates are regressed from the condition c (adaLN-zero). There is no
    time input: the field is autonomous, so the time embedding of the flow-matching DiT is
    replaced by zeros. The tokens are pooled for the output and for the side head's features.
    Same interface and zero-initialised output as VectorField.
    """

    def __init__(self, cond_dim, state_dim, hidden_dim, num_layers, out_dim, pose_dim=POSE_DIM,
                 num_heads=8, n_tokens=4, mlp_ratio=4.0):
        super().__init__()
        from ode_policy.utils.models.flow_net import AdaLNBlock
        self.n_tokens, self.hidden_dim, self.feat_dim = n_tokens, hidden_dim, hidden_dim
        self.inp = nn.Linear(pose_dim + state_dim, n_tokens * hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, hidden_dim))
        self.cond = nn.Linear(cond_dim, hidden_dim)
        self.blocks = nn.ModuleList([AdaLNBlock(hidden_dim, num_heads, mlp_ratio)
                                     for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def precompute(self, c):
        """The adaLN scales/shifts/gates depend on c only (no time input): compute them once
        per integration instead of at every one of the 4 x n_steps field evaluations."""
        cc = F.silu(self.cond(c))
        return [blk.ada_ln(cc).view(c.shape[0], 6, 1, self.hidden_dim).unbind(1)
                for blk in self.blocks]

    def forward(self, a, s, c, film=None, return_h=False):
        B = a.shape[0]
        x = self.inp(torch.cat([a, s], dim=-1)).view(B, self.n_tokens, self.hidden_dim) + self.pos
        mods = self.precompute(c) if film is None else film
        for blk, (g1, g2, sc1, sc2, sh1, sh2) in zip(self.blocks, mods):
            # AdaLNBlock.forward with t = 0 and its modulation precomputed
            x = x + blk.attn(blk.norm1(x) * (1 + sc1) + sh1) * g1
            x = x + blk.mlp(blk.norm2(x) * (1 + sc2) + sh2) * g2
        h = self.norm(x).mean(dim=1)
        return (self.out(h), h) if return_h else self.out(h)


class ODEImagePolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        obs_as_global_cond=True,
        hidden_dim=512,
        num_layers=4,
        cond_dim=512,
        solver="euler",
        substeps=1,
        # f's raw output is multiplied by this, in normalised pose units per control
        # step. The demo moves ~0.01-0.05 normalised units per step, so this keeps the
        # initial field in range without the net having to learn a tiny output scale.
        vel_scale=0.05,
        perturb_prob=0.5,
        perturb_pos=0.01,     # m (joint_space: a's own units, scalar or one per dim)
        perturb_rot=0.03,     # rad
        perturb_decay=0.7,    # target offset shrinks by this per step
        gripper_loss_weight=0.1,
        # "mlp" (FiLM residual MLP, ODE-MLP) or "dit" (fm_dit's adaLN blocks, ODE-DiT)
        field_arch="mlp",
        dit_hidden=512,
        dit_layers=4,
        dit_heads=8,
        dit_tokens=4,
        # where the proprioception s enters: field_use_s feeds it to f directly;
        # cond_use_s also leaves it in the obs encoder's input (False zeroes it there)
        field_use_s=True,
        cond_use_s=True,
        # diversity head with this many mixture components; 0 = off
        side_head=0,
        side_hidden=256,
        side_mode="velocity",
        # floor on the side head's per-step std, in units of vel_scale
        spread_min=0.02,
        # floor on a mixture component's width
        mix_min=0.05,
        # 6 = end-effector pose (LIBERO). Any other value treats the first pose_dim
        # entries of agent_pos as the pose itself (e.g. 2 for the nav2d agent).
        pose_dim=POSE_DIM,
        # joint_space=True: a is a vector of joint positions (absolute joint targets). The
        # live pose is read straight from agent_pos and the start perturbation is
        # perturb_pos on every dim. Needed because pose_dim == 6 on its own is read as an
        # end-effector pose (pose_signal_torch, 1 cm / 0.03 rad perturbation).
        joint_space=False,
        # gripper columns after the pose in the action (2 for a two-arm robot)
        n_grippers=1,
        # joint_space: which entries of agent_pos are the live pose, in the order of a
        # (None = the first pose_dim)
        state_pose_index=None,
        # the gripper as one more state of the ODE, integrated next to the pose, instead of a
        # classification head. The data ramps each open/close switch over a few steps
        # (build_ode_zarr.py --grip_ramp), so the state is continuous; the command sent stays
        # discrete. grip_vel: its velocity scale (normalised units per step; a flip is 2),
        # grip_ode_weight: its loss weight.
        grip_ode=False,
        grip_vel=1.0,
        grip_ode_weight=1.0,
        # grip_carry: "state" continues the next chunk from the integrated gripper state (it is
        # also the policy's timer through a grasp pause); "sign" snaps it to the command
        grip_carry="state",
        # gripper command value meaning "open" (LIBERO: -1) -- the state at an episode start
        grip_open=-1.0,
        # command = closed once the state leaves the open side past this value, open again only
        # when it returns; None = 0.8 * grip_open (switch as soon as a ramp starts closing)
        grip_thresh=None,
        **kwargs,
    ):
        super().__init__()
        self.pose_dim = pose_dim
        self.joint_space = joint_space
        self.n_grippers = int(n_grippers)
        self.grip_ode, self.grip_vel, self.grip_ode_weight = grip_ode, grip_vel, grip_ode_weight
        assert grip_carry in ("state", "sign"), grip_carry
        self.grip_carry, self.grip_open = grip_carry, float(grip_open)
        self.grip_thresh = 0.8 * self.grip_open if grip_thresh is None else float(grip_thresh)
        assert not grip_ode or (self.n_grippers == 1 and side_mode == "velocity"), \
            "grip_ode: one gripper, velocity-mode side head only"
        fp = pose_dim + int(grip_ode)   # field state: pose (+ gripper)
        assert shape_meta["action"]["shape"][0] == pose_dim + self.n_grippers, \
            "ODE policy expects the pose zarr from ode_policy/libero/build_ode_zarr.py: action = [pose, grippers]"
        state_dim = shape_meta["obs"]["agent_pos"]["shape"][0]
        assert horizon - n_obs_steps >= n_action_steps, "window too short for the rollout"
        if state_pose_index is not None:
            state_pose_index = [int(i) for i in state_pose_index]
            assert joint_space and len(state_pose_index) == pose_dim and 0 <= min(state_pose_index) \
                and max(state_pose_index) < state_dim, \
                f"state_pose_index {state_pose_index}: needs joint_space and {pose_dim} indices into agent_pos"
        self.state_pose_index = state_pose_index
        self.field_use_s = field_use_s
        self.cond_use_s = cond_use_s
        field_state_dim = state_dim if field_use_s else 0
        obs_feature_dim = obs_encoder.output_shape()[0]

        self.obs_encoder = obs_encoder
        self.cond_net = nn.Sequential(
            nn.Linear(obs_feature_dim * n_obs_steps, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        if field_arch == "dit":
            self.field = DiTVectorField(cond_dim, field_state_dim, dit_hidden, dit_layers,
                                        fp, fp, num_heads=dit_heads, n_tokens=dit_tokens)
        else:
            assert field_arch == "mlp", field_arch
            self.field = VectorField(cond_dim, field_state_dim, hidden_dim, num_layers, fp, fp)
        assert side_mode in ("position", "velocity"), side_mode
        self.side_k, self.side_mode = side_head, side_mode
        if side_head:
            # [n_action_steps * P amplitude rates, K logits, K * P means, K * P raw widths]
            n_amp = n_action_steps * pose_dim
            self.side_head = nn.Sequential(
                nn.Linear(self.field.feat_dim, side_hidden), nn.SiLU(),
                nn.Linear(side_hidden, n_amp + side_head * (1 + 2 * pose_dim)))
            nn.init.zeros_(self.side_head[-1].weight)
            nn.init.zeros_(self.side_head[-1].bias)
            nn.init.normal_(self.side_head[-1].bias[n_amp + side_head:
                                                    n_amp + side_head * (1 + pose_dim)])
        self.spread_min, self.mix_min = spread_min, mix_min
        self.grip_head = nn.Sequential(
            nn.Linear(cond_dim, cond_dim), nn.SiLU(), nn.Linear(cond_dim, n_action_steps * self.n_grippers))
        self.normalizer = LinearNormalizer()

        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.state_dim = state_dim
        assert solver in ("euler", "rk2", "rk4"), solver
        self.solver = solver
        self.substeps = substeps
        self.vel_scale = vel_scale
        self.perturb_prob = perturb_prob
        if pose_dim == POSE_DIM and not joint_space:
            self.perturb_std = [perturb_pos] * 3 + [perturb_rot] * 3
        else:
            # one std per dim of a, in a's own units; a scalar is broadcast
            p = np.asarray(perturb_pos, dtype=np.float64).ravel()
            assert p.size in (1, pose_dim), f"perturb_pos: expected 1 or {pose_dim} values, got {p.size}"
            self.perturb_std = (p.repeat(pose_dim) if p.size == 1 else p).tolist()
        self.perturb_decay = perturb_decay
        self.gripper_loss_weight = gripper_loss_weight
        self._g_carry = None      # grip_ode: the gripper state carried between chunks
        self.kwargs = kwargs

    def reset(self):
        self._g_carry = None

    # ========= core ============
    def _condition(self, nobs):
        To = self.n_obs_steps
        B = next(iter(nobs.values())).shape[0]
        this = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        if not self.cond_use_s:
            this = dict(this, agent_pos=torch.zeros_like(this["agent_pos"]))
        feat = self.obs_encoder(this).reshape(B, -1)
        c = self.cond_net(feat)
        s = self._field_s(nobs["agent_pos"][:, To - 1])
        return c, s

    def _field_s(self, s):
        return s if self.field_use_s else s[..., :0]

    def _f(self, a, s, c, film=None):
        v = self.vel_scale * (self.field(a, s, c) if film is None else self.field(a, s, c, film))
        if self.grip_ode:
            P = self.pose_dim
            v = torch.cat([v[..., :P], v[..., P:] * (self.grip_vel / self.vel_scale)], dim=-1)
        return v

    def integrate(self, a0, s, c, n_steps, step_offset=None, rates=None, substeps=None,
                  emit_substeps=False):
        """Integrate from a0 over n_steps control steps (time unit = one step).
        Returns (B, n_steps, pose_dim): the pose after each step, normalised.
        step_offset (B, n_steps, pose_dim): a velocity added during each control step.
        rates: a list; if given, the field's own velocity at the start of each control
        step, f(a_i) (with step_offset), (B, pose_dim) normalised per step, is appended.
        substeps: integration steps per control step (None = the trained self.substeps);
        emit_substeps=True returns the state after every substep, (B, n_steps * substeps,
        pose_dim), i.e. the same trajectory sampled substeps times as finely."""
        substeps = substeps or self.substeps
        h = 1.0 / substeps
        P = self.pose_dim
        Pf = P + int(self.grip_ode)   # grip_ode: a0 and the result carry the gripper last
        a, out = a0, []
        # c is fixed across the rollout: hoist the field's FiLM / adaLN projections out of
        # the 4 * n_steps field evaluations (exact; saves most of the head's kernel launches)
        film = self.field.precompute(c)
        for i in range(n_steps):
            f = lambda x: self._f(x, s, c, film)
            if step_offset is not None:
                off = step_offset[:, i]
                if Pf > P:   # the side head offsets the pose only
                    off = torch.cat([off, torch.zeros_like(off[..., :Pf - P])], dim=-1)
                f = lambda x, f=f, off=off: f(x) + off
            if rates is not None:
                rates.append(f(a)[..., :P])
            for _ in range(substeps):
                if self.solver == "euler":
                    a = a + h * f(a)
                elif self.solver == "rk2":      # explicit midpoint
                    a = a + h * f(a + 0.5 * h * f(a))
                else:
                    k1 = f(a)
                    k2 = f(a + 0.5 * h * k1)
                    k3 = f(a + 0.5 * h * k2)
                    k4 = f(a + h * k3)
                    a = a + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                if emit_substeps:
                    out.append(a)
            if not emit_substeps:
                out.append(a)
        return torch.stack(out, dim=1)

    def _pose_norm(self):
        p = self.normalizer["action"].params_dict
        return p["scale"][:self.pose_dim], p["offset"][:self.pose_dim]

    def _grip_norm(self):
        p = self.normalizer["action"].params_dict
        return p["scale"][self.pose_dim], p["offset"][self.pose_dim]

    def _split_grip(self, traj):
        """grip_ode: (pose, gripper state) of an integrated path; otherwise (path, None)."""
        return (traj[..., :self.pose_dim], traj[..., self.pose_dim]) if self.grip_ode else (traj, None)

    def _live_pose(self, agent_pos):
        """Measured pose in the policy's pose coordinates."""
        if self.pose_dim == POSE_DIM and not self.joint_space:
            return pose_signal_torch(agent_pos)
        if self.state_pose_index is not None:
            return agent_pos[..., self.state_pose_index]
        return agent_pos[..., :self.pose_dim]

    def _grip_logits(self, c):
        """(B, n_action_steps, n_grippers) gripper logits from the condition."""
        return self.grip_head(c).view(c.shape[0], self.n_action_steps, self.n_grippers)

    def _start_feat(self, a0, s, c):
        """What the side head is conditioned on: the field's trunk features at the chunk
        start, detached so the head never changes the field."""
        return self.field(a0, s, c, return_h=True)[1].detach()

    def _side(self, a0, s, c):
        """Side head at a chunk start: amplitude rates (B, n_steps, P) >= 0 in normalised
        pose units per step, and the eps mixture (log weights (B, K), means, widths (B, K, P))."""
        K, P, n = self.side_k, self.pose_dim, self.n_action_steps
        out = self.side_head(self._start_feat(a0, s, c))
        rate = self.vel_scale * F.softplus(out[:, :n * P]).view(-1, n, P)
        mean, raw = out[:, n * P + K:].view(-1, 2, K, P).unbind(dim=1)
        # softplus(0.54) = 1: widths start at the unit scale of eps
        return (rate, F.log_softmax(out[:, n * P:n * P + K], dim=-1), mean,
                self.mix_min + F.softplus(raw + 0.54))

    @staticmethod
    def _draw_eps(logw, mean, width, center=False):
        """Draw a component by weight, then eps inside it; center=True takes the
        component's mean instead (which mode is still random, the spread inside it is not)."""
        w = logw.exp()
        bad = ~torch.isfinite(w).all(dim=-1)
        if bad.any():
            # diverged rollout (non-finite features): draw anything, the non-finite mean makes
            # the integrated reference non-finite and the evaluator fails the episode
            w = torch.where(bad[:, None], torch.ones_like(w), w)
        k = torch.multinomial(w, 1)[:, 0]
        i = torch.arange(len(k), device=k.device)
        return mean[i, k] if center else mean[i, k] + width[i, k] * torch.randn_like(mean[:, 0])

    # ========= inference ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor], a0=None, eps=None,
                       offset_mode=None, eps_center=False, return_rate=False, n_steps=None,
                       substeps=None, emit_substeps=False) -> Dict[str, torch.Tensor]:
        """`action` rows are [pose after step i+1, grippers].

        a0: (B, pose_dim) un-normalised pose to integrate from. None starts from the LIVE
        pose; passing the pose the previous integration reached instead keeps a as a state
        of its own, so the reference can run ahead of a lagging arm (carried execution).
        eps: (B, pose_dim) direction for the side head this chunk; None draws one from the
        learned mixture, zeros give the plain ODE.
        offset_mode: side head only. "velocity" feeds eps into the integration, "position"
        adds amplitude * eps to the mean path. None = the policy's side_mode.
        eps_center: when eps is drawn here, take the drawn component's mean, without the
        noise inside the component (the default for single-mode tasks such as LIBERO).
        return_rate: adds `rate` (B, n, pose_dim), the field's own velocity f(a_i) at the
        start of each step, in pose units per control step.
        n_steps / substeps / emit_substeps: integrate n_steps control steps (None =
        n_action_steps; beyond the trained length the side head's last step amplitude and
        the last gripper logit are held), substeps integration steps per control step, and
        with emit_substeps return every substep (n_steps * substeps rows), i.e. actions at
        substeps times the control rate.
        """
        nobs = self._normalize_obs(obs_dict)
        c, s = self._condition(nobs)
        scale, offset = self._pose_norm()
        if a0 is None:
            a0 = self._live_pose(obs_dict["agent_pos"][:, self.n_obs_steps - 1])
        a0 = a0.to(scale.dtype) * scale + offset
        if self.grip_ode:
            # the gripper state continues from the previous chunk; open at an episode start
            gs, go = self._grip_norm()
            g0 = self._g_carry if self._g_carry is not None else (self.grip_open * gs + go).view(1, 1)
            a0 = torch.cat([a0, g0.to(a0.dtype).expand(a0.shape[0], 1)], dim=-1)
        extra, rates = {}, ([] if return_rate else None)
        n = n_steps or self.n_action_steps
        ikw = dict(substeps=substeps, emit_substeps=emit_substeps)

        def fit(x):   # (B, m, ...) per control step -> (B, n, ...), holding the last step
            if x.shape[1] >= n:
                return x[:, :n]
            return torch.cat([x, x[:, -1:].expand(-1, n - x.shape[1], *x.shape[2:])], dim=1)
        if self.side_k:
            rate, logw, mean, width = self._side(a0, s, c)
            if eps is None:
                eps = self._draw_eps(logw, mean, width, eps_center)
            eps, sig = eps.to(a0.dtype), rate.cumsum(dim=1)
            if (offset_mode or self.side_mode) == "position":
                traj = self.integrate(a0, s, c, self.n_action_steps) + sig * eps[:, None]
            else:
                traj = self.integrate(a0, s, c, n, step_offset=fit(rate) * eps[:, None],
                                      rates=rates, **ikw)
            traj, gtraj = self._split_grip(traj)
            traj = (traj - offset) / scale
            extra.update(eps=eps, spread=sig / scale, mix_prob=logw.exp(), mix_mean=mean,
                         mix_width=width)
        else:
            traj, gtraj = self._split_grip(self.integrate(a0, s, c, n, rates=rates, **ikw))
            traj = (traj - offset) / scale
        if return_rate:
            assert rates, "return_rate: plain or velocity-mode side-head policies only"
            extra["rate"] = torch.stack(rates, dim=1) / scale
        if self.grip_ode:
            gs, go = self._grip_norm()
            extra["grip_state"] = (gtraj - go) / gs          # raw command units
            if self.grip_carry == "state":
                self._g_carry = gtraj[:, -1:].detach()
            else:                                             # snap to the command
                g_end = extra["grip_state"][:, -1:]
                self._g_carry = torch.where(g_end >= 0, 1.0, -1.0) * gs + go
            # discrete command: open while the state stays on the open side of grip_thresh,
            # closed once it has moved past it (a closing ramp starts the close at once)
            past = (extra["grip_state"] - self.grip_thresh) * (1.0 if self.grip_open < 0 else -1.0)
            glogit = torch.where(past >= 0, -self.grip_open, self.grip_open)[..., None]
        else:
            glogit = self._grip_logits(c)
            if n != glogit.shape[1] or emit_substeps:
                glogit = fit(glogit)
                if emit_substeps:
                    glogit = glogit.repeat_interleave(substeps or self.substeps, dim=1)
        grip = torch.where(glogit >= 0, 1.0, -1.0)
        action = torch.cat([traj, grip], dim=-1)
        return {"action": action, "action_pred": action, **extra}

    # ========= training ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        nobs = self.normalizer.normalize(batch["obs"])
        nact = self.normalizer["action"].normalize(batch["action"])
        To, Ta = self.n_obs_steps, self.n_action_steps
        c, s = self._condition(nobs)

        a0 = nact[:, To - 1, :self.pose_dim]
        target = nact[:, To:To + Ta, :self.pose_dim]

        if self.training and self.perturb_prob > 0:
            scale, _ = self._pose_norm()
            std = torch.tensor(self.perturb_std, device=a0.device, dtype=a0.dtype) * scale
            on = (torch.rand(a0.shape[0], 1, device=a0.device) < self.perturb_prob).to(a0.dtype)
            delta = torch.randn_like(a0) * std * on
            decay = self.perturb_decay ** torch.arange(
                1, Ta + 1, device=a0.device, dtype=a0.dtype)
            a0 = a0 + delta
            target = target + delta[:, None, :] * decay[None, :, None]
        if self.grip_ode:
            # gripper column: row t = the command at t, so step i of the chunk should reach
            # row To-1+i's command, starting from the previous one (as carried at deployment)
            assert To >= 2, "grip_ode needs the previous command (n_obs_steps >= 2)"
            a0 = torch.cat([a0, nact[:, To - 2, self.pose_dim:self.pose_dim + 1]], dim=-1)

        traj, gtraj = self._split_grip(self.integrate(a0, s, c, Ta))
        # in units of one typical step of motion, so a step-sized miss costs ~1 and the
        # gripper term does not swamp it (raw normalised errors are ~1e-4)
        pose_loss = F.mse_loss(traj, target) / self.vel_scale ** 2

        if self.grip_ode:
            # the integrated gripper state against the commands, in normalised units
            grip_loss = F.mse_loss(gtraj, nact[:, To - 1:To - 1 + Ta, self.pose_dim]) * (
                self.grip_ode_weight / self.gripper_loss_weight)
        else:
            grip_tgt = (batch["action"][:, To - 1:To - 1 + Ta, self.pose_dim:self.pose_dim + self.n_grippers]
                        > 0).to(traj.dtype)
            grip_loss = F.binary_cross_entropy_with_logits(self._grip_logits(c), grip_tgt)
        side_loss = 0.0
        if self.side_k:
            # the side head fits the plain ODE's own (detached) path error: its size
            # (Gaussian NLL under the accumulated amplitude) and, in units of that
            # amplitude, its shape (mixture NLL of the demo's eps)
            rate, logw, mean, width = self._side(a0, s, c)
            sig = rate.cumsum(dim=1) / self.vel_scale + self.spread_min
            err = ((traj - target) / self.vel_scale).detach()
            eps1 = -(sig.detach() * err).sum(dim=1) / (sig.detach() ** 2).sum(dim=1)
            logp = (-0.5 * ((eps1[:, None] - mean) / width) ** 2 - width.log()).sum(dim=-1)
            side_loss = ((0.5 * (err / sig) ** 2 + sig.log()).mean()
                         - torch.logsumexp(logw + logp, dim=1).mean())
        return pose_loss + self.gripper_loss_weight * grip_loss + side_loss
