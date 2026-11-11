"""LIBERO virtual references and a lightweight control-step ODE policy head.

ImageWAM's raw state is xyz + absolute axis-angle + two gripper joint positions.
Euler uses h=1 per control index, not one second of physical time.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CANONICAL_QUAT = (0.99959104, -0.00150985, -0.02851977, 0.00144439)
POS_SCALE = 0.05
ROT_SCALE = 0.5


def quat_mul(a, b):
    a, b = torch.broadcast_tensors(a, b)
    av, aw = a[..., :3], a[..., 3:]
    bv, bw = b[..., :3], b[..., 3:]
    return torch.cat((aw * bv + bw * av + torch.cross(av, bv, dim=-1),
                      aw * bw - (av * bv).sum(-1, keepdim=True)), -1)


def quat_inv(q):
    return torch.cat((-q[..., :3], q[..., 3:]), -1)


def rotvec_to_quat(r):
    angle = r.norm(dim=-1, keepdim=True)
    # sinc avoids the removable singularity, including its backward at zero.
    return torch.cat((r * (0.5 * torch.sinc(angle / (2 * torch.pi))),
                      torch.cos(angle * 0.5)), -1)


def quat_to_rotvec(q):
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q = torch.where(q[..., 3:] < 0, -q, q)
    n = q[..., :3].norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(n, q[..., 3:])
    factor = torch.where(n > 1e-7, angle / n.clamp_min(1e-12),
                         2 / q[..., 3:].clamp_min(1e-12))
    return q[..., :3] * factor


def canonical(q, convention=None):
    ref = q.new_tensor(CANONICAL_QUAT if convention is None else convention["canonical_quat"])
    return ref / ref.norm()


def state_to_reference(state, convention=None):
    q = rotvec_to_quat(state[..., 3:6])
    return torch.cat((state[..., :3], quat_to_rotvec(quat_mul(q, quat_inv(canonical(q, convention))))), -1)


def command_to_reference(state, command, convention=None):
    pos_scale = POS_SCALE if convention is None else float(convention["position_scale"])
    rot_scale = ROT_SCALE if convention is None else float(convention["rotation_scale"])
    q_now = rotvec_to_quat(state[..., 3:6])
    q_target = quat_mul(rotvec_to_quat(command[..., 3:6] * rot_scale), q_now)
    return torch.cat((state[..., :3] + pos_scale * command[..., :3],
                      quat_to_rotvec(quat_mul(q_target, quat_inv(canonical(q_target, convention))))), -1)


def reference_to_command(reference, state, convention=None):
    pos_scale = POS_SCALE if convention is None else float(convention["position_scale"])
    rot_scale = ROT_SCALE if convention is None else float(convention["rotation_scale"])
    q_relative = rotvec_to_quat(reference[..., 3:6])
    q_target = quat_mul(q_relative, canonical(q_relative, convention))
    q_now = rotvec_to_quat(state[..., 3:6])
    return torch.cat(((reference[..., :3] - state[..., :3]) / pos_scale,
                      quat_to_rotvec(quat_mul(q_target, quat_inv(q_now))) / rot_scale), -1)


def linear32(x, layer):
    return F.linear(x.float(), layer.weight.float(),
                    None if layer.bias is None else layer.bias.float())


class FiLMLayer(nn.Module):
    def __init__(self, width, cond_dim):
        super().__init__()
        self.lin = nn.Linear(width, width)
        self.film = nn.Linear(cond_dim, 2 * width)

    def forward(self, x, modulation):
        g, b = modulation
        return x + F.silu(linear32(x, self.lin) * (1 + g) + b)


class ActionODEHead(nn.Module):
    def __init__(self, feature_dim=1024, cond_dim=512, width=512, layers=4,
                 horizon=12, vel_scale=0.05):
        super().__init__()
        if min(feature_dim, cond_dim, width, layers, horizon) < 1 or vel_scale <= 0:
            raise ValueError("Head dimensions, horizon and velocity scale must be positive")
        self.horizon = int(horizon)
        self.vel_scale = float(vel_scale)
        self.context_norm = nn.LayerNorm(feature_dim)
        self.context_proj = nn.Linear(feature_dim, cond_dim)
        self.inp = nn.Linear(6 + 8, width)
        self.layers = nn.ModuleList([FiLMLayer(width, cond_dim) for _ in range(layers)])
        self.out = nn.Linear(width, 6)
        self.grip_in = nn.Linear(cond_dim, cond_dim)
        self.grip_out = nn.Linear(cond_dim, horizon)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def condition(self, feature):
        z = F.layer_norm(feature.float(), self.context_norm.normalized_shape,
                         self.context_norm.weight.float(), self.context_norm.bias.float())
        return linear32(z, self.context_proj)

    def field(self, reference, state, modulations):
        h = linear32(torch.cat((reference.float(), state.float()), -1), self.inp)
        for layer, gb in zip(self.layers, modulations):
            h = layer(h, gb)
        return self.vel_scale * linear32(h, self.out)

    def forward(self, feature, state, reference, horizon=None):
        horizon = self.horizon if horizon is None else int(horizon)
        if not 1 <= horizon <= self.horizon:
            raise ValueError(f"ODE horizon must be in [1,{self.horizon}], got {horizon}")
        # FP32 accumulation even when the backbone / DeepSpeed parameters are bf16.
        with torch.autocast(device_type=feature.device.type, enabled=False):
            c = self.condition(feature)
            modulations = [linear32(c, layer.film).chunk(2, -1) for layer in self.layers]
            r = reference.float()
            trajectory = []
            for _ in range(horizon):
                r = r + self.field(r, state, modulations)  # h=1 control step, Euler
                trajectory.append(r)
            grip = linear32(F.silu(linear32(c, self.grip_in)), self.grip_out)[:, :horizon]
            return torch.stack(trajectory, 1), grip
