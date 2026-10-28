"""ACT on the shared observation encoder used by every method in the paper.

Stock ACT sees the scene through its own ImageNet-pretrained ResNet18, one frame, and gets the task only as a vector
appended to qpos. For a fair comparison every method uses the same encoder: ResNet18 with GroupNorm, randomly
initialised, 8 frames of image history plus the CLIP task embedding. This variant keeps ACT's action model -- the
CVAE over action chunks, the DETR encoder-decoder with one learned query per chunk step, L1 + KL loss, z = 0 (the
prior mean) at inference -- and changes only how observations reach it:

* `obs_encoder` is the MultiImageObsEncoder every other policy uses (same config block as fm_dit.yaml), applied to
  each of the n_obs_steps frames. Its per-frame feature covers the camera, agent_pos and task_emb keys.
* Each frame's feature becomes one transformer token (linear projection + learned position embedding), in place of
  the backbone's spatial feature map. The latent and proprio tokens stay as in ACT.
* The CVAE encoder still reads [CLS, qpos (+ task_emb) at the current step, action chunk], as in ACT.
* dec_layers defaults to 1. Upstream ACT only ever reads decoder layer 0 (building it with dec_layers=1 reproduces
  the 7-layer output bit for bit), so this is the same decoder without six layers of dead weights, which would
  otherwise count toward the parameter budget.

The ~24M parameters outside obs_encoder.* and normalizer.* (matched to the other methods) are reached through
dim_feedforward in act.yaml.
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ode_policy.policies.act.detr.models.detr_vae import (
    build_encoder,
    get_sinusoid_encoding_table,
    reparametrize,
)
from ode_policy.policies.act.detr.models.transformer import build_transformer
from ode_policy.policies.base_image_policy import BaseImagePolicy
from ode_policy.utils.normalizer import LinearNormalizer
from ode_policy.utils.pytorch_util import dict_apply
from ode_policy.utils.vision.multi_image_obs_encoder import MultiImageObsEncoder


class DETRVAESharedObs(nn.Module):
    """DETRVAE (detr/models/detr_vae.py) with observation tokens in place of a CNN backbone.

    Same parameters and forward as DETRVAE's image branch, except that the transformer memory is
    [latent, proprio, obs_1 .. obs_T] built from per-frame features instead of a flattened feature map.
    """

    def __init__(self, transformer, encoder, state_dim, action_dim, num_queries, obs_feature_dim, n_obs_steps):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj_obs = nn.Linear(obs_feature_dim, hidden_dim)       # replaces input_proj (Conv2d on features)
        self.obs_pos_embed = nn.Embedding(n_obs_steps, hidden_dim)         # replaces the 2D sine position embedding
        self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        # CVAE encoder side, as in DETRVAE
        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        self.register_buffer("pos_table", get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim))
        # decoder side, as in DETRVAE
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)            # latent, proprio

    def forward(self, qpos, obs_features, actions=None, is_pad=None, disable_vae=False):
        """
        qpos: (B, state_dim) at the current step
        obs_features: (B, T_obs, obs_feature_dim) from the shared obs encoder
        actions: (B, num_queries, action_dim) in training, None at inference (z = prior mean)
        """
        bs = qpos.shape[0]
        if actions is not None and not disable_vae:
            action_embed = self.encoder_action_proj(actions)                           # (B, Q, D)
            qpos_embed = self.encoder_joint_proj(qpos).unsqueeze(1)                    # (B, 1, D)
            cls_embed = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)            # (B, 1, D)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], dim=1).permute(1, 0, 2)
            cls_joint_is_pad = torch.full((bs, 2), False, device=qpos.device)
            is_pad = torch.cat([cls_joint_is_pad, is_pad], dim=1)
            pos_embed = self.pos_table.clone().detach().permute(1, 0, 2)
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)[0]
            latent_info = self.latent_proj(encoder_output)
            mu, logvar = latent_info[:, : self.latent_dim], latent_info[:, self.latent_dim:]
            latent_input = self.latent_out_proj(reparametrize(mu, logvar))
        else:
            mu = logvar = None
            latent_input = self.latent_out_proj(torch.zeros(bs, self.latent_dim, device=qpos.device))

        proprio_input = self.input_proj_robot_state(qpos)
        obs_tokens = self.input_proj_obs(obs_features)                                 # (B, T, D)
        src = torch.cat([latent_input.unsqueeze(1), proprio_input.unsqueeze(1), obs_tokens], dim=1)
        pos = torch.cat([self.additional_pos_embed.weight, self.obs_pos_embed.weight[: obs_tokens.shape[1]]], dim=0)
        hs = self.transformer(src, None, self.query_embed.weight, pos)[0]               # decoder layer 0, as DETRVAE
        return self.action_head(hs), self.is_pad_head(hs), [mu, logvar]


class ActSharedEncoderImagePolicy(BaseImagePolicy):

    def __init__(
        self,
        shape_meta: Mapping[str, Any],
        obs_encoder: MultiImageObsEncoder,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        hidden_dim: int = 512,
        dim_feedforward: int = 3200,
        nheads: int = 8,
        enc_layers: int = 4,
        dec_layers: int = 1,
        dropout: float = 0.1,
        pre_norm: bool = False,
        kl_weight: float = 10.0,
        disable_vae: bool = False,
        use_task_emb: bool = True,
    ):
        super().__init__()
        obs_meta = shape_meta["obs"]
        self.state_key = "agent_pos"
        if self.state_key not in obs_meta:
            raise ValueError(f"ActSharedEncoderImagePolicy needs an 'agent_pos' obs key; shape_meta has {sorted(obs_meta)}.")
        action_dim = int(shape_meta["action"]["shape"][0])
        qpos_dim = int(obs_meta[self.state_key]["shape"][0])
        self.use_task_emb = bool(use_task_emb) and "task_emb" in obs_meta
        if self.use_task_emb:
            qpos_dim += int(obs_meta["task_emb"]["shape"][0])

        # The DETR builders read an argparse Namespace (as in upstream ACT).
        args = argparse.Namespace(hidden_dim=hidden_dim, dim_feedforward=dim_feedforward, nheads=nheads,
                                  enc_layers=enc_layers, dec_layers=dec_layers, dropout=dropout, pre_norm=pre_norm)
        self.obs_encoder = obs_encoder
        self.model = DETRVAESharedObs(
            build_transformer(args), build_encoder(args), state_dim=qpos_dim, action_dim=action_dim,
            num_queries=horizon, obs_feature_dim=obs_encoder.output_shape()[0], n_obs_steps=n_obs_steps,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self.kl_weight = float(kl_weight)
        self.disable_vae = bool(disable_vae)

    # ── observation plumbing ────────────────────────────────────────────────

    def _obs_features(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """-> (B, n_obs_steps, obs_feature_dim): the shared encoder on every history frame, as fm_dit does."""
        To = self.n_obs_steps
        B = next(iter(nobs.values())).shape[0]
        this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
        return self.obs_encoder(this_nobs).reshape(B, To, -1)

    def _qpos(self, nobs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """-> (B, qpos_dim) at the current (last) obs step, with the task embedding appended, as in ACT."""
        t = self.n_obs_steps - 1
        q = nobs[self.state_key][:, t]
        if self.use_task_emb:
            q = torch.cat([q, nobs["task_emb"][:, t]], dim=-1)
        return q

    # ── training ────────────────────────────────────────────────────────────

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])[:, : self.horizon]
        B, T, _ = nactions.shape
        # The shared dataset replicate-pads at episode boundaries, so nothing in a batch is padding (see ACT).
        is_pad = torch.zeros((B, T), dtype=torch.bool, device=nactions.device)
        a_hat, _, (mu, logvar) = self.model(self._qpos(nobs), self._obs_features(nobs), nactions, is_pad,
                                            disable_vae=self.disable_vae)
        l1 = F.l1_loss(nactions, a_hat)
        loss = l1
        metrics = {"l1_loss": l1.item()}
        if mu is not None:
            kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(dim=1).mean()
            loss = loss + self.kl_weight * kl
            metrics["kl_loss"] = kl.item()
        self._last_metrics = metrics
        return loss

    # ── inference ───────────────────────────────────────────────────────────

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self._normalize_obs(obs_dict)
        a_hat, _, _ = self.model(self._qpos(nobs), self._obs_features(nobs), None)   # z = 0: deterministic
        action_pred = self.normalizer["action"].unnormalize(a_hat)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        return {"action": action_pred[:, start:end], "action_pred": action_pred}

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.predict_action(obs_dict)
