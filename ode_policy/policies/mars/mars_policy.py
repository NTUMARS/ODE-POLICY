"""
MARS Policy: End-to-End Learned Router for Flow Starting Point.

A learned router network outputs a **per-action-dimension** weight vector
w ∈ [0,1]^D that independently controls each action dimension's
history/noise mixture for the flow starting point:

  w_d ≈ 0 (this dim is unimodal)   → start_d ≈ history_d, no noise
  w_d ≈ 1 (this dim is multimodal) → start_d ≈ pure noise

Per-dim weighting is critical for high-D action spaces (e.g. 9-D Franka joints)
where the bimodal signal lives in only a few dimensions. With a scalar w, the
flow noise grows as √D and drowns the signal at unused dims; per-dim w lets the
router zero out noise on those dims, decoupling the signal/noise ratio from D.

Key learning signal: **diversity loss** — within a mini-batch, finds samples with
similar histories and checks whether starts cover the diversity of their
targets. If targets are diverse but starts are not (because w≈0 forces
similar starts), the loss penalizes this, pushing the relevant w_d up.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ode_policy.utils.normalizer import LinearNormalizer
from ode_policy.utils.pytorch_util import dict_apply
from ode_policy.policies.base_image_policy import BaseImagePolicy

from ode_policy.utils.models.flow_net import FlowTransformer
from ode_policy.utils.vision.multi_image_obs_encoder import MultiImageObsEncoder
from ode_policy.utils.flow.flow_matchers import TorchFlowMatcher
from ode_policy.utils.history import assert_history_compatible


class MarsImagePolicy(BaseImagePolicy):

    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        flow_matcher: TorchFlowMatcher,
        # DiT architecture params
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
        diffusion_step_embed_dim=256,
        latent_dim=512,
        # Loss weights
        flow_loss_weight=1.0,
        consistency_weight=1.0,
        # Router params
        router_entropy_weight=0.01,
        # Diversity loss params
        diversity_weight=1.0,
        diversity_k=5,
        # Adaptive sampling steps (inference only)
        adaptive_steps=True,
        adaptive_max_steps=10,
        # History source for flow start-point ("agent_pos" reuses state as a
        # surrogate — fine when state_dim == action_dim, e.g., metasim joint_pos
        # setups; "past_action" reads the real stored past actions and requires
        # batch["action"] (train) / obs_dict["past_action"] (inference) to
        # carry the right dim. Default "agent_pos" preserves existing behavior.
        history_source: str = "agent_pos",
        # Action dims whose flow source is ALWAYS Gaussian noise, never the
        # history. Use for channels the A2A premise does not fit: LIBERO's
        # gripper (dim 6) is a near-binary open/close signal, so its "history"
        # is a step function and transporting it from a past value carries no
        # useful structure. Forcing the router weight to 1 there makes the start
        # pure noise AND zeroes the consistency gate (1-w), so the deterministic
        # precision term stops penalising a channel that is inherently discrete.
        noise_dims=None,
        **kwargs,
    ):
        super().__init__()

        # Catch yaml typos: any unrecognized kwarg means a misnamed config key
        # (e.g. `consistency_weighht` instead of `consistency_weight`) was
        # silently swallowed and the default ran instead. Use TypeError (not
        # assert) so it's not stripped under `python -O`.
        if kwargs:
            raise TypeError(
                f"MarsImagePolicy got unexpected kwargs: "
                f"{sorted(kwargs.keys())}. Likely a typo in the policy yaml "
                f"— check against the constructor signature."
            )

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]

        assert n_obs_steps == n_action_steps, (
            f"MARS Policy requires n_obs_steps == n_action_steps, "
            f"got {n_obs_steps} vs {n_action_steps}"
        )

        # Validate history_source against shape_meta. When history_source
        # is "agent_pos", state is reused as a history-action surrogate and
        # must therefore have the same dim as the action. Fails loud here
        # so mis-configured tasks (e.g. LIBERO pose-state vs delta-action)
        # can't silently train with a miscalibrated flow start.
        state_shape = shape_meta.get("obs", {}).get("agent_pos", {}).get("shape")
        if state_shape is not None:
            assert_history_compatible(
                history_source, state_shape[0], action_dim,
                owner="MarsImagePolicy",
            )

        self.noise_dims = tuple(noise_dims) if noise_dims else ()
        if self.noise_dims:
            bad = [d for d in self.noise_dims if not (0 <= d < action_dim)]
            if bad:
                raise ValueError(f"noise_dims {bad} out of range for action_dim={action_dim}")
            if len(set(self.noise_dims)) >= action_dim:
                raise ValueError(
                    "noise_dims covers every action dim — the router would have "
                    "nothing left to control."
                )
            mask = torch.zeros(1, action_dim, dtype=torch.bool)
            mask[0, list(self.noise_dims)] = True
            self.register_buffer("_noise_dim_mask", mask, persistent=False)
        else:
            self._noise_dim_mask = None

        # Obs encoder
        self.obs_encoder = obs_encoder
        global_cond_dim = obs_feature_dim * n_obs_steps

        # DiT flow network
        self.flow_net = FlowTransformer(
            input_dim=action_dim,
            condition_dim=global_cond_dim,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            time_embed_dim=diffusion_step_embed_dim,
        )

        # History encoder
        self.history_encoder = nn.Sequential(
            nn.Linear(action_dim * n_obs_steps, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # ─── Router network (end-to-end, no score supervision) ───────────
        router_hidden = 256
        self.router_obs_proj = nn.Sequential(
            nn.LayerNorm(global_cond_dim),
            nn.Linear(global_cond_dim, router_hidden),
            nn.GELU(),
        )
        self.router_hist_proj = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, router_hidden),
            nn.GELU(),
        )
        # Per-dim router head: outputs one logit per action dimension so each
        # dim can independently decide how much noise to inject (see module
        # docstring). Output shape is (B, action_dim).
        self.router_fusion = nn.Sequential(
            nn.Linear(router_hidden * 2, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, router_hidden),
            nn.GELU(),
            nn.Linear(router_hidden, action_dim),
        )

        self.flow_matcher = flow_matcher
        self.num_sampling_steps = flow_matcher.num_sampling_steps
        self.flow_loss_weight = flow_loss_weight
        self.consistency_weight = consistency_weight
        self.router_entropy_weight = router_entropy_weight
        self.diversity_weight = diversity_weight
        self.diversity_k = diversity_k
        self.adaptive_steps = adaptive_steps
        self.adaptive_max_steps = adaptive_max_steps
        self.latent_dim = latent_dim
        # Backwards-compat placeholder: older checkpoints saved a `_train_step`
        # buffer (from a removed warm-up scheme). Keeping a no-op buffer here
        # lets strict load_state_dict accept those checkpoints.
        self.register_buffer("_train_step", torch.zeros(1, dtype=torch.long))

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.history_source = history_source
        self._weight_history = []
        self._steps_history = []

    # ─── helpers ──────────────────────────────────────────────────────────

    def _get_history_actions(self, nobs, raw_obs=None, full_actions=None, measured_delta=None):
        """Return normalized history actions for the flow source.

        * history_source == "agent_pos" (default): normalize raw agent_pos with the
          action normalizer. This is the classical A2A setup where state doubles
          as a history-action surrogate (state_dim == action_dim on metasim).
        * history_source == "past_action":
            - training: slice the first n_obs_steps from `full_actions` (= batch["action"]).
            - inference: read from `raw_obs["past_action"]` (eval runner supplies this).
        """
        if self.history_source == "measured_delta":
            raw = measured_delta
            if raw is None and raw_obs is not None:
                raw = raw_obs.get("measured_delta")
            assert raw is not None, (
                "history_source='measured_delta' needs batch['measured_delta'] (training) "
                "or obs_dict['measured_delta'] (inference); the LIBERO dataset and eval "
                "runner both supply it."
            )
            # Its own normalizer: the achieved motion is several times smaller
            # than the command that produced it, so the action normalizer would
            # squash it toward zero.
            return self.normalizer["measured_delta"].normalize(raw[:, :self.n_obs_steps, :])
        if self.history_source == "past_action":
            if full_actions is not None:
                raw = full_actions[:, :self.n_obs_steps, :]
            else:
                assert raw_obs is not None and "past_action" in raw_obs, (
                    "history_source='past_action' needs obs_dict['past_action'] at inference."
                )
                raw = raw_obs["past_action"]
        else:
            raw = raw_obs["agent_pos"] if raw_obs is not None else nobs["agent_pos"]
        return self.normalizer["action"].normalize(raw)

    def _predict_router_weight(self, obs_latents, history_latents):
        """Predict per-dim routing weights w ∈ [0,1]^D, shape (B, action_dim).

        Each w_d controls how much noise to mix into action dimension d
        (see module docstring). High w_d → that dim is multimodal.
        """
        h_obs = self.router_obs_proj(obs_latents)
        h_hist = self.router_hist_proj(history_latents)
        fused = torch.cat([h_obs * h_hist, h_obs + h_hist], dim=-1)
        return self._finalize_router_weight(torch.sigmoid(self.router_fusion(fused)))

    def _router_scalar(self, weight):
        """Per-sample w-bar: the max router weight over the dims it CONTROLS.

        noise_dims are pinned to w=1 so their flow source is pure noise. Folding
        that 1 into the max would make w-bar identically 1 and silently disable
        every adaptive mechanism downstream — which is exactly what happened on
        LIBERO: with the gripper pinned, L_eff collapsed to 1 and the policy saw
        a single observation frame for the whole run.
        """
        if self._noise_dim_mask is None:
            return weight.amax(dim=-1)
        return weight.masked_fill(self._noise_dim_mask, float("-inf")).amax(dim=-1)

    def _active_dim_mean(self, per_dim):
        """Mean over the dims the router controls, ignoring noise_dims.

        A pinned dim contributes a constant, so averaging it in only dilutes the
        term by |noise_dims|/D without carrying any gradient.
        """
        if self._noise_dim_mask is None:
            return per_dim.mean()
        keep = ~self._noise_dim_mask
        return (per_dim * keep).sum() / (keep.sum() * per_dim.shape[0]).clamp(min=1)

    def _finalize_router_weight(self, weight):
        """Pin noise_dims to w=1 so their flow source is pure noise.

        Applied to every router weight before it is used, so the start mixture,
        the consistency gate and the diversity term all see the same value.
        Idempotent — subclasses may call it again on their own branches.
        """
        if self._noise_dim_mask is None:
            return weight
        return torch.where(self._noise_dim_mask, torch.ones_like(weight), weight)

    def _encode_condition(self, nobs, batch_size):
        """Encode observation into global condition."""
        this_nobs = dict_apply(
            nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)
        return nobs_features.reshape(batch_size, -1)

    def _mix_start(self, history_actions, weight):
        """Build flow source: (1-w) * history + w * noise, per action dim.

        weight: (B, D)   per-dim mix coefficient
        history_actions: (B, T, D)
        Returns (B, T, D) where each dim independently controls its noise level.
        """
        # Broadcast (B, D) → (B, 1, D) over the time axis so each dim's weight
        # is shared across all T history steps but independent across dims.
        w = weight.unsqueeze(1)
        noise = torch.randn_like(history_actions)
        return (1.0 - w) * history_actions + w * noise

    def _encode_history_latent(self, history_actions):
        """Encode history action sequence → latent vector for router."""
        B = history_actions.shape[0]
        return self.history_encoder(history_actions.reshape(B, -1))

    def _weight_to_steps(self, weight):
        """Map weight [0,1]^D → integer steps per sample. Uses the per-sample
        max across dims (the most multimodal dim dictates the step budget).
        """
        max_steps = self.adaptive_max_steps
        # weight is (B, D); reduce to (B,) by max-over-dim to be conservative
        # (any dim that needs heavy noise integration sets the budget).
        w_scalar = self._router_scalar(weight) if weight.dim() > 1 else weight
        steps = torch.ceil(w_scalar.clamp(0.0, 1.0) * max_steps)
        steps = steps.clamp(min=1, max=max_steps).long()
        return steps

    def _compute_diversity_loss_global(self, start, target_spread, nn_histories, weight, nn_weights=None):
        """Per-dim diversity loss using pre-computed global KNN neighbors.

        Both target_spread and start_spread are computed PER ACTION DIMENSION
        (mean |·| over K neighbors and T steps, keeping the D axis). The relu
        comparison fires independently for each dim, giving the per-dim router
        weight a direct, decoupled gradient signal:

          - signal dim d (target_spread_d high)  → push w_d up
          - noise  dim d (target_spread_d low)   → push w_d toward 0

        Each w_d is no longer entangled with the other dims through an L2-collapsed
        spread, which is the design assumption that makes per-dim routing
        actually pay off (vs the scalar formulation it replaces).

        target_spread shape: (B, D) — per-dim, precomputed from the dataset.
        weight shape: (B, D) — per-dim router output.

        Gradient path (per dim, two-sided):
        diversity_loss_d → (start_d - nn_start_d) → w_d → router
        """
        B = start.shape[0]

        # nn_histories: (B, K, n_obs, D) — detach (fixed data)
        nn_hist = nn_histories.detach()
        D = nn_hist.shape[-1]

        # Hypothetical neighbor starts with CURRENT sample's per-dim w and
        # fresh noise per neighbor. (B, D) → (B, 1, 1, D) broadcasts each
        # dim's mix coefficient across K neighbors and T history steps.
        nn_noise = torch.randn_like(nn_hist)
        w_expanded = weight.view(B, 1, 1, D)
        nn_start = (1.0 - w_expanded) * nn_hist + w_expanded * nn_noise  # (B, K, n_obs, D)

        # Per-dim |·| spread vs neighbors. mean over n_obs (T) → (B, K, D), then
        # combine over the K neighbors: distance-weighted if nn_weights given
        # (closer neighbors contribute more), else uniform mean (legacy).
        start_self = start.unsqueeze(1)                          # (B, 1, n_obs, D)
        per_pair = (start_self - nn_start).abs().mean(dim=2)     # (B, K, D)
        if nn_weights is not None:
            per_dim_start_spread = (nn_weights.detach().unsqueeze(-1) * per_pair).sum(dim=1)  # (B, D)
        else:
            per_dim_start_spread = per_pair.mean(dim=1)          # (B, D)

        # Independent relu per (sample, dim) — each dim's gap is its own
        # gradient channel into its own w_d. Compares the data's per-dim
        # spread directly against the noise-induced start spread (no scale
        # hyperparameter — target_spread itself is the task-calibrated target).
        diversity_gap = F.relu(target_spread.detach() - per_dim_start_spread)  # (B, D)
        # noise_dims are excluded outright: their w is pinned, so the gap carries
        # no gradient and averaging it in just scales the term down.
        return self._active_dim_mean(diversity_gap)

    def _diversity_term(self, start, future_actions, history_actions, router_weight, batch):
        """Hook: return the diversity loss for this batch.

        Default = global KNN spread from the dataset (target_spread / nn_histories).
        Subclasses may override this to compute diversity without the offline KNN.
        `future_actions` / `history_actions` are passed so overrides can use them;
        the default ignores them.
        """
        assert "target_spread" in batch and "nn_histories" in batch, (
            "MARS Policy requires the KNN dataset (RobotImageDatasetKNN) "
            "to supply 'target_spread' and 'nn_histories' in the batch."
        )
        return self._compute_diversity_loss_global(
            start, batch["target_spread"], batch["nn_histories"], router_weight,
            nn_weights=batch.get("nn_weights"),
        )

    def _sample_adaptive_steps(self, start, obs_cond, weight):
        """Run flow_matcher.sample with per-sample step count (inference only)."""
        if not self.adaptive_steps:
            return self.flow_matcher.sample(
                model=self.flow_net,
                shape=start.shape,
                device=start.device,
                start=start,
                global_cond=obs_cond,
            )

        steps_per_sample = self._weight_to_steps(weight)
        output = torch.empty_like(start)
        for s in steps_per_sample.unique().tolist():
            mask = (steps_per_sample == s)
            sub_start = start[mask]
            sub_cond = obs_cond[mask]
            sub_out = self.flow_matcher.sample(
                model=self.flow_net,
                shape=sub_start.shape,
                device=sub_start.device,
                num_steps=int(s),
                start=sub_start,
                global_cond=sub_cond,
            )
            output[mask] = sub_out
        return output

    # ─── training ─────────────────────────────────────────────────────────

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]

        # 1. Visual + proprioceptive condition
        obs_cond = self._encode_condition(nobs, batch_size)

        # 2. History & future actions
        history_actions = self._get_history_actions(
            nobs, batch["obs"], full_actions=batch.get("action"),
            measured_delta=batch.get("measured_delta"),
        )[:, :self.n_obs_steps, :]
        future_start = self.n_obs_steps - 1
        future_end = future_start + self.n_action_steps
        future_actions = nactions[:, future_start:future_end, :]

        # 3. Router weight (NOT detached — end-to-end gradient through start)
        history_latent = self._encode_history_latent(history_actions)
        router_weight = self._predict_router_weight(obs_cond, history_latent)

        # 4. Build flow source (gradient flows through router_weight)
        start = self._mix_start(history_actions, router_weight)

        # 5. Flow matching loss
        flow_loss, metrics = self.flow_matcher.compute_loss(
            self.flow_net,
            target=future_actions,
            start=start,
            global_cond=obs_cond,
        )
        loss = self.flow_loss_weight * flow_loss
        metrics["flow_loss"] = flow_loss.item()

        # 6. Single-step prediction (shared by consistency loss + diversity loss)
        t_zero = torch.zeros(batch_size, device=start.device)
        velocity = self.flow_net(start, t_zero, global_cond=obs_cond)
        pred_actions = start + velocity  # single-step Euler

        # 7. Consistency loss — single-step Euler prediction error, scaled by
        #    a per-dim (1 - w_d) gate.
        #
        #    Gradient flow into the router (this is what the docstring
        #    previously got wrong — the detach only blocks ONE of two paths):
        #
        #      consistency_loss
        #        = mean( (1 - w).detach()  *  | (start + v(start)) - future | )
        #                       ↑                  ↑
        #                       │                  │
        #     (a) coefficient   │                  │  (b) error term
        #         path: gradient│ blocked by       │      path: NOT blocked.
        #         detach. Stops │ the router       │      `start` carries w
        #         a degenerate  │ from gaming the  │      via _mix_start, so
        #         shortcut where│ loss by setting  │      d(loss)/dw flows
        #         w → 1 simply  │ (1 - w) → 0.     │      back to the router.
        #         to zero out   │                  │
        #         the loss.     │                  │  Direction: w↑ → start
        #                                          │  more noisy → single-step
        #                                          │  pred drifts → loss↑ →
        #                                          │  gradient pushes w↓.
        #
        #    Net effect: consistency loss is a learned counter-pressure to
        #    diversity loss (which pushes w↑). Dims that are unimodal end up
        #    with low w because their consistency penalty dominates; dims
        #    that are multimodal accept the consistency hit because
        #    diversity_loss gain outweighs it.
        if self.consistency_weight > 0:
            per_sample_per_dim_loss = F.l1_loss(
                pred_actions, future_actions, reduction="none"
            ).mean(dim=-2)  # (B, D) — average across the time horizon
            sample_weight = (1.0 - router_weight).detach()  # (B, D), see (a) above
            consistency_loss = (sample_weight * per_sample_per_dim_loss).mean()

            loss += self.consistency_weight * consistency_loss
            metrics["consistency_loss"] = consistency_loss.item()
            metrics["consistency_weight_mean"] = sample_weight.mean().item()

        # 8. Diversity loss (global KNN target_spread + nn_histories from dataset)
        if self.diversity_weight > 0:
            assert "target_spread" in batch and "nn_histories" in batch, (
                "MARS Policy requires the KNN dataset (RobotImageDatasetKNN) "
                "to supply 'target_spread' and 'nn_histories' in the batch."
            )
            diversity_loss = self._compute_diversity_loss_global(
                start, batch["target_spread"], batch["nn_histories"], router_weight,
            )
            loss += self.diversity_weight * diversity_loss
            metrics["diversity_loss"] = diversity_loss.item()

        # 9. Entropy regularization (safety net against total collapse).
        #    Per-dim binary entropy averaged over batch and dims. With per-dim
        #    w, each dim's entropy is independent, naturally encouraging some
        #    dims to stay near 0.5 instead of all collapsing together.
        if self.router_entropy_weight > 0:
            w = router_weight  # (B, D)
            entropy = -(
                w * torch.log(w + 1e-6)
                + (1.0 - w) * torch.log(1.0 - w + 1e-6)
            ).mean()
            loss -= self.router_entropy_weight * entropy
            metrics["router_entropy"] = entropy.item()

        # router_weight is (B, D). Track scalar summaries + per-dim mean for
        # diagnosing which dims the router learns to activate.
        _act = router_weight if self._noise_dim_mask is None else router_weight[
            :, (~self._noise_dim_mask)[0]
        ]
        metrics["router_weight_mean"] = _act.mean().item()
        metrics["router_weight_std"] = router_weight.std().item()
        metrics["router_weight_max_dim"] = _act.mean(dim=0).max().item()
        metrics["router_weight_min_dim"] = _act.mean(dim=0).min().item()

        self._last_metrics = metrics
        return loss

    # ─── inference ────────────────────────────────────────────────────────

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # _normalize_obs strips the flow-source keys, which have no obs normalizer.
        nobs = self._normalize_obs(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        # Visual + proprioceptive condition
        obs_cond = self._encode_condition(nobs, B)

        # History actions (reads past_action from obs_dict when history_source="past_action")
        history_actions = self._get_history_actions(nobs, obs_dict)[:, :self.n_obs_steps, :]

        # Predict router weight
        with torch.no_grad():
            history_latent = self._encode_history_latent(history_actions)
            router_weight = self._predict_router_weight(obs_cond, history_latent)

        # Build flow source
        start = self._mix_start(history_actions, router_weight)

        # Flow sampling (adaptive steps based on router weight)
        if self.adaptive_steps:
            action_pred = self._sample_adaptive_steps(
                start=start, obs_cond=obs_cond, weight=router_weight,
            )
        else:
            action_pred = self.flow_matcher.sample(
                model=self.flow_net,
                shape=start.shape,
                device=start.device,
                start=start,
                global_cond=obs_cond,
            )

        action_pred = self.normalizer["action"].unnormalize(action_pred)
        action = action_pred[:, :self.n_action_steps]

        # router_weight is (B, D). Store the per-sample max for the legacy
        # scalar history; downstream eval scripts that just want one number
        # per step still work.
        self._weight_history.append(router_weight[0].max().item())
        if self.adaptive_steps:
            steps_used = self._weight_to_steps(router_weight)[0].item()
        else:
            steps_used = self.num_sampling_steps
        self._steps_history.append(steps_used)

        return {
            "action": action,
            "action_pred": action_pred,
            "router_weight": router_weight,
            # Legacy scalar score for eval_runner: per-sample max across dims,
            # matching the `_weight_history` entry stored above.
            "score": self._router_scalar(router_weight),
        }

    def reset(self):
        self._weight_history = []
        self._steps_history = []

    def get_weight_history(self):
        return list(self._weight_history)

    # eval compatibility: eval_2d_navigation.py checks for get_score_history
    def get_score_history(self):
        return list(self._weight_history)

    def get_steps_history(self):
        return list(self._steps_history)

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    # ─── visualization ────────────────────────────────────────────────────

    @torch.no_grad()
    def get_flow_trajectories(self, batch, num_steps=None, n_samples=5):
        if num_steps is None:
            num_steps = self.num_sampling_steps

        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        n_samples = min(n_samples, batch_size)

        nobs_sub = dict_apply(nobs, lambda x: x[:n_samples])
        obs_cond = self._encode_condition(nobs_sub, n_samples)

        history_actions = self._get_history_actions(
            nobs, batch["obs"], full_actions=batch.get("action"),
            measured_delta=batch.get("measured_delta"),
        )[:n_samples, :self.n_obs_steps, :]
        future_start = self.n_obs_steps - 1
        future_end = future_start + self.n_action_steps
        future_actions = nactions[:n_samples, future_start:future_end, :]

        # Router-adaptive start
        history_latent = self._encode_history_latent(history_actions)
        router_weight = self._predict_router_weight(obs_cond, history_latent)
        start = self._mix_start(history_actions, router_weight)

        _, (traj_history, vel_history) = self.flow_matcher.sample(
            model=self.flow_net,
            shape=start.shape,
            device=start.device,
            num_steps=num_steps,
            start=start,
            return_traces=True,
            global_cond=obs_cond,
        )

        traj_flat = []
        for t in traj_history:
            if hasattr(t, "cpu"):
                traj_flat.append(t.cpu().reshape(n_samples, -1))
            else:
                traj_flat.append(torch.tensor(t).reshape(n_samples, -1))

        traj_stacked = torch.stack(traj_flat, dim=0)
        trajectories = [traj_stacked[:, i, :].numpy() for i in range(n_samples)]
        future_flat = future_actions.cpu().reshape(n_samples, -1).numpy()

        return trajectories, future_flat
