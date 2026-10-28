"""
Action-to-Action Flow Matching with DiT (A2A-DiT)

A variant of A2A that removes the autoencoder and performs flow matching
directly in the raw action space using a DiT (FlowTransformer) backbone.

Architecture:
    History Actions [a_{t-n+1}, ..., a_t]  (B, n_obs_steps, action_dim)  --> x_0
    Visual Obs [img_{t-n+1}, ..., img_t]                                 --> condition

    Flow Matching (DiT): x_0 --flow(condition)--> x_1

    Future Actions [a_t, ..., a_{t+k}]    (B, n_action_steps, action_dim) --> x_1
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
from ode_policy.utils.history import assert_history_compatible
from ode_policy.utils.flow.flow_matchers import TorchFlowMatcher


class A2ADiTImagePolicy(BaseImagePolicy):
    """
    Action-to-Action Flow Matching with DiT in raw action space.

    - Flow START (x_0): History proprioceptive actions (past n_obs_steps steps)
    - Flow TARGET (x_1): Future actions (next n_action_steps steps)
    - CONDITION: Visual observation features (global conditioning via AdaLN)

    Requires n_obs_steps == n_action_steps for shape compatibility.
    """

    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        # Flow matcher (instantiated by Hydra)
        flow_matcher: TorchFlowMatcher,
        # DiT architecture params
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
        diffusion_step_embed_dim=256,
        # Obs projector
        latent_dim=512,
        # Optional losses / augmentation
        flow_loss_weight=1.0,
        consistency_weight=0.0,
        consistency_train_steps=2,
        history_noise_var=0.0,
        history_action_weight=1.0,
        # Where the flow's source (history) signal comes from. "agent_pos" reuses
        # the state as a history-action surrogate and therefore needs
        # state_dim == action_dim (true on metasim joint_pos). "past_action" reads
        # the real past actions and is always valid — required on LIBERO, whose
        # state is an 8-dim pose while the action is a 7-dim delta.
        history_source="agent_pos",
        # Action dims whose flow source is ALWAYS Gaussian noise. LIBERO's
        # gripper (dim 6) is a near-binary open/close signal; its history is a
        # step function, so transporting the future chunk from a past value of it
        # carries no useful structure.
        noise_dims=None,
        **kwargs,
    ):
        super().__init__()

        # Parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_feature_dim = obs_encoder.output_shape()[0]

        # Fail loudly at construction rather than training a miscalibrated flow
        # start: with history_source="agent_pos" the state is fed straight into
        # the action normalizer, which is meaningless when the two differ.
        state_shape = shape_meta.get("obs", {}).get("agent_pos", {}).get("shape")
        if state_shape is not None:
            assert_history_compatible(
                history_source, state_shape[0], action_dim,
                owner=type(self).__name__,
            )

        assert n_obs_steps == n_action_steps, (
            f"A2A-DiT requires n_obs_steps == n_action_steps for shape compatibility, "
            f"got {n_obs_steps} vs {n_action_steps}"
        )

        self.noise_dims = tuple(noise_dims) if noise_dims else ()
        if self.noise_dims:
            bad = [d for d in self.noise_dims if not (0 <= d < action_dim)]
            if bad:
                raise ValueError(f"noise_dims {bad} out of range for action_dim={action_dim}")
            mask = torch.zeros(1, 1, action_dim, dtype=torch.bool)
            mask[0, 0, list(self.noise_dims)] = True
            self.register_buffer("_noise_dim_mask", mask, persistent=False)
        else:
            self._noise_dim_mask = None

        # Obs encoder: includes all obs keys (visual + agent_pos), same as FM-DiT
        self.obs_encoder = obs_encoder

        # DiT flow network: each action timestep is a token
        # Global cond is flattened obs_encoder features, same as FM-DiT
        global_cond_dim = obs_feature_dim * n_obs_steps
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

        self.flow_matcher = flow_matcher
        self.num_sampling_steps = flow_matcher.num_sampling_steps
        self.flow_loss_weight = flow_loss_weight
        self.consistency_weight = consistency_weight
        self.consistency_train_steps = consistency_train_steps
        self.history_noise_var = history_noise_var
        self.history_action_weight = history_action_weight
        self.history_source = history_source
        self.latent_dim = latent_dim

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

    def _get_history_actions(self, nobs, raw_obs=None, full_actions=None, measured_delta=None):
        """Return the normalized history actions used as the flow's source.

        * history_source == "agent_pos" (default): agent_pos has the same physical
          meaning as the action (both velocities, or both joint positions), so the
          state doubles as a history-action surrogate. Normalized with the ACTION
          normalizer so flow start and target live in one space.
        * history_source == "past_action": use the real past actions —
            - training: the first n_obs_steps of `full_actions` (= batch["action"]),
            - inference: `raw_obs["past_action"]`, supplied by the eval runner.
          Required whenever state and action differ in dim or semantics (LIBERO).
        """
        if self.history_source == "measured_delta":
            raw = measured_delta
            if raw is None and raw_obs is not None:
                raw = raw_obs.get("measured_delta")
            assert raw is not None, (
                "history_source='measured_delta' needs batch['measured_delta'] (training) "
                "or obs_dict['measured_delta'] (inference)."
            )
            # Own normalizer: achieved motion is several times smaller than the
            # command that produced it.
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

    def _add_history_noise(self, history_actions: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to history actions (applied after normalization)."""
        if self.history_noise_var > 0:
            noise_std = torch.sqrt(torch.tensor(self.history_noise_var, device=history_actions.device))
            noise = torch.randn_like(history_actions) * noise_std
            return history_actions + noise
        return history_actions

    def _mix_history_with_noise(self, history_actions: torch.Tensor) -> torch.Tensor:
        """Mix history actions with pure noise based on history_action_weight.

        start = w * history + (1-w) * noise
        w=1: pure A2A (start from history), w=0: pure noise (standard flow matching)
        """
        w = self.history_action_weight
        if w >= 1.0:
            return self._apply_noise_dims(history_actions)
        noise = torch.randn_like(history_actions)
        if w <= 0.0:
            return noise
        return self._apply_noise_dims(w * history_actions + (1.0 - w) * noise)

    def _apply_noise_dims(self, start: torch.Tensor) -> torch.Tensor:
        """Replace the flow source of `noise_dims` with pure Gaussian noise."""
        if self._noise_dim_mask is None:
            return start
        return torch.where(self._noise_dim_mask, torch.randn_like(start), start)

    def _encode_condition(self, nobs, batch_size):
        """Encode observation into global condition (same as FM-DiT).

        Uses obs_encoder output directly (which already includes agent_pos),
        flattens time dimension, no extra projector.
        """
        this_nobs = dict_apply(
            nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:])
        )
        nobs_features = self.obs_encoder(this_nobs)
        return nobs_features.reshape(batch_size, -1)  # (B, obs_feature_dim * n_obs_steps)

    # ========= training ============
    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]

        # 1. Encode visual + proprioceptive observations -> global condition
        obs_cond = self._encode_condition(nobs, batch_size)

        # 2. History actions (x_0) and future actions (x_1)
        history_actions = self._get_history_actions(nobs, batch["obs"], full_actions=batch["action"], measured_delta=batch.get("measured_delta"))[:, :self.n_obs_steps, :]  # (B, 8, action_dim)

        future_start = self.n_obs_steps - 1
        future_end = future_start + self.n_action_steps
        future_actions = nactions[:, future_start:future_end, :]  # (B, 8, action_dim)

        # 3. Optional: add Gaussian noise, then mix with pure noise
        history_actions = self._add_history_noise(history_actions)
        history_actions = self._mix_history_with_noise(history_actions)

        # 4. Flow matching loss in raw action space
        flow_loss, metrics = self.flow_matcher.compute_loss(
            self.flow_net,
            target=future_actions,
            start=history_actions,
            global_cond=obs_cond,
        )
        loss = self.flow_loss_weight * flow_loss
        metrics["flow_loss"] = flow_loss.item()

        # 5. Optional: consistency loss
        if self.consistency_weight > 0:
            pred_actions = self.flow_matcher.sample(
                model=self.flow_net,
                shape=history_actions.shape,
                device=history_actions.device,
                start=history_actions,
                num_steps=self.consistency_train_steps,
                global_cond=obs_cond,
            )
            consistency_loss = F.l1_loss(pred_actions, future_actions)
            loss = loss + self.consistency_weight * consistency_loss
            metrics["consistency_loss"] = consistency_loss.item()

        self._last_metrics = metrics
        return loss

    # ========= inference ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs = self._normalize_obs(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        # Encode visual + proprioceptive observations
        obs_cond = self._encode_condition(nobs, B)

        # History actions as flow starting point
        history_actions = self._get_history_actions(nobs, obs_dict)[:, :self.n_obs_steps, :]  # (B, n_obs_steps, action_dim)

        # Add noise, then mix with pure noise (same as training)
        history_actions = self._add_history_noise(history_actions)
        history_actions = self._mix_history_with_noise(history_actions)

        # Flow sampling: history -> future via flow_matcher.sample (Euler ODE)
        action_pred = self.flow_matcher.sample(
            model=self.flow_net,
            shape=history_actions.shape,
            device=history_actions.device,
            start=history_actions,
            global_cond=obs_cond,
        )

        # Unnormalize (no decoder needed)
        action_pred = self.normalizer["action"].unnormalize(action_pred)
        action = action_pred[:, :self.n_action_steps]

        return {"action": action, "action_pred": action_pred}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    # ========= visualization ============
    @torch.no_grad()
    def get_flow_trajectories(self, batch, num_steps=None, n_samples=5):
        """
        Get flow trajectories for visualization.

        Returns:
            trajectories: List of (num_steps+1, n_action_steps * action_dim) arrays
            future_actions_flat: (n_samples, n_action_steps * action_dim) ground truth
        """
        if num_steps is None:
            num_steps = self.num_sampling_steps

        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        n_samples = min(n_samples, batch_size)

        # Encode visual + proprioceptive observations
        nobs_sub = dict_apply(nobs, lambda x: x[:n_samples])
        obs_cond = self._encode_condition(nobs_sub, n_samples)

        # History and future actions
        history_actions = self._get_history_actions(nobs, batch["obs"], full_actions=batch["action"], measured_delta=batch.get("measured_delta"))[:n_samples, :self.n_obs_steps, :]

        future_start = self.n_obs_steps - 1
        future_end = future_start + self.n_action_steps
        future_actions = nactions[:n_samples, future_start:future_end, :]

        # Flow sampling with traces
        _, (traj_history, vel_history) = self.flow_matcher.sample(
            model=self.flow_net,
            shape=history_actions.shape,
            device=history_actions.device,
            num_steps=num_steps,
            start=history_actions,
            return_traces=True,
            global_cond=obs_cond,
        )

        # Flatten to 2D for t-SNE visualization
        traj_flat = []
        for t in traj_history:
            if hasattr(t, "cpu"):
                traj_flat.append(t.cpu().reshape(n_samples, -1))
            else:
                traj_flat.append(torch.tensor(t).reshape(n_samples, -1))

        traj_stacked = torch.stack(traj_flat, dim=0)  # (num_steps+1, n_samples, S*D)
        trajectories = [traj_stacked[:, i, :].numpy() for i in range(n_samples)]
        future_flat = future_actions.cpu().reshape(n_samples, -1).numpy()

        return trajectories, future_flat
