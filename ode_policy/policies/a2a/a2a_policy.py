"""
Action-to-Action Flow Matching Policy (A2A)

A flow matching policy that directly transforms history action distributions
to future action distributions, conditioned on visual observations.

Architecture:
    History States [s_{t-n+1}, ..., s_t] --encode--> history_latents (x_0)
    Visual Obs [img_{t-n+1}, ..., img_t] --encode--> obs_latents (condition)
    
    Flow Matching: x_0 --flow(condition)--> x_1 (future_action_latents)
    
    x_1 --decode--> Future Actions [a_t, a_{t+1}, ..., a_{t+k}]
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ode_policy.utils.normalizer import LinearNormalizer
from ode_policy.utils.pytorch_util import dict_apply
from ode_policy.policies.base_image_policy import BaseImagePolicy

from ode_policy.utils.models.flow_net import SimpleFlowNet
from ode_policy.policies.a2a.action_ae import CNNActionEncoder, SimpleActionDecoder
from ode_policy.utils.vision.multi_image_obs_encoder import MultiImageObsEncoder
from ode_policy.utils.flow.flow_matchers import TorchFlowMatcher


class A2AImagePolicy(BaseImagePolicy):
    """
    Action-to-Action Flow Matching Policy.
    
    Flow Matching from history states to future actions, conditioned on visual observations.
    
    - Flow START: History states (past n_obs_steps proprioceptive states)
    - Flow TARGET: Future actions (next n_action_steps)
    - CONDITION: Visual observation latents
    """

    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        # A2A specific params
        flow_net,
        flow_matcher: TorchFlowMatcher,
        decode_flow_latents=True,
        consistency_weight=1.0,
        enc_contrastive_weight=1e-4,
        flow_contrastive_weight=0.0,
        latent_dim=512,
        action_ae=None,
        adaptive_weights=False,
        use_residual_images=False,
        use_temporal_conv=False,
        **kwargs,
    ):
        super().__init__()

        # A2A: visual-only condition — remove agent_pos from obs_encoder
        # (agent_pos is used as history flow source, not as condition)
        if "agent_pos" in obs_encoder.key_shape_map:
            obs_encoder.low_dim_keys = [k for k in obs_encoder.low_dim_keys if k != "agent_pos"]
            del obs_encoder.key_shape_map["agent_pos"]

        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim (now excludes agent_pos)
        obs_feature_dim = obs_encoder.output_shape()[0]

        self.decode_flow_latents = decode_flow_latents
        self.consistency_weight = consistency_weight
        self.enc_contrastive_weight = enc_contrastive_weight
        self.flow_contrastive_weight = flow_contrastive_weight
        self.latent_dim = latent_dim
        self.num_sampling_steps = flow_matcher.num_sampling_steps

        self.flow_matcher = flow_matcher
        self.action_ae = action_ae

        # Visual observation encoder (condition on all n_obs_steps images)
        self.n_cond_steps = n_obs_steps
        self.use_residual_images = use_residual_images
        self.use_temporal_conv = use_temporal_conv
        self.obs_encoder = obs_encoder

        # Learnable temporal convolution: applies 1D conv along time axis on per-frame features
        # The kernel learns optimal temporal filter (can recover residual, smoothing, etc.)
        if use_temporal_conv:
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(obs_feature_dim, obs_feature_dim, kernel_size=3, padding=1, groups=1),
                nn.ReLU(),
                nn.Conv1d(obs_feature_dim, obs_feature_dim, kernel_size=3, padding=1, groups=1),
            )
        else:
            self.temporal_conv = None

        # residual mode: 1st frame original + 7 residual frames = 8 total, same dim
        proj_steps = self.n_cond_steps
        self.obs_projector = nn.Linear(
            obs_feature_dim * proj_steps,
            latent_dim
        )

        # Flow network with condition support
        self.flow_net = SimpleFlowNet(
            input_dim=latent_dim,
            hidden_dim=flow_net.hidden_dim,
            output_dim=latent_dim,
            num_layers=flow_net.num_layers,
            mlp_ratio=flow_net.mlp_ratio,
            dropout=flow_net.dropout,
            condition_dim=latent_dim,  # obs_latents as condition
        )
        
        # History state encoder (n_obs_steps frames)
        self.history_action_encoder = CNNActionEncoder(
            pred_horizon=n_obs_steps,  # History length = n_obs_steps
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=action_ae.net.enc_hidden_dim,
        )

        # Future action encoder/decoder (n_action_steps future actions)
        future_horizon = n_action_steps  # 8 future actions
        self.future_horizon = future_horizon

        self.action_encoder = CNNActionEncoder(
            pred_horizon=future_horizon,
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=action_ae.net.enc_hidden_dim,
        )
        self.action_decoder = SimpleActionDecoder(
            dec_hidden_dim=action_ae.net.dec_hidden_dim,
            latent_dim=latent_dim,
            pred_horizon=future_horizon,
            action_dim=action_dim,
            num_layers=action_ae.net.num_layers,
            dropout=action_ae.net.dropout,
        )

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

        # Adaptive loss weighting (Kendall et al., 2018)
        # Learns log-variance per auxiliary loss: w_i = exp(-s_i), regularized by +0.5*s_i
        self.adaptive_weights = adaptive_weights
        if self.adaptive_weights:
            self.log_vars = nn.ParameterDict({
                'consistency': nn.Parameter(torch.tensor(0.0)),
                'flow_recon': nn.Parameter(torch.tensor(0.0)),
                'enc_recon': nn.Parameter(torch.tensor(0.0)),
            })
        self._last_metrics = {}

    def _get_history_actions(self, nobs, raw_obs=None):
        """Get history actions normalized with action normalizer.

        agent_pos in obs has the same physical meaning as action (e.g. both are
        velocities or both are joint positions). We normalize with the action
        normalizer so that flow start and target live in the same space.
        """
        raw = raw_obs["agent_pos"] if raw_obs is not None else nobs["agent_pos"]
        return self.normalizer["action"].normalize(raw)

    def _encode_visual_condition(self, nobs, batch_size):
        """Encode visual observations into condition latents, with optional residual images."""
        # Slice last n_cond_steps frames
        sliced = dict_apply(nobs, lambda x: x[:, self.n_obs_steps - self.n_cond_steps:self.n_obs_steps, ...])

        if self.use_residual_images:
            # 1st frame original + 7 residual frames = 8 total (same time dimension)
            result = {}
            for key, val in sliced.items():
                if val.dim() == 5 and val.shape[2] == 3:  # (B, T, C, H, W) image
                    first_frame = val[:, :1]  # (B, 1, C, H, W)
                    residuals = val[:, 1:] - val[:, :-1]  # (B, T-1, C, H, W)
                    result[key] = torch.cat([first_frame, residuals], dim=1)  # (B, T, C, H, W)
                else:
                    result[key] = val  # low-dim unchanged, same T
            this_nobs = dict_apply(result, lambda x: x.reshape(-1, *x.shape[2:]))
        else:
            this_nobs = dict_apply(sliced, lambda x: x.reshape(-1, *x.shape[2:]))

        nobs_features = self.obs_encoder(this_nobs)

        # Apply learnable temporal convolution on per-frame features
        if self.temporal_conv is not None:
            # (B*T, D) → (B, T, D) → (B, D, T) → Conv1d → (B, D, T) → (B, T, D) → (B*T, D)
            T = self.n_cond_steps
            nobs_features = nobs_features.reshape(batch_size, T, -1)  # (B, T, D)
            nobs_features = nobs_features.transpose(1, 2)  # (B, D, T)
            nobs_features = self.temporal_conv(nobs_features)  # (B, D, T)
            nobs_features = nobs_features.transpose(1, 2)  # (B, T, D)

        nobs_features = nobs_features.reshape(batch_size, -1)
        return self.obs_projector(nobs_features)

    def compute_loss(self, batch):
        # normalize input
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]

        # Encode visual observations as condition
        obs_latents = self._encode_visual_condition(nobs, batch_size)

        # Encode history states (n_obs_steps frames)
        history_states = self._get_history_actions(nobs, batch["obs"])[:, :self.n_obs_steps, :]  # (B, 8, D)
        history_latents = self.history_action_encoder(history_states)

        # Encode future actions (8 frames starting from current time)
        future_start = self.n_obs_steps - 1  # = 7
        future_end = future_start + self.n_action_steps  # = 15
        future_actions = nactions[:, future_start:future_end, :]  # (B, 8, D)
        future_action_latents = self.action_encoder(future_actions)
        
        # Flow matching: history_latents -> future_action_latents, conditioned on obs_latents
        flow_loss, metrics = self.flow_matcher.compute_loss(
            self.flow_net,
            target=future_action_latents,
            start=history_latents,  # History states as flow source
            global_cond=obs_latents,  # Visual observations as condition
        )
        
        loss = flow_loss
        metrics['flow_loss'] = flow_loss.item()

        # Encoder contrastive loss
        if self.enc_contrastive_weight > 0:
            image_features = obs_latents.view(batch_size, -1)
            action_features = future_action_latents.view(batch_size, -1)
            contrastive_loss = self._compute_contrastive_loss(image_features, action_features)
            loss += self.enc_contrastive_weight * contrastive_loss
            metrics['enc_contrastive_loss'] = contrastive_loss.item()

        # Flow latent decoding
        if self.decode_flow_latents:
            # Sample with history states as start and obs as condition
            action_latents_pred = self.flow_matcher.sample(
                self.flow_net,
                shape=(batch_size, self.latent_dim),
                device=obs_latents.device,
                start=history_latents,
                num_steps=self.num_sampling_steps,
                global_cond=obs_latents,
            )

            if self.consistency_weight > 0:
                consistency_loss = F.mse_loss(action_latents_pred, future_action_latents)
                metrics['consistency_loss'] = consistency_loss.item()
                if self.adaptive_weights:
                    w = torch.exp(-self.log_vars['consistency'])
                    loss += 0.5 * w * consistency_loss + 0.5 * self.log_vars['consistency']
                else:
                    loss += self.consistency_weight * consistency_loss

            if self.flow_contrastive_weight > 0:
                image_features = obs_latents.view(batch_size, -1)
                action_features = action_latents_pred.view(batch_size, -1)
                contrastive_loss = self._compute_contrastive_loss(image_features, action_features)
                loss += self.flow_contrastive_weight * contrastive_loss
                metrics['flow_contrastive_loss'] = contrastive_loss.item()

            if self.action_ae["flow_recon_weight"] > 0:
                actions_recon = self.action_decoder(action_latents_pred)
                action_recon_loss = F.l1_loss(actions_recon, future_actions)
                metrics['flow_action_recon_loss'] = action_recon_loss.item()
                if self.adaptive_weights:
                    w = torch.exp(-self.log_vars['flow_recon'])
                    loss += 0.5 * w * action_recon_loss + 0.5 * self.log_vars['flow_recon']
                else:
                    loss += self.action_ae["flow_recon_weight"] * action_recon_loss
        else:
            action_latents_pred = future_action_latents

        # Encoder reconstruction losses
        if self.action_ae["enc_recon_weight"] > 0:
            actions_recon = self.action_decoder(future_action_latents)
            action_recon_loss = F.l1_loss(actions_recon, future_actions)
            metrics['enc_action_recon_loss'] = action_recon_loss.item()
            if self.adaptive_weights:
                w = torch.exp(-self.log_vars['enc_recon'])
                loss += 0.5 * w * action_recon_loss + 0.5 * self.log_vars['enc_recon']
            else:
                loss += self.action_ae["enc_recon_weight"] * action_recon_loss

        # Composite "consistent loss" = weighted consistency + weighted flow_recon
        consistent_loss_val = 0.0
        if 'consistency_loss' in metrics:
            consistent_loss_val += self.consistency_weight * metrics['consistency_loss']
        if 'flow_action_recon_loss' in metrics:
            consistent_loss_val += self.action_ae["flow_recon_weight"] * metrics['flow_action_recon_loss']
        metrics['consistent_loss'] = consistent_loss_val
        metrics['total_loss'] = loss.item()

        # Log adaptive weights
        if self.adaptive_weights:
            for name, lv in self.log_vars.items():
                metrics[f'aw_{name}'] = torch.exp(-lv).item()
                metrics[f'log_var_{name}'] = lv.item()

        self._last_metrics = metrics
        return loss

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # normalize input
        nobs = self._normalize_obs(obs_dict)
        value = next(iter(nobs.values()))
        B = value.shape[0]

        # Encode visual observations as condition
        obs_latents = self._encode_visual_condition(nobs, B)

        # Encode history states (n_obs_steps frames)
        history_states = self._get_history_actions(nobs, obs_dict)[:, :self.n_obs_steps, :]  # (B, 8, D)
        history_latents = self.history_action_encoder(history_states)
        
        # Run flow sampling: history -> future, conditioned on visual obs
        action_latents_pred = self.flow_matcher.sample(
            self.flow_net,
            shape=(B, self.latent_dim),
            device=obs_latents.device,
            num_steps=self.num_sampling_steps,
            start=history_latents,  # History states as flow source
            global_cond=obs_latents,  # Visual observations as condition
            return_traces=False
        )

        with torch.no_grad():
            action_pred = self.action_decoder(action_latents_pred)

        # unnormalize prediction
        action_pred = self.normalizer["action"].unnormalize(action_pred)

        # action_pred is 8 future actions [a_t, ..., a_{t+7}]
        action = action_pred[:, :self.n_action_steps]  # (B, 8, D)

        result = {"action": action, "action_pred": action_pred}
        return result

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    @torch.no_grad()
    def get_latents_for_visualization(self, batch):
        """
        Extract history and future latents for t-SNE visualization.
        
        Returns:
            history_latents: (B, latent_dim) - encoded from past states
            future_latents: (B, latent_dim) - encoded from future actions
        """
        # normalize input
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        
        # Encode history states (past n_obs_steps states)
        history_states = self._get_history_actions(nobs, batch["obs"])[:, :self.n_obs_steps, :]  # (B, 8, D)
        history_latents = self.history_action_encoder(history_states)

        # Encode future actions (next n_action_steps actions)
        future_start = self.n_obs_steps - 1
        future_end = future_start + self.n_action_steps
        future_actions = nactions[:, future_start:future_end, :]  # (B, 8, D)
        future_latents = self.action_encoder(future_actions)
        
        return history_latents, future_latents

    @torch.no_grad()
    def get_flow_trajectories(self, batch, num_steps=None, n_samples=5):
        """
        Get flow trajectories for visualization.
        
        Args:
            batch: Input batch
            num_steps: Number of flow steps to sample (default: use self.num_sampling_steps)
            n_samples: Number of sample trajectories to return
            
        Returns:
            trajectories: List of (num_steps+1, latent_dim) arrays for each sample
            future_latents: (n_samples, latent_dim) - ground truth targets
        """
        # Use configured num_sampling_steps if not specified
        if num_steps is None:
            num_steps = self.num_sampling_steps
        
        # normalize input
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]
        
        # Only use first n_samples
        n_samples = min(n_samples, batch_size)
        
        # Encode visual observations as condition
        nobs_subset = dict_apply(nobs, lambda x: x[:n_samples])
        obs_latents = self._encode_visual_condition(nobs_subset, n_samples)

        # Encode history states
        history_states = self._get_history_actions(nobs, batch["obs"])[:n_samples, :self.n_obs_steps, :]
        history_latents = self.history_action_encoder(history_states)
        
        # Encode future actions (ground truth target)
        future_start = self.n_obs_steps - 1
        future_end = future_start + self.n_action_steps
        future_actions = nactions[:n_samples, future_start:future_end, :]
        future_latents = self.action_encoder(future_actions)
        
        # Run flow sampling with traces
        _, (traj_history, _) = self.flow_matcher.sample(
            self.flow_net,
            shape=(n_samples, self.latent_dim),
            device=obs_latents.device,
            num_steps=num_steps,
            start=history_latents,
            global_cond=obs_latents,
            return_traces=True
        )
        
        # Convert to numpy arrays (ensure all tensors are on CPU first)
        traj_history_cpu = []
        for t in traj_history:
            if hasattr(t, 'cpu'):
                traj_history_cpu.append(t.cpu())
            else:
                traj_history_cpu.append(torch.tensor(t))
        
        traj_stacked = torch.stack(traj_history_cpu, dim=0)  # (num_steps+1, n_samples, latent_dim)
        trajectories = [traj_stacked[:, i, :].numpy() for i in range(n_samples)]
        future_latents_np = future_latents.cpu().numpy()
        
        return trajectories, future_latents_np

    @staticmethod
    def _compute_contrastive_loss(image_features, action_features, temperature=0.07):
        """Contrastive loss between image and action features (InfoNCE)"""
        batch_size = image_features.size(0)
        image_features = F.normalize(image_features, dim=1)
        action_features = F.normalize(action_features, dim=1)

        # Compute similarity matrix
        logits = torch.matmul(image_features, action_features.T) / temperature

        # Symmetric contrastive loss (image-to-action + action-to-image)
        labels = torch.arange(batch_size, device=logits.device)
        loss_i2a = F.cross_entropy(logits, labels)
        loss_a2i = F.cross_entropy(logits.T, labels)

        return (loss_i2a + loss_a2i) / 2
