from typing import Dict

import torch
import torch.nn.functional as F
from einops import reduce

from ode_policy.utils.models.flow_net import FlowTransformer
from ode_policy.policies.dp.models.diffusion.mask_generator import LowdimMaskGenerator
from ode_policy.utils.vision.multi_image_obs_encoder import MultiImageObsEncoder
from ode_policy.utils.normalizer import LinearNormalizer
from ode_policy.utils.pytorch_util import dict_apply
from ode_policy.policies.base_image_policy import BaseImagePolicy


class FlowMatchingDiTImagePolicy(BaseImagePolicy):
    def __init__(
        self,
        shape_meta: dict,
        obs_encoder: MultiImageObsEncoder,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        init_noise_std=1.0,
        relative_dims=0,
        relative_scale=None,
        **kwargs,
    ):
        """relative_dims > 0: the first relative_dims action dims (an absolute pose) are
        learned relative to the pose at the current step (row n_obs_steps - 1 of the window):
        in normalised units, minus that row, times relative_scale (per dim; the offsets are
        small, so they are scaled up to about the spread of the absolute poses). At inference
        the current pose comes in as predict_action(..., a0=pose) and is added back."""
        super().__init__()

        # parse shapes
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        # Instantiate the Unet model
        model = FlowTransformer(
            input_dim=input_dim,
            condition_dim=global_cond_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            time_embed_dim=diffusion_step_embed_dim,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        self.num_inference_steps = num_inference_steps  # number of inference steps for sampling
        self.init_noise_std = init_noise_std
        self.relative_dims = relative_dims
        self.register_buffer("relative_scale", torch.as_tensor(
            list(relative_scale) if relative_scale is not None else [1.0] * relative_dims,
            dtype=torch.float32), persistent=False)   # rebuilt from the cfg: old checkpoints load

    # ========= inference  ============
    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        # keyword arguments to scheduler.step
        **kwargs,
    ):
        model = self.model

        # Start from scaled Gaussian noise (std controls multi-modal diversity)
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        ) * self.init_noise_std

        time_steps = torch.linspace(0, 1.0, self.num_inference_steps + 1)
        for i in range(self.num_inference_steps):

            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. compute next sample
            # Midpoint ODE Solver x_end = x_t + (t_end - t_start) * f(x_mid, t_mid)
            t_start = time_steps[i].view(1).expand(trajectory.shape[0])
            t_start = t_start.to(self.device)
            t_end = time_steps[i + 1].view(1).expand(trajectory.shape[0])
            t_end = t_end.to(self.device)
            t_start_expa = t_start.view(-1, 1, 1)
            t_end_expa = t_end.view(-1, 1, 1)
            t_mid = t_start + (t_end - t_start) / 2
            trajectory_mid = (
                trajectory
                + model(trajectory, t_start, local_cond=local_cond, global_cond=global_cond)
                * (t_end_expa - t_start_expa)
                / 2
            )
            trajectory = trajectory + (t_end_expa - t_start_expa) * model(
                trajectory_mid, t_mid, local_cond=local_cond, global_cond=global_cond
            )
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def _to_relative(self, nactions):
        d, To = self.relative_dims, self.n_obs_steps
        rel = (nactions[..., :d] - nactions[:, To - 1:To, :d]) * self.relative_scale
        return torch.cat([rel, nactions[..., d:]], dim=-1)

    def _from_relative(self, nsample, a0):
        d = self.relative_dims
        p = self.normalizer["action"].params_dict
        na0 = a0.to(nsample.dtype) * p["scale"][:d] + p["offset"][:d]
        pose = nsample[..., :d] / self.relative_scale + na0[:, None]
        return torch.cat([pose, nsample[..., d:]], dim=-1)

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], a0=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        a0: (B, relative_dims) current absolute pose, required when relative_dims > 0
        result: must include "action" key
        """
        assert "past_action" not in obs_dict  # not implemented yet
        # print("!!obs_dict", obs_dict["head_cam"].shape)
        # normalize input
        nobs = self._normalize_obs(obs_dict)
        # print("!!nobs", nobs["head_cam"].shape)
        value = next(iter(nobs.values()))
        B = value.shape[0]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None  # used in p(A_t/O_t) policy
        if self.obs_as_global_cond:  # p(A_t/O_t) policy
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:  # p(A_t, O_t)
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs,
        )

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        if self.relative_dims:
            assert a0 is not None, "relative_dims > 0: pass the current pose as a0"
            naction_pred = self._from_relative(naction_pred, a0)
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {"action": action, "action_pred": action_pred}
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])  # batch_size * horizon * action_dim
        if self.relative_dims:
            nactions = self._to_relative(nactions)
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Start from scaled noise (matches inference init_noise_std)
        noise = torch.randn(trajectory.shape, device=trajectory.device) * self.init_noise_std

        # compute loss mask
        loss_mask = ~condition_mask

        # Sample timesteps
        timesteps = torch.rand(trajectory.shape[0], device=trajectory.device)
        timesteps_exp = timesteps.view(-1, 1, 1)
        # Sample middle trajectory
        trajectory_inter = (1 - timesteps_exp) * noise + timesteps_exp * trajectory
        # Ture flow vector
        vertor_fm_true = trajectory - noise
        # Predict the noise residual
        vector_fm_pred = self.model(trajectory_inter, timesteps, local_cond=local_cond, global_cond=global_cond)
        # Compute the loss
        loss = F.mse_loss(vector_fm_pred, vertor_fm_true, reduction="none")
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss.mean()
        return loss
