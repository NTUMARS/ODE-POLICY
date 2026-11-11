# Adapted for ODE-POLICY public integration; see THIRD_PARTY_NOTICES.md.
"""Single-query GR00T context reader and a causal SONIC latent ODE head.

Only PyTorch is imported here. The caller first loads the official GR00T
checkpoint, then installs this head. The VLM stays outside this module and is
frozen by the caller. No action labels, proprioception, or initial body token
enter the context reader. This is a control-step ODE in SONIC latent space,
not a six-dimensional Cartesian reference tracker or flow matching.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SonicODEConfig:
    horizon: int = 16
    body_dim: int = 64
    hand_dim: int = 12
    state_dim: int = 44
    feature_dim: int = 1024
    cond_dim: int = 512
    width: int = 512
    layers: int = 4
    field_scale: float = 1.0

    def __post_init__(self):
        for name in (
            "horizon",
            "body_dim",
            "hand_dim",
            "state_dim",
            "feature_dim",
            "cond_dim",
            "width",
            "layers",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.horizon > 16:
            raise ValueError("This H16 version cannot generate more than 16 control steps")
        if (self.body_dim, self.hand_dim, self.state_dim) != (64, 12, 44):
            raise ValueError("This schema requires body64, continuous hands12, and state44")
        if self.layers != 4 or self.field_scale != 1.0:
            raise ValueError("This version requires four FiLM layers and field_scale=1")


def _linear32(values, layer):
    return F.linear(
        values.float(),
        layer.weight.float(),
        None if layer.bias is None else layer.bias.float(),
    )


def _finite_tensor(value, name, dimensions):
    if not isinstance(value, torch.Tensor) or value.ndim != dimensions:
        raise ValueError(f"{name} must be a {dimensions}-dimensional tensor")
    if not value.is_floating_point() or not torch.isfinite(value).all():
        raise ValueError(f"{name} must contain finite floating-point values")


def _binary_mask(value, name, shape):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape):
        raise ValueError(f"{name} has an incorrect shape")
    if not torch.all((value == 0) | (value == 1)):
        raise ValueError(f"{name} must contain only zero or one")
    return value.bool()


class ConstantTimeEncoder(nn.Module):
    """Learnable constant AdaLN conditioning, initialized from official t=0.

    The original AlternateVLDiT API still takes a timestep-shaped batch index.
    Its values are ignored: no FM time is sampled, encoded, or integrated.
    """

    def __init__(self, initial_vector):
        super().__init__()
        _finite_tensor(initial_vector, "initial modulation", 2)
        if initial_vector.shape[0] != 1:
            raise ValueError("Initial modulation must have one row")
        self.context_vector = nn.Parameter(initial_vector.detach().clone())

    @classmethod
    def from_native(cls, encoder):
        parameter = next(encoder.parameters(), None)
        if parameter is None:
            raise ValueError("Official timestep encoder has no parameters")
        was_training = encoder.training
        try:
            encoder.eval()
            with torch.no_grad():
                initial = encoder(torch.zeros(1, dtype=torch.long, device=parameter.device))
        finally:
            encoder.train(was_training)
        return cls(initial)

    def forward(self, batch_indices):
        if not isinstance(batch_indices, torch.Tensor) or batch_indices.ndim != 1:
            raise ValueError("The context reader requires one batch index per sample")
        return self.context_vector.expand(batch_indices.shape[0], -1)


class FiLMLayer(nn.Module):
    def __init__(self, width, cond_dim):
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.modulation = nn.Linear(cond_dim, 2 * width)

    def forward(self, hidden, modulation):
        scale, shift = modulation
        return hidden + F.silu(_linear32(hidden, self.linear) * (1 + scale) + shift)


class SonicLatentODE(nn.Module):
    """FP32 Euler rollout, with one step from previous body token to target t."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.context_norm = nn.LayerNorm(config.feature_dim)
        self.context_proj = nn.Linear(config.feature_dim, config.cond_dim)
        self.input_proj = nn.Linear(config.body_dim + config.state_dim, config.width)
        self.layers = nn.ModuleList([FiLMLayer(config.width, config.cond_dim) for _ in range(config.layers)])
        self.output_proj = nn.Linear(config.width, config.body_dim)
        self.hand_hidden = nn.Linear(config.cond_dim, config.cond_dim)
        self.hand_output = nn.Linear(config.cond_dim, config.horizon * config.hand_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def condition(self, feature):
        normalized = F.layer_norm(
            feature.float(),
            self.context_norm.normalized_shape,
            self.context_norm.weight.float(),
            self.context_norm.bias.float(),
        )
        return _linear32(normalized, self.context_proj)

    def field(self, reference, state, modulations):
        hidden = _linear32(torch.cat((reference.float(), state.float()), dim=-1), self.input_proj)
        for layer, modulation in zip(self.layers, modulations):
            hidden = layer(hidden, modulation)
        return self.config.field_scale * _linear32(hidden, self.output_proj)

    def forward(self, feature, state, initial_body, horizon=None):
        horizon = self.config.horizon if horizon is None else horizon
        if type(horizon) is not int or not 1 <= horizon <= self.config.horizon:
            raise ValueError("Requested horizon is outside the configured range")
        _finite_tensor(feature, "context feature", 2)
        _finite_tensor(state, "current state", 2)
        _finite_tensor(initial_body, "initial body", 2)
        batch_size = feature.shape[0]
        if (
            tuple(feature.shape) != (batch_size, self.config.feature_dim)
            or tuple(state.shape) != (batch_size, self.config.state_dim)
            or tuple(initial_body.shape) != (batch_size, self.config.body_dim)
            or state.device != feature.device
            or initial_body.device != feature.device
        ):
            raise ValueError("Context, state, and initial body shapes/devices disagree")
        with torch.autocast(device_type=feature.device.type, enabled=False):
            condition = self.condition(feature)
            modulations = [_linear32(condition, layer.modulation).chunk(2, dim=-1) for layer in self.layers]
            reference = initial_body.float()
            trajectory = []
            for _ in range(horizon):
                # dt=1 recorded control step. No FM-time or Gaussian source exists.
                reference = reference + self.field(reference, state, modulations)
                trajectory.append(reference)
            hands = _linear32(F.silu(_linear32(condition, self.hand_hidden)), self.hand_output)
            hands = hands.reshape(batch_size, self.config.horizon, self.config.hand_dim)
            return torch.stack(trajectory, dim=1), hands[:, :horizon]


class SonicODEHead(nn.Module):
    """A GR00T-native reader plus a new latent ODE and continuous hand decoder.

    ``initial_body`` is supplied by an explicitly causal dataset/runtime: previous
    recorded body token in teacher-forced training, normalized-zero at episode
    start, or last actually dispatched body token during online execution.
    This class deliberately has no target-based initializer or internal carry.
    Cached VLM features are [B,S,2048] for official N1.7, with matching [B,S]
    binary valid/image masks. This cached API does not implement the stock
    GR00T model-level ``action_head(backbone, action_input)``/``get_action``
    interface: its caller must explicitly supply the causal initial body token.
    """

    checkpoint_format = "g1_gr00t_single_query_sonic_ode_v1"

    def __init__(self, native_head, config=None):
        super().__init__()
        self.config = SonicODEConfig() if config is None else config
        if not isinstance(self.config, SonicODEConfig):
            raise TypeError("config must be SonicODEConfig")
        if not bool(getattr(native_head.config, "use_alternate_vl_dit", False)):
            raise ValueError("This version requires the official AlternateVLDiT reader")
        if native_head.config.hidden_size != self.config.feature_dim:
            raise ValueError("ODE feature dimension must match the official DiT output")
        if native_head.model.inner_dim != native_head.config.input_embedding_dim:
            raise ValueError("Official DiT and query embedding dimensions disagree")

        native_shapes = {name: list(value.shape) for name, value in native_head.state_dict().items()}
        retained_prefixes = ("model.", "vlln.", "vl_self_attention.")
        retained_mapping = {
            name: ("reader." + name[len("model.") :] if name.startswith("model.") else name)
            for name in native_shapes
            if name.startswith(retained_prefixes) and not name.startswith("model.timestep_encoder.")
        }
        removed = {name: shape for name, shape in native_shapes.items() if name not in retained_mapping}

        self.reader = native_head.model
        self.vlln = native_head.vlln
        self.vl_self_attention = native_head.vl_self_attention
        self.backbone_dim = int(getattr(native_head.config, "backbone_embedding_dim", self.config.feature_dim))
        self.reader.timestep_encoder = ConstantTimeEncoder.from_native(self.reader.timestep_encoder)
        parameter = next(self.reader.parameters())
        query_dim = native_head.config.input_embedding_dim
        self.query = nn.Parameter(torch.empty(1, 1, query_dim, device=parameter.device, dtype=parameter.dtype))
        nn.init.normal_(self.query, mean=0.0, std=0.02)
        self.ode = SonicLatentODE(self.config).to(device=parameter.device)
        self._installation_manifest = {
            "format": self.checkpoint_format,
            "retained_native_to_ode_keys": retained_mapping,
            "removed_native_weight_shapes": removed,
            "new_modules": ["query", "reader.timestep_encoder.context_vector", "ode"],
            "constant_modulation_initialization": "official_timestep_encoder_at_zero_detached",
            "query_shape": [1, 1, query_dim],
            "cached_backbone_shape": ["batch", "current_valid_or_padded_tokens", self.backbone_dim],
            "reader_transformer_layers": len(self.reader.transformer_blocks),
            "reader_context_uses_proprio": False,
            "reader_context_uses_actions_or_initial_body": False,
            "initial_body_source": "caller_previous_body_or_normalized_zero_BOS",
            "initial_perturbation_probability": 0.0,
            "body_reference_space": "normalized_sonic_latent64",
            "hand_target_offset_frames": 0,
            "hand_prediction": "continuous_normalized_state_no_sigmoid_no_residual",
            "euler_dt_control_steps": 1.0,
        }

    def configuration_dict(self):
        return asdict(self.config)

    def installation_manifest(self):
        # Return fresh containers so callers cannot mutate the recorded mapping.
        import copy

        return copy.deepcopy(self._installation_manifest)

    def context_cached(self, backbone_output):
        """Read only current image/language tokens; never accepts labels or state."""
        if not hasattr(backbone_output, "__getitem__"):
            raise TypeError("backbone_output must be a mapping or BatchFeature")
        features = backbone_output["backbone_features"]
        _finite_tensor(features, "backbone features", 3)
        batch_size, tokens, _ = features.shape
        if batch_size < 1 or tokens < 1 or features.shape[-1] != self.backbone_dim:
            raise ValueError("Empty or incorrectly dimensioned backbone token batches are invalid")
        valid = _binary_mask(
            backbone_output["backbone_attention_mask"],
            "backbone attention mask",
            (batch_size, tokens),
        )
        image = _binary_mask(backbone_output["image_mask"], "image mask", (batch_size, tokens))
        if valid.device != features.device or image.device != features.device:
            raise ValueError("Backbone features and masks must share a device")
        if not (valid & image).any(dim=1).all() or not (valid & ~image).any(dim=1).all():
            raise ValueError("Each current observation requires valid image and text tokens")
        if "frame_ids" in backbone_output:
            frame_ids = backbone_output["frame_ids"]
            if tuple(frame_ids.shape) != (batch_size, tokens) or not (frame_ids[valid] == 0).all():
                raise ValueError("Only a single current observation slot may enter the ODE reader")

        features = features.to(dtype=self.query.dtype)
        if isinstance(self.vl_self_attention, nn.Identity):
            encoded = self.vlln(features)
            encoded = torch.where(valid[..., None], encoded, torch.zeros_like(encoded))
        else:
            # The native SelfAttentionTransformer has no padding-mask argument.
            # Batch equal-length valid token sequences, never including padding.
            # A typical fixed-instruction camera batch uses one grouped call.
            lengths = valid.sum(dim=1)
            max_tokens = int(lengths.max().item())
            encoded = features.new_zeros(batch_size, max_tokens, features.shape[-1])
            compact_image = torch.zeros(batch_size, max_tokens, dtype=torch.bool, device=features.device)
            for length_value in lengths.unique(sorted=True).tolist():
                rows = torch.nonzero(lengths == length_value, as_tuple=False).flatten()
                group_valid = valid.index_select(0, rows)
                compact = features.index_select(0, rows)[group_valid]
                compact = compact.reshape(rows.shape[0], length_value, features.shape[-1])
                group_encoded = self.vl_self_attention(self.vlln(compact))
                padding = max_tokens - length_value
                encoded = encoded.index_copy(0, rows, F.pad(group_encoded, (0, 0, 0, padding)))
                group_image = image.index_select(0, rows)[group_valid]
                group_image = group_image.reshape(rows.shape[0], length_value)
                compact_image = compact_image.index_copy(0, rows, F.pad(group_image, (0, padding), value=False))
            image = compact_image
            valid = torch.arange(max_tokens, device=features.device)[None] < lengths[:, None]
        batch_indices = torch.zeros(batch_size, dtype=torch.long, device=features.device)
        context = self.reader(
            hidden_states=self.query.expand(batch_size, -1, -1),
            encoder_hidden_states=encoded,
            timestep=batch_indices,
            encoder_attention_mask=valid,
            image_mask=image,
            backbone_attention_mask=valid,
        )
        if isinstance(context, tuple):
            context = context[0]
        if tuple(context.shape) != (batch_size, 1, self.config.feature_dim):
            raise ValueError("Official reader returned an unexpected query shape")
        if not torch.isfinite(context).all():
            raise FloatingPointError("Nonfinite ODE context")
        return context[:, 0].float()

    def predict_cached(self, backbone_output, state, initial_body, horizon=None):
        """Differentiable prediction; inference callers should use torch.no_grad()."""
        if isinstance(state, torch.Tensor) and state.ndim == 3 and state.shape[1] == 1:
            state = state[:, 0]
        context = self.context_cached(backbone_output)
        body, hands = self.ode(context, state, initial_body, horizon)
        action = torch.cat((body, hands), dim=-1)
        if not torch.isfinite(action).all():
            raise FloatingPointError("Nonfinite ODE action prediction")
        return {"action_pred": action, "body_pred": body, "hand_pred": hands, "context": context}

    def forward_cached(self, backbone_output, state, initial_body, target, action_mask):
        _finite_tensor(target, "action target", 3)
        if target.shape[-1] != self.config.body_dim + self.config.hand_dim:
            raise ValueError("Action targets require exactly 76 dimensions")
        prediction = self.predict_cached(backbone_output, state, initial_body, target.shape[1])
        predicted = prediction["action_pred"]
        if target.shape != predicted.shape or target.device != predicted.device:
            raise ValueError("Target and prediction shapes/devices disagree")
        if isinstance(action_mask, torch.Tensor) and action_mask.ndim == 2:
            mask = _binary_mask(action_mask, "action mask", target.shape[:2])[..., None].expand_as(target)
        else:
            mask = _binary_mask(action_mask, "action mask", target.shape)
        if mask.device != predicted.device:
            raise ValueError("Action masks must share the prediction device")
        error = predicted.float() - target.float()
        squared, absolute = error.square(), error.abs()

        def mean(values, start, stop):
            selected_mask = mask[..., start:stop].to(dtype=torch.float32)
            return (values[..., start:stop] * selected_mask).sum() / selected_mask.sum().clamp_min(1)

        body_loss = mean(squared, 0, 64)
        hand_loss = mean(squared, 64, 76)
        result = {
            **prediction,
            "loss": (64 * body_loss + 12 * hand_loss) / 76,
            "body_loss": body_loss.detach(),
            "hand_loss": hand_loss.detach(),
            "body_mae_normalized": mean(absolute, 0, 64).detach(),
            "left_hand_loss": mean(squared, 64, 70).detach(),
            "right_hand_loss": mean(squared, 70, 76).detach(),
            "left_hand_mae_normalized": mean(absolute, 64, 70).detach(),
            "right_hand_mae_normalized": mean(absolute, 70, 76).detach(),
            # Fixed hand physical [0,1] maps affinely to normalized [-1,1].
            "left_hand_mae_physical": (0.5 * mean(absolute, 64, 70)).detach(),
            "right_hand_mae_physical": (0.5 * mean(absolute, 70, 76)).detach(),
            "valid_body_values": mask[..., :64].sum().detach(),
            "valid_hand_values": mask[..., 64:76].sum().detach(),
        }
        if not torch.isfinite(result["loss"]):
            raise FloatingPointError("Nonfinite ODE supervised loss")
        return result

    def forward(self, backbone_output, state, initial_body, target, action_mask):
        return self.forward_cached(backbone_output, state, initial_body, target, action_mask)


def install_sonic_ode(native_head, config=None):
    """Install after official weight loading; the original head is consumed.

    Retained module objects and weights are reused without reset. The original
    FM action/state encoders, action decoder, positions, and timestep network
    are not registered in the returned head. Strict reload is done by creating
    this same architecture and calling ``load_state_dict(..., strict=True)``.
    """
    return SonicODEHead(native_head, config)
