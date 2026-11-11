"""Shared cached/RGB policy used by offline inference and native deployment."""

from __future__ import annotations
import re
import numpy as np
import torch
from .carry import ExecutedBodyCarry, decode_actions
from .features import FeatureBatch


def state44_from_body29(body29, left6, right6, projected_gravity3):
    """Concatenate controller feedback already in the checkpoint's body29 order.

    The caller must supply the matching MuJoCo joint order; this function does
    not infer/reorder hardware motor IDs or convert units.
    """
    body = np.asarray(body29, np.float32)
    left = np.asarray(left6, np.float32)
    right = np.asarray(right6, np.float32)
    gravity = np.asarray(projected_gravity3, np.float32)
    if body.shape != (29,) or left.shape != (6,) or right.shape != (6,) or gravity.shape != (3,):
        raise ValueError("Expected body29/left6/right6/gravity3 feedback")
    out = np.r_[body, left, right, gravity].astype(np.float32)
    if not np.isfinite(out).all() or np.any((np.r_[left, right] < 0) | (np.r_[left, right] > 1)):
        raise ValueError("Invalid physical feedback/hand units")
    return out


def state44_from_source43(body43, left6, right6, projected_gravity3):
    """Project the original LeRobot state43 layout, then use body29 feedback.

    Source columns 0:22 and 29:36 contain the same ordered 29 body joints.
    Do not apply this source-layout projection to an existing 29-joint vector.
    """
    body = np.asarray(body43, np.float32)
    if body.shape != (43,):
        raise ValueError("Expected source-layout body43")
    return state44_from_body29(np.r_[body[:22], body[29:36]], left6, right6, projected_gravity3)


def sonic_token_fields(motion_token64, frame_index):
    """Return only token_state/frame_index arrays accepted by SONIC's pose packer.

    The native transport serializes these with SONIC Protocol v4. DFX6 hands
    travel separately over DDS, not through legacy 7-joint hand fields.
    """
    token = np.asarray(motion_token64, np.float32)
    if token.shape != (64,) or not np.isfinite(token).all() or type(frame_index) is not int or frame_index < 0:
        raise ValueError("Finite token64 and nonnegative frame index required")
    return {"token_state": token[None].copy(), "frame_index": np.asarray([frame_index], np.int64)}


class RGBEncoder:
    """Use the external official processor/transform on caller-supplied current uint8 RGB.

    Two camera arrays must correspond to the same observation. This facade does
    not capture, synchronize or communicate with hardware. The training transform,
    chat template and FP16 cache quantization are reproduced before head input.
    """

    def __init__(
        self,
        model,
        *,
        camera_keys=("observation.images.ego_view", "observation.images.chest"),
        device="cuda",
        eval_transform=None,
    ):
        self.model = model
        self.camera_keys = tuple(camera_keys)
        manifest = getattr(model, "_ode_policy_manifest", None)
        if manifest and self.camera_keys != tuple(manifest["binding"]["camera_keys"]):
            raise ValueError("Camera order differs from the policy manifest")
        self.device = torch.device(device)
        if len(self.camera_keys) != 2 or len(set(self.camera_keys)) != 2:
            raise ValueError("Two distinct ordered cameras required")
        if eval_transform is None:
            from gr00t.model.gr00t_n1d7.image_augmentations import build_image_transformations_albumentations

            cfg = model.config
            if not getattr(cfg, "use_albumentations_transforms", True):
                raise ValueError("Official Albumentations evaluation transform required")
            _, eval_transform = build_image_transformations_albumentations(
                **{
                    k: getattr(cfg, k)
                    for k in ["image_target_size", "image_crop_size", "shortest_image_edge", "crop_fraction"]
                },
                random_rotation_angle=getattr(cfg, "random_rotation_angle", None),
                color_jitter_params=getattr(cfg, "color_jitter_params", None),
                extra_augmentation_config=getattr(cfg, "extra_augmentation_config", None),
            )
        self.transform = eval_transform
        self.model.backbone.requires_grad_(False).eval()

    def __call__(self, images, instruction):
        from PIL import Image

        if set(images) != set(self.camera_keys) or not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Provide exactly the configured current cameras and nonempty instruction")
        views = []
        for key in self.camera_keys:
            raw = np.asarray(images[key])
            if raw.dtype != np.uint8 or raw.ndim != 3 or raw.shape[-1] != 3:
                raise ValueError("RGB images must be uint8 HWC; BGR must be converted by caller")
            image = self.transform(image=raw)["image"]
            if image.dtype == np.float32:
                image = (image * 255).astype(np.uint8)
            elif image.dtype != np.uint8:
                raise ValueError("Unexpected transformed image dtype")
            views.append(Image.fromarray(image))
        text = re.sub(r"[^\w\s]", "", instruction.lower()) if self.model.config.formalize_language else instruction
        conversation = [
            {
                "role": "user",
                "content": [*[{"type": "image", "image": v} for v in views], {"type": "text", "text": text}],
            }
        ]
        prompt = self.model.collator.processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False
        )
        inputs = self.model.collator(
            [{"vlm_content": {"text": prompt, "images": views, "conversation": conversation}}]
        )["inputs"]
        parameter = next(self.model.backbone.parameters(), None)
        dtype = parameter.dtype if parameter is not None else torch.float32
        prepared = {k: v.to(self.device, dtype=dtype if v.is_floating_point() else v.dtype) for k, v in inputs.items()}
        with torch.inference_mode():
            encoded = self.model.backbone(self.model.backbone.prepare_input(prepared))
        f = encoded["backbone_features"]
        valid = encoded["backbone_attention_mask"].bool()
        image = encoded["image_mask"].bool()
        if (
            f.ndim != 3
            or f.shape[0] != 1
            or valid.shape != f.shape[:2]
            or image.shape != valid.shape
            or not torch.isfinite(f).all()
            or not (valid & image).any()
        ):
            raise ValueError("Invalid current RGB backbone output")
        mask = valid[0]
        count = int(mask.sum())
        return FeatureBatch(
            backbone_features=f[:, mask].detach().cpu().to(torch.float16).float(),
            backbone_attention_mask=torch.ones((1, count), dtype=torch.bool),
            image_mask=image[:, mask].cpu(),
            frame_ids=torch.zeros((1, count), dtype=torch.long),
        )


class ODEPolicy:
    """Preview one chunk; commit only a prefix the external caller actually sent."""

    def __init__(self, head, normalizer, *, device="cpu", rgb_encoder=None):
        self.head = head
        self.normalizer = normalizer
        self.device = torch.device(device)
        self.rgb_encoder = rgb_encoder
        self.carry = ExecutedBodyCarry(head.config.horizon)

    def reset_episode(self):
        self.carry.reset()

    def initialize_last_sent_body(self, body64):
        body = np.asarray(body64, np.float32)
        if body.shape != (64,) or not np.isfinite(body).all():
            raise ValueError("Finite last-sent physical body64 required")
        if self.carry.executed_steps:
            raise ValueError("Reset episode before initializing its external history")
        self.carry.commit_executed(np.r_[body, np.zeros(12, np.float32)][None])

    def predict_cached(self, condition, physical_state44, *, initial_body=None):
        values = np.asarray(physical_state44, np.float32)
        if values.shape != (44,) or not np.isfinite(values).all():
            raise ValueError("Finite physical state44 required")
        if np.any((values[29:41] < 0) | (values[29:41] > 1)):
            raise ValueError("Measured hand state must be in [0,1]")
        state = torch.from_numpy(self.normalizer.normalize_state(values)[None, None]).to(self.device)
        r0 = (
            self.carry.normalized_initial_body(self.normalizer)
            if initial_body is None
            else np.asarray(initial_body, np.float32)
        )
        if r0.shape != (64,) or not np.isfinite(r0).all():
            raise ValueError("Finite normalized initial body64 required")
        initial = torch.from_numpy(r0[None].copy()).to(self.device)
        batch = {k: torch.as_tensor(v).to(self.device) for k, v in condition.items()}
        self.head.eval()
        with torch.inference_mode():
            result = self.head.predict_cached(batch, state, initial)["action_pred"]
        return decode_actions(result.float().cpu().numpy(), self.normalizer)[0]

    def predict_rgb(self, images, instruction, physical_state44, *, initial_body=None):
        if self.rgb_encoder is None:
            raise ValueError("A current-RGB encoder must be explicitly supplied")
        r0 = self.carry.normalized_initial_body(self.normalizer) if initial_body is None else initial_body
        return self.predict_cached(self.rgb_encoder(images, instruction), physical_state44, initial_body=r0)

    def commit_sent_prefix(self, actual_sent_actions76):
        """Explicit external execution acknowledgement; prediction never calls this."""
        self.carry.commit_executed(actual_sent_actions76)
