"""Temporally consistent image augmentation for LIBERO camera clips.

VideoAugmentation is adapted under MIT; see THIRD_PARTY_NOTICES.md.
"""
import torch
from torch import nn
import torchvision.transforms as TF
import torchvision.transforms.functional as F


class ToTensor(nn.Module):
    def forward(self, value):
        return value.float() / 255 if value.dtype == torch.uint8 else value.float()


class ConcatLeftAlign:
    def set_shape_meta(self, shape_meta):
        self.shape_meta = shape_meta

    def forward(self, batch):
        for field in ("state", "action"):
            if field in batch:
                batch[field] = torch.cat([batch[field][meta["key"]] for meta in self.shape_meta[field]], dim=-1)
        return batch


class VideoAugmentation(nn.Module):
    """Weak train-time augmentation for a camera clip in float [0, 1].

    The same sampled parameters are applied to every frame in the clip to keep
    temporal consistency within a camera view.
    """

    def __init__(
        self,
        p: float = 1.0,
        augment_types: list[str] | tuple[str, ...] = ("corrupt_only", "color_only", "both"),
        color_jitter: dict | None = None,
        gamma: dict | None = None,
        exposure: dict | None = None,
        gaussian_noise: dict | None = None,
        random_resized_crop: dict | None = None,
        rotate: dict | None = None,
    ):
        super().__init__()
        self.p = float(p)
        self.augment_types = tuple(augment_types)
        self.color_jitter = color_jitter or {}
        self.gamma = gamma or {}
        self.exposure = exposure or {}
        self.gaussian_noise = gaussian_noise or {}
        self.random_resized_crop = random_resized_crop or {}
        self.rotate = rotate or {}

    @staticmethod
    def _rand(device: torch.device) -> torch.Tensor:
        return torch.rand((), device=device)

    @staticmethod
    def _uniform(device: torch.device, low: float, high: float) -> float:
        return float(torch.empty((), device=device).uniform_(float(low), float(high)).item())

    def _sample_augment_type(self) -> str:
        if not self.augment_types:
            return "both"
        index = int(torch.randint(0, len(self.augment_types), (), device=torch.device("cpu")).item())
        return str(self.augment_types[index])

    def _apply_color_jitter(self, frames: torch.Tensor) -> torch.Tensor:
        cfg = self.color_jitter
        if not cfg:
            return frames

        ops = []
        brightness = float(cfg.get("brightness", 0.0))
        if brightness > 0:
            factor = self._uniform(frames.device, max(0.0, 1.0 - brightness), 1.0 + brightness)
            ops.append(lambda img, factor=factor: F.adjust_brightness(img, factor))

        contrast = float(cfg.get("contrast", 0.0))
        if contrast > 0:
            factor = self._uniform(frames.device, max(0.0, 1.0 - contrast), 1.0 + contrast)
            ops.append(lambda img, factor=factor: F.adjust_contrast(img, factor))

        saturation = float(cfg.get("saturation", 0.0))
        if saturation > 0:
            factor = self._uniform(frames.device, max(0.0, 1.0 - saturation), 1.0 + saturation)
            ops.append(lambda img, factor=factor: F.adjust_saturation(img, factor))

        hue = float(cfg.get("hue", 0.0))
        if hue > 0:
            factor = self._uniform(frames.device, -min(hue, 0.5), min(hue, 0.5))
            ops.append(lambda img, factor=factor: F.adjust_hue(img, factor))

        if len(ops) > 1:
            order = torch.randperm(len(ops)).tolist()
            ops = [ops[i] for i in order]
        for op in ops:
            frames = op(frames)
        return frames

    def _apply_gamma(self, frames: torch.Tensor) -> torch.Tensor:
        cfg = self.gamma
        if not cfg:
            return frames
        gamma_range = cfg.get("range", [0.9, 1.1])
        gamma = self._uniform(frames.device, float(gamma_range[0]), float(gamma_range[1]))
        return F.adjust_gamma(frames.clamp(0.0, 1.0), gamma=gamma)

    def _apply_exposure(self, frames: torch.Tensor) -> torch.Tensor:
        cfg = self.exposure
        if not cfg:
            return frames
        ev_range = cfg.get("ev_range", cfg.get("range", [-0.1, 0.1]))
        ev = self._uniform(frames.device, float(ev_range[0]), float(ev_range[1]))
        return frames * (2.0 ** ev)

    def _apply_random_resized_crop(self, frames: torch.Tensor) -> torch.Tensor:
        cfg = self.random_resized_crop
        if not cfg:
            return frames

        _, _, height, width = frames.shape
        scale = tuple(float(v) for v in cfg.get("scale", [0.95, 1.0]))
        ratio = cfg.get("ratio", "preserve")
        if ratio == "preserve":
            side_scale = self._uniform(frames.device, scale[0], scale[1])
            crop_h = max(1, min(height, int(round(height * side_scale))))
            crop_w = max(1, min(width, int(round(width * side_scale))))
            top = int(torch.randint(0, height - crop_h + 1, (), device=frames.device).item())
            left = int(torch.randint(0, width - crop_w + 1, (), device=frames.device).item())
            return F.resized_crop(
                frames,
                top=top,
                left=left,
                height=crop_h,
                width=crop_w,
                size=[height, width],
                interpolation=F.InterpolationMode.BILINEAR,
                antialias=True,
            )

        ratio = tuple(float(v) for v in ratio)
        top, left, crop_h, crop_w = TF.RandomResizedCrop.get_params(frames[0], scale=scale, ratio=ratio)
        return F.resized_crop(
            frames,
            top=top,
            left=left,
            height=crop_h,
            width=crop_w,
            size=[height, width],
            interpolation=F.InterpolationMode.BILINEAR,
            antialias=True,
        )

    def _apply_rotate(self, frames: torch.Tensor) -> torch.Tensor:
        cfg = self.rotate
        if not cfg:
            return frames
        degrees = cfg.get("degrees", 5.0)
        if isinstance(degrees, (list, tuple)):
            angle = self._uniform(frames.device, float(degrees[0]), float(degrees[1]))
        else:
            max_degrees = float(degrees)
            angle = self._uniform(frames.device, -max_degrees, max_degrees)
        fill = cfg.get("fill", "mean")
        if fill == "mean":
            fill = float(frames.mean().item())
        return F.rotate(
            frames,
            angle=angle,
            interpolation=F.InterpolationMode.BILINEAR,
            fill=fill,
        )

    def _apply_gaussian_noise(self, frames: torch.Tensor) -> torch.Tensor:
        cfg = self.gaussian_noise
        if not cfg:
            return frames
        std = float(cfg.get("std", 0.01))
        noise = torch.randn((1,) + tuple(frames.shape[1:]), device=frames.device, dtype=frames.dtype) * std
        return frames + noise

    @staticmethod
    def _validate_input_range(frames: torch.Tensor) -> None:
        min_value = float(frames.min().item())
        max_value = float(frames.max().item())
        eps = 1e-6
        if min_value < -eps or max_value > 1.0 + eps:
            raise ValueError(
                "`VideoAugmentation` expects input frames in [0, 1] before augmentation, "
                f"got min={min_value:.6f}, max={max_value:.6f}."
            )

    def _augment_frames(self, frames: torch.Tensor) -> torch.Tensor:
        self._validate_input_range(frames)
        aug_type = self._sample_augment_type()
        if aug_type in {"color_only", "both"}:
            frames = self._apply_color_jitter(frames)
            frames = self._apply_gamma(frames)
            frames = self._apply_exposure(frames)
        if aug_type in {"corrupt_only", "both"}:
            frames = self._apply_gaussian_noise(frames)
        frames = self._apply_random_resized_crop(frames)
        frames = self._apply_rotate(frames)
        return frames.clamp(0.0, 1.0)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim == 5:
            if not torch.is_floating_point(frames):
                raise TypeError(f"`VideoAugmentation` expects floating frames, got {frames.dtype}")
            if self._rand(torch.device("cpu")).item() >= self.p:
                return frames
            augmented = frames.clone()
            for cam_idx in range(frames.shape[0]):
                augmented[cam_idx] = self._augment_frames(frames[cam_idx])
            return augmented

        squeeze_time = False
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)
            squeeze_time = True
        if frames.ndim != 4:
            raise ValueError(f"`VideoAugmentation` expects [T,C,H,W], got {tuple(frames.shape)}")
        if not torch.is_floating_point(frames):
            raise TypeError(f"`VideoAugmentation` expects floating frames, got {frames.dtype}")
        if self._rand(torch.device("cpu")).item() >= self.p:
            return frames.squeeze(0) if squeeze_time else frames

        frames = self._augment_frames(frames)
        return frames.squeeze(0) if squeeze_time else frames
