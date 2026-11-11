"""Two-camera RGB and continuous LIBERO action/state preprocessing."""
import torch
from torch.nn import functional as F

from .normalizer import LinearNormalizer


class ImageWAMProcessor:
    def __init__(self, shape_meta, num_obs_steps=13, num_output_cameras=2,
                 action_output_dim=7, proprio_output_dim=8, image_obs_steps=2,
                 action_state_transforms=None, use_stepwise_action_norm=False,
                 norm_default_mode="min/max", norm_exception_mode=None,
                 action_state_merger=None, train_transforms=None, val_transforms=None,
                 delta_action_dim_mask=None):
        if action_state_transforms is not None or use_stepwise_action_norm or norm_default_mode != "min/max" or norm_exception_mode:
            raise ValueError("LIBERO uses raw state/actions with global min/max normalization")
        if (int(num_output_cameras), int(action_output_dim), int(proprio_output_dim)) != (2, 7, 8):
            raise ValueError("LIBERO requires two cameras, action7 and state8")
        if len(shape_meta["images"]) != 2:
            raise ValueError("Configure exactly two ordered camera fields")
        self.shape_meta = shape_meta
        self.num_obs_steps, self.image_obs_steps = int(num_obs_steps), int(image_obs_steps)
        self._normalizer = None
        self._is_train = False
        self.train_transforms, self.val_transforms = train_transforms, val_transforms
        self.delta_mask = torch.tensor([True] * 6 + [False])
        if delta_action_dim_mask is not None and list(delta_action_dim_mask["default"]) != self.delta_mask.tolist():
            raise ValueError("Only the six pose channels are delta-action channels")

    @property
    def normalizer(self):
        if self._normalizer is None:
            raise ValueError("Set normalization statistics before processing data")
        return self._normalizer

    def set_normalizer_from_stats(self, stats):
        self._normalizer = LinearNormalizer(stats)

    def train(self):
        self._is_train = True
        return self

    def eval(self):
        self._is_train = False
        return self

    def action_state_transform(self, batch):
        for field, width in (("state", 8), ("action", 7)):
            if field in batch and batch[field]["default"].shape[-1] != width:
                raise ValueError(f"Expected {field} width {width}")
        return batch

    def process_images(self, images, expected_steps=None):
        expected_steps = self.image_obs_steps if expected_steps is None else int(expected_steps)
        pixels = []
        for meta in self.shape_meta["images"]:
            image = torch.as_tensor(images[meta["key"]])
            if image.ndim != 4 or image.shape[1] != 3 or image.shape[0] != expected_steps:
                raise ValueError("Camera data must have shape [image_steps,3,H,W]")
            transforms = self.train_transforms if self._is_train else self.val_transforms
            transforms = transforms[meta["key"]] if isinstance(transforms, dict) else transforms
            if transforms is None:
                image = image.float() / 255 if image.dtype == torch.uint8 else image.float()
                image = F.interpolate(image, size=list(meta["shape"])[1:], mode="bilinear", align_corners=False, antialias=True)
            else:
                for transform in transforms:
                    image = transform(image)
            if list(image.shape[1:]) != list(meta["shape"]) or not torch.is_floating_point(image):
                raise ValueError("Camera transforms must return float RGB with the configured shape")
            pixels.append(image.float())
        return torch.stack(pixels)

    @staticmethod
    def pack_video(pixels, video_size):
        if pixels.ndim != 5 or pixels.shape[0] != 2 or pixels.shape[2] != 3:
            raise ValueError("Pack two ordered [camera,time,3,height,width] RGB views")
        video = torch.cat((pixels[0], pixels[1]), dim=-1).float()
        video_size = tuple(int(value) for value in video_size)
        if tuple(video.shape[-2:]) != video_size:
            height, width = video.shape[-2:]
            scale = max(video_size[0] / height, video_size[1] / width)
            video = F.interpolate(video, size=(round(height * scale), round(width * scale)),
                                  mode="bilinear", align_corners=False, antialias=True)
            top, left = (video.shape[-2] - video_size[0]) // 2, (video.shape[-1] - video_size[1]) // 2
            video = video[..., top:top + video_size[0], left:left + video_size[1]]
        return (video * 2 - 1).permute(1, 0, 2, 3)

    def preprocess(self, data):
        pixels = self.process_images(data["images"])
        if "action" in data:
            actions = data["action"]["default"].clone()
            actions[torch.as_tensor(data["action_is_pad"]).bool(), :6] = 0
            data["action"] = {"default": actions}
        data = self.normalizer.forward(self.action_state_transform(data))
        result = {"pixel_values": pixels, "proprio": data["state"]["default"],
                  "proprio_is_pad": data["state_is_pad"], "image_is_pad": data["image_is_pad"],
                  "proprio_dim_is_pad": torch.zeros(8, dtype=torch.bool),
                  "instruction": str(data["task"]), "idx": data["idx"]}
        if "action" in data:
            result.update(action=data["action"]["default"], action_is_pad=data["action_is_pad"],
                          action_dim_is_pad=torch.zeros(7, dtype=torch.bool))
        for key in ("episode_index", "dataset_index", "task_index"):
            if key in data:
                result[key] = data[key]
        return result
