"""Native full-frame LeRobot v2 LIBERO dataset with aligned ODE references."""
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from .data.normalizer import load_dataset_stats_from_json
from .data.processor import ImageWAMProcessor
from .data.video import decode_video_frames
from .sidecar import SidecarStore, sha256_file

DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


class ODERobotVideoDataset(torch.utils.data.Dataset):
    def __init__(self, ode_sidecar_dir, dataset_dirs, shape_meta, num_frames=13,
                 video_size=(224, 448), camera_key=None, processor=None,
                 pretrained_norm_stats=None, val_set_proportion=0.0, is_training_set=True,
                 global_sample_stride=1, past_action_size=0, action_video_freq_ratio=1,
                 skip_padding_as_possible=False, concat_multi_camera="horizontal",
                 video_augmentation=None, require_text_cache=False, endpoint_frames_only=True,
                 qwen_text_cache_dir=None, qwen_context_len=128,
                 qwen_text_cache_format="qwen3_flux2", text_embedding_cache_dir=None,
                 context_len=128, lerobot_video_backend="pyav", video_reader=None):
        if (int(global_sample_stride), int(past_action_size), int(action_video_freq_ratio)) != (1, 0, 1):
            raise ValueError("LIBERO ODE requires stride 1, no past-action window and matched control/video rate")
        if val_set_proportion or not is_training_set or skip_padding_as_possible:
            raise ValueError("This recipe uses every frame without validation splitting or padding retries")
        if camera_key is not None or concat_multi_camera != "horizontal" or not endpoint_frames_only:
            raise ValueError("Use two horizontally concatenated camera views at the current/future endpoints")
        if require_text_cache or text_embedding_cache_dir is not None:
            raise ValueError("Use the Qwen3 cache, not a legacy T5 text cache")
        if qwen_text_cache_format != "qwen3_flux2" or lerobot_video_backend != "pyav":
            raise ValueError("Supported data formats are Qwen3-FLUX2 cache and PyAV video")
        self.ode_store = SidecarStore(ode_sidecar_dir, dataset_dirs)
        self.ode_meta = self.ode_store.metadata
        if pretrained_norm_stats is None:
            pretrained_norm_stats = self.ode_store.root / "dataset_stats.json"
        if sha256_file(pretrained_norm_stats) != self.ode_meta["normalization_stats_sha256"]:
            raise ValueError("Normalization statistics differ from the prepared sidecar")
        self.dataset_dirs = [Path(directory).expanduser().resolve() for directory in dataset_dirs]
        self.num_frames, self.horizon = int(num_frames), int(num_frames) - 1
        if self.horizon < 1:
            raise ValueError("num_frames must include a current frame and a future endpoint")
        self.video_size = tuple(int(value) for value in video_size)
        self.shape_meta = shape_meta
        self.processor = processor or ImageWAMProcessor(shape_meta, num_obs_steps=self.num_frames)
        self.processor.set_normalizer_from_stats(load_dataset_stats_from_json(pretrained_norm_stats))
        self.processor.train()
        self.qwen_text_cache_dir = Path(qwen_text_cache_dir).expanduser() if qwen_text_cache_dir else None
        self.qwen_context_len = int(qwen_context_len)
        self._text_cache = OrderedDict()
        self.video_reader = video_reader or decode_video_frames
        if video_augmentation is not None and not isinstance(video_augmentation, torch.nn.Module):
            from hydra.utils import instantiate
            video_augmentation = instantiate(video_augmentation)
        self.video_augmentation = video_augmentation
        self._root_info = [json.loads((root / "meta/info.json").read_text()) for root in self.dataset_dirs]
        self._tasks = []
        for root in self.dataset_dirs:
            records = [json.loads(line) for line in (root / "meta/tasks.jsonl").read_text().splitlines() if line.strip()]
            tasks = {int(record["task_index"]): str(record["task"]) for record in records}
            if len(tasks) != len(records) or any(not task.strip() for task in tasks.values()):
                raise ValueError("Task metadata must contain unique indices and nonempty instructions")
            self._tasks.append(tasks)
        self.episodes = self._read_episodes()
        self._episode_starts = np.asarray([record["start"] for record in self.episodes], dtype=np.int64)

    def _read_episodes(self):
        import pyarrow.parquet as pq
        episodes = []
        for episode in self.ode_meta["episodes"]:
            record = dict(episode)
            slot = record["dataset_slot"]
            path = self.dataset_dirs[slot] / record["parquet"]
            table = pq.read_table(path, columns=["timestamp", "task_index"])
            timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
            tasks = np.asarray(table["task_index"].to_numpy(), dtype=np.int64)
            if len(table) != record["end"] - record["start"] or len(set(tasks.tolist())) != 1:
                raise ValueError("Each numeric episode must match its sidecar and contain one task")
            fps = float(self._root_info[slot]["fps"])
            if not np.isfinite(timestamps).all() or not np.allclose(timestamps, np.arange(len(table)) / fps, atol=1e-4, rtol=0):
                raise ValueError("Episode timestamps must follow the declared video/control rate from zero")
            record.update(timestamps=timestamps, task_index=int(tasks[0]))
            if record["task_index"] not in self._tasks[slot]:
                raise ValueError("Parquet task_index is absent from tasks.jsonl")
            episodes.append(record)
        return episodes

    def __len__(self):
        return int(self.ode_meta["num_frames"])

    def _video_path(self, episode, meta):
        slot = episode["dataset_slot"]
        info = self._root_info[slot]
        key = meta.get("lerobot_key") or f"observation.images.{meta['key']}"
        feature = info.get("features", {}).get(key)
        if feature is not None and feature.get("dtype") != "video":
            raise ValueError(f"Camera feature is not a video: {key}")
        template = info.get("video_path")
        if not template:
            raise ValueError("LeRobot info.json must declare its video_path template")
        relative = template.format(video_key=key, episode_index=episode["episode_index"],
            episode_chunk=episode["episode_index"] // int(info.get("chunks_size", 1000)))
        path = (self.dataset_dirs[slot] / relative).resolve()
        if not path.is_relative_to(self.dataset_dirs[slot]):
            raise ValueError("Video template escapes the dataset root")
        return path

    def _get_cached_qwen_context(self, prompt):
        if self.qwen_text_cache_dir is None:
            raise ValueError("Set qwen_text_cache_dir to the prepared Qwen3 embedding cache")
        if prompt in self._text_cache:
            self._text_cache.move_to_end(prompt)
            return self._text_cache[prompt]
        hashed = hashlib.sha256(prompt.encode()).hexdigest()
        path = self.qwen_text_cache_dir / f"{hashed}.qwen3_flux2_len{self.qwen_context_len}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing Qwen text embedding cache: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        context, mask = payload["text_hidden_states"], payload["text_attention_mask"]
        if context.ndim != 2 or context.shape[0] != self.qwen_context_len or mask.shape != (self.qwen_context_len,):
            raise ValueError("Qwen cache dimensions differ from configured context length")
        if not torch.isfinite(context).all() or not ((mask == 0) | (mask == 1)).all():
            raise ValueError("Qwen cache must contain finite embeddings and a binary attention mask")
        result = context, mask.bool()
        self._text_cache[prompt] = result
        if len(self._text_cache) > 64:
            self._text_cache.popitem(last=False)
        return result

    def __getitem__(self, index):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = self.episodes[int(np.searchsorted(self._episode_starts, index, side="right")) - 1]
        slot, end = episode["dataset_slot"], int(episode["end"])
        state_indices = np.minimum(index + np.arange(self.num_frames), end - 1)
        frame_indices = state_indices[[0, -1]] - episode["start"]
        timestamps = episode["timestamps"][frame_indices].tolist()
        images = {meta["key"]: self.video_reader(self._video_path(episode, meta), timestamps, 1e-3)
                  for meta in self.shape_meta["images"]}
        window = self.ode_store.window(index, self.horizon)
        raw = {"state": {"default": torch.from_numpy(np.array(self.ode_store.states[state_indices], copy=True))},
               "action": {"default": torch.from_numpy(window["commands"].copy())},
               "images": images, "state_is_pad": torch.from_numpy(index + np.arange(self.num_frames) >= end),
               "action_is_pad": torch.from_numpy(~window["valid"]),
               "image_is_pad": torch.tensor([False, index + self.horizon >= end]),
               "task": self._tasks[slot][episode["task_index"]], "idx": index,
               "episode_index": episode["episode_index"], "dataset_index": slot,
               "task_index": episode["task_index"]}
        sample = self.processor.preprocess(raw)
        video = sample["pixel_values"]
        if self.video_augmentation is not None:
            video = self.video_augmentation(video)
        video = self.processor.pack_video(video, self.video_size)
        prompt = DEFAULT_PROMPT.format(task=sample["instruction"])
        result = {key: value for key, value in sample.items() if key != "pixel_values"}
        result.update(video=video, proprio=sample["proprio"][:-1],
                      prompt=prompt, instruction=prompt, ode_start=torch.from_numpy(window["start"]),
                      ode_target=torch.from_numpy(window["targets"]),
                      ode_raw_state=torch.from_numpy(window["raw_state"]),
                      ode_grip_target=torch.from_numpy(window["commands"][:, 6].copy()))
        result["text_hidden_states"], result["text_attention_mask"] = self._get_cached_qwen_context(prompt)
        return result


def build_datasets(data_cfg):
    from hydra.utils import instantiate
    train = instantiate(data_cfg.train)
    if data_cfg.get("val") is not None:
        raise ValueError("The full-data LIBERO recipe does not create a validation dataset")
    return train, None
