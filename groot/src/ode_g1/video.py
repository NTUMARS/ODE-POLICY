"""Exact indexed RGB decoding with bounded CPU decoder/frame caches."""

from collections import OrderedDict
from pathlib import Path
import numpy as np
from .data import _verify_source


class VideoFrameReader:
    def __init__(self, max_open_videos=4, *, fingerprints=None, decoder_factory=None, max_cached_frames_per_video=32):
        if max_open_videos < 1 or max_cached_frames_per_video < 1:
            raise ValueError("Positive cache bounds required")
        self.max_open_videos = max_open_videos
        self.max_cached_frames_per_video = max_cached_frames_per_video
        self.fingerprints = fingerprints or {}
        self.decoder_factory = decoder_factory
        self.decoders = OrderedDict()
        self.frames = {}

    def read(self, path, index):
        if not isinstance(index, (int, np.integer)) or index < 0:
            raise IndexError("Nonnegative video frame required")
        key = str(Path(path).resolve())
        if key not in self.decoders:
            if key in self.fingerprints:
                _verify_source(key, self.fingerprints[key])
            factory = self.decoder_factory
            if factory is None:
                try:
                    from torchcodec.decoders import VideoDecoder
                except ImportError as exc:
                    raise ImportError("Feature preparation needs torchcodec and FFmpeg; see README") from exc
                factory = VideoDecoder
            self.decoders[key] = factory(
                key, dimension_order="NHWC", num_ffmpeg_threads=1, device="cpu", seek_mode="exact"
            )
            self.frames[key] = OrderedDict()
            while len(self.decoders) > self.max_open_videos:
                old, decoder = self.decoders.popitem(last=False)
                self.frames.pop(old)
                close = getattr(decoder, "close", None)
                if close:
                    close()
        decoder = self.decoders[key]
        self.decoders.move_to_end(key)
        if index >= len(decoder):
            raise IndexError("Video frame exceeds episode")
        frames = self.frames[key]
        if index not in frames:
            raw = decoder[int(index)]
            raw = raw.numpy() if hasattr(raw, "numpy") else raw
            array = np.asarray(raw)
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
                raise ValueError("Decoder must return uint8 RGB HWC")
            frames[index] = array.copy()
            while len(frames) > self.max_cached_frames_per_video:
                frames.popitem(last=False)
        frames.move_to_end(index)
        return frames[index].copy()
