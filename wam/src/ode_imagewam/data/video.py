"""Read selected RGB video timestamps with a bounded PyAV decode window."""
from pathlib import Path

import numpy as np
import torch


def decode_video_frames(path, timestamps, tolerance_s):
    import av

    timestamps = np.asarray(timestamps, dtype=np.float64)
    if timestamps.ndim != 1 or not len(timestamps) or not np.isfinite(timestamps).all():
        raise ValueError("Video timestamps must be a nonempty finite vector")
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(max(0, int(float(timestamps.min()) / stream.time_base)),
                       stream=stream, backward=True, any_frame=False)
        frames, times = [], []
        for frame in container.decode(stream):
            if frame.time is None:
                raise ValueError("Video frames require presentation timestamps")
            frames.append(frame.to_ndarray(format="rgb24"))
            times.append(float(frame.time))
            if frame.time > timestamps.max() + tolerance_s:
                break
    if not frames:
        raise ValueError(f"Video decoder returned no frames: {path}")
    distance = np.abs(timestamps[:, None] - np.asarray(times)[None, :])
    indices = distance.argmin(axis=1)
    if np.any(distance[np.arange(len(indices)), indices] > tolerance_s):
        raise ValueError("Requested timestamps do not match decoded video frames")
    rgb = np.stack([frames[index] for index in indices])
    return torch.from_numpy(rgb).permute(0, 3, 1, 2)
