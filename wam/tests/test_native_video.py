"""Exercise PyAV encoding and timestamp-based GOP seeking on a tiny synthetic clip."""
import av
import numpy as np
import pytest
import torch

from ode_imagewam.data.video import decode_video_frames


def encode_clip(path):
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=20)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = 2
        for index in range(5):
            image = np.full((32, 32, 3), 20 + 40 * index, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_real_pyav_seek_matches_frame_timestamps_and_last_frame(tmp_path):
    path = tmp_path / "episode.mp4"
    encode_clip(path)
    actual = decode_video_frames(path, [0.05, 0.2], 1e-3)
    assert actual.dtype == torch.uint8 and actual.shape == (2, 3, 32, 32)
    torch.testing.assert_close(actual.float().mean((1, 2, 3)), torch.tensor([60.0, 180.0]), atol=3, rtol=0)
    seeked = decode_video_frames(path, [0.15, 0.2, 0.2], 1e-3)
    torch.testing.assert_close(seeked.float().mean((1, 2, 3)), torch.tensor([140.0, 180.0, 180.0]), atol=3, rtol=0)
    assert torch.equal(seeked[1], seeked[2])
    with pytest.raises(ValueError, match="timestamps"):
        decode_video_frames(path, [0.35], 1e-3)
