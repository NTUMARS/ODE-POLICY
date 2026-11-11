import json
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ode_imagewam.sidecar import SidecarStore, _action_bounds, prepare


def dataset(root, bias=0.0):
    (root / "meta").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps({"total_frames": 7, "fps": 20}))
    (root / "meta/tasks.jsonl").write_text('{"task_index":0,"task":"pick up the object"}\n')
    offset = 0
    for episode, length in enumerate([3, 4]):
        states = np.zeros((length, 8), dtype=np.float64)
        states[:, 0] = np.arange(length) / 100 + bias
        commands = np.zeros((length, 7), dtype=np.float64)
        commands[:, 0] = 0.2
        commands[:, 6] = np.arange(length) % 2
        table = pa.table({"index": np.arange(offset, offset + length), "frame_index": np.arange(length),
                          "episode_index": np.full(length, episode), "observation.state": states.tolist(),
                          "action": commands.tolist()})
        pq.write_table(table, root / f"data/chunk-000/episode_{episode:06d}.parquet")
        offset += length
    return root


@pytest.fixture
def prepared(tmp_path):
    a, b = dataset(tmp_path / "a"), dataset(tmp_path / "b", 0.2)
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps({"action": {"default": {"global_min": [-1] * 6 + [0], "global_max": [1] * 7}}}))
    sidecar = tmp_path / "prepared"
    prepare([a, b], stats, sidecar)
    return a, b, stats, sidecar


def test_migration_identity_and_episode_padding(prepared, tmp_path):
    a, b, _, sidecar = prepared
    moved = tmp_path / "moved"
    shutil.copytree(a, moved / "first")
    shutil.copytree(b, moved / "second")
    shutil.copytree(sidecar, moved / "sidecar")
    store = SidecarStore(moved / "sidecar", [moved / "first", moved / "second"])
    window = store.window(1, 12)
    np.testing.assert_array_equal(window["indices"], [1] + [2] * 11)
    np.testing.assert_array_equal(window["valid"], [True, True] + [False] * 10)
    np.testing.assert_allclose(store.starts[3, :3], [0, 0, 0])
    assert store.metadata["num_frames"] == 14
    manifest = json.dumps(store.metadata)
    assert str(tmp_path) not in manifest
    with pytest.raises(ValueError, match="content or order"):
        SidecarStore(sidecar, [b, a])
    (moved / "first/meta/tasks.jsonl").write_text('{"task":"different task"}\n')
    with pytest.raises(ValueError, match="content or order"):
        SidecarStore(sidecar, [moved / "first", moved / "second"])


def test_modified_payload_or_statistics_rejected(prepared):
    _, _, _, sidecar = prepared
    with (sidecar / "states.npy").open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"x")
    with pytest.raises(ValueError, match="hash mismatch"):
        SidecarStore(sidecar)
    (sidecar / "dataset_stats.json").write_text("{}")
    with pytest.raises(ValueError, match="statistics hash"):
        SidecarStore(sidecar)


def test_refuses_overwrite_and_wrong_fps(prepared, tmp_path):
    a, _, stats, sidecar = prepared
    with pytest.raises(FileExistsError):
        prepare([a], stats, sidecar)
    (a / "meta/info.json").write_text('{"total_frames":7,"fps":30}')
    with pytest.raises(ValueError, match="fps"):
        prepare([a], stats, tmp_path / "wrong")


def test_constant_normalization_channel_maps_to_zero():
    low, high = _action_bounds({"action": {"default": {"global_min": [0.3] * 7, "global_max": [0.3] * 7}}})
    np.testing.assert_allclose(2 * (0.3 - low) / (high - low) - 1, 0)


def test_prepare_can_generate_bound_minmax_statistics(prepared, tmp_path):
    a, _, _, _ = prepared
    directory = tmp_path / "computed"
    metadata = prepare([a], None, directory)
    store = SidecarStore(directory, [a])
    stats = json.loads((directory / "dataset_stats.json").read_text())
    assert metadata["normalization_stats_origin"] == "computed_minmax_over_selected_numeric_training_frames"
    np.testing.assert_allclose(stats["state"]["default"]["global_max"], store.states.max(0))
    np.testing.assert_allclose(stats["action"]["default"]["global_min"], store.commands.min(0))
