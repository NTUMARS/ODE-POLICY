import json
import shutil
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ode_g1 import data
from ode_g1.config import load_config
from ode_g1.loader import base_identity
from test_g1_data_windows import columns


def source_fixture(root):
    root.mkdir()
    (root / "meta").mkdir()
    cameras = ["observation.images.front_rgb", "observation.images.side_rgb"]
    features = {
        key: {"shape": [width]}
        for key, width in [
            ("observation.state", 43),
            ("action.motion_token", 64),
            ("observation.inspire_left_hand_state", 6),
            ("observation.inspire_right_hand_state", 6),
            ("observation.projected_gravity", 3),
        ]
    }
    for key in cameras:
        features[key] = {"dtype": "video", "shape": [12, 18, 3]}
    info = {
        "codebase_version": "v2.1",
        "fps": 30,
        "total_episodes": 2,
        "total_frames": 5,
        "chunks_size": 1,
        "splits": {"train": "0:2"},
        "total_videos": 4,
        "features": features,
        "data_path": "tables/block_{episode_chunk:02d}/row_{episode_index:03d}.parquet",
        "video_path": "rgb/{video_key}/block_{episode_chunk:02d}/clip_{episode_index:03d}.mp4",
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    for name, rows in [
        ("tasks", [{"task_index": 0, "task": "Move, the object!"}]),
        ("episodes", [{"episode_index": i, "length": n} for i, n in enumerate([3, 2])]),
        ("episodes_stats", [{"episode_index": i, "stats": {"state": {"count": [n]}}} for i, n in enumerate([3, 2])]),
    ]:
        (root / f"meta/{name}.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    start = 0
    for i, n in enumerate([3, 2]):
        raw = columns(n, i, start)
        table = pa.table({k: v.tolist() for k, v in raw.items()})
        target = root / info["data_path"].format(episode_chunk=i, episode_index=i)
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target)
        for camera in cameras:
            target = root / info["video_path"].format(video_key=camera, episode_chunk=i, episode_index=i)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"fixture-video-no-decoder-used" + bytes([i]))
        start += n
    return cameras


def test_prepare_custom_task_cameras_templates_and_relocate_content(tmp_path):
    source = tmp_path / "source"
    cameras = source_fixture(source)
    spec = data.task_spec_from_root(source, name="my-task", camera_keys=cameras)
    assert (spec.episodes, spec.frames, spec.instruction) == (2, 5, "Move, the object!")
    prepared = data.prepare_task_dataset(source, spec, tmp_path / "prepared")
    assert len(prepared) == 5
    assert prepared.video_path(1, cameras[0]).name == "clip_001.mp4"
    assert prepared[0]["camera_frames"] == {k: 0 for k in cameras}
    assert prepared[2]["action_mask"].sum() == 76
    assert prepared[3]["r0_is_bos"]
    relocated = tmp_path / "other-mount"
    shutil.copytree(source, relocated)
    loaded = data.TaskDataset.load(tmp_path / "prepared", source_root=relocated)
    assert loaded.content_identity_sha256 == prepared.content_identity_sha256
    rebuilt = data.prepare_task_dataset(relocated, spec, tmp_path / "prepared2")
    assert rebuilt.content_identity_sha256 == prepared.content_identity_sha256
    assert rebuilt.manifest["identity_sha256"] != prepared.manifest["identity_sha256"]
    loaded.video_path(0, cameras[0]).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="source content changed"):
        loaded.verify_sources()


def test_metadata_escape_and_frozen_snapshot_rejected(tmp_path):
    source = tmp_path / "source"
    cameras = source_fixture(source)
    spec = data.task_spec_from_root(source, name="my-task", camera_keys=cameras)
    data.prepare_task_dataset(source, spec, tmp_path / "prepared")
    with pytest.raises(ValueError, match="outside source"):
        data.prepare_task_dataset(source, spec, source / "prepared")
    with pytest.raises(FileExistsError):
        data.prepare_task_dataset(source, spec, tmp_path / "prepared")
    with pytest.raises(ValueError, match="escapes"):
        data._inside(source, "../escape")
    snapshot = tmp_path / "prepared/numeric/episode_000000.npz"
    with snapshot.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="snapshot content"):
        data.TaskDataset.load(tmp_path / "prepared")


def test_yaml_paths_overrides_and_schema_are_portable(tmp_path, monkeypatch):
    config = tmp_path / "configs/config.yaml"
    config.parent.mkdir()
    config.write_text("device: cpu\npaths:\n  dataset_root: ${MY_G1_DATA}\nbase_model:\n  path: ../models/base\n")
    monkeypatch.setenv("MY_G1_DATA", "../dataset")
    cfg = load_config(config, ["training.epochs=2", "head.horizon=4"])
    assert cfg["paths"]["dataset_root"] == str(tmp_path / "dataset")
    assert cfg["base_model"]["path"] == str(tmp_path / "models/base")
    assert cfg["training"]["epochs"] == 2
    for invalid in [
        "head.state_dim=43",
        "head.horizon=17",
        "features.require_cache_hits='false'",
        "training.seed=-1",
        "training.batch_size=0",
        "training.max_train_seconds=-1",
        "extra.value=1",
    ]:
        with pytest.raises(ValueError):
            load_config(config, [invalid])


def test_base_identity_uses_actual_shards_without_directory_basename(tmp_path):
    one = tmp_path / "arbitrary-name"
    one.mkdir()
    (one / "model.safetensors").write_bytes(b"toy-shard")
    two = tmp_path / "different-mount"
    shutil.copytree(one, two)
    assert base_identity(one, "revision") == base_identity(two, "revision")
    (two / "model.safetensors").write_bytes(b"changed")
    assert base_identity(one, "revision") != base_identity(two, "revision")


def test_help_and_inspect_do_not_import_torch_groot_or_start_network(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text("device: cpu\n")
    code = "from ode_g1.cli import main; import sys; main(['inspect','--config',sys.argv[1]]); assert 'torch' not in sys.modules; assert 'gr00t' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code, str(config)], text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["device"] == "cpu"
    result = subprocess.run([sys.executable, "-m", "ode_g1", "--help"], text=True, capture_output=True, check=True)
    assert "offline" in result.stdout
    from ode_g1.cli import _single_process

    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="one optimizer"):
        _single_process()


def test_installed_template_is_the_only_source_and_init_is_offline(tmp_path):
    from importlib.resources import files

    template = files("ode_g1").joinpath("configs/g1.yaml").read_text()
    target = tmp_path / "new-config.yaml"
    code = "from ode_g1.cli import main; import sys; main(['init-config',sys.argv[1]]); assert 'torch' not in sys.modules; assert 'gr00t' not in sys.modules"
    subprocess.run([sys.executable, "-c", code, str(target)], check=True, capture_output=True)
    assert target.read_text() == template
    assert load_config(target)["base_model"]["cosmos_path"] is None
