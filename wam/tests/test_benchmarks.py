import hashlib
import json
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ode_imagewam.benchmark_identity import PROMPT_TEMPLATE, asset_identity, text_identity
from ode_imagewam.benchmark_protocol import (build_manifest, digest, instruction_text, load_protocol,
                                           shard_tasks)
from ode_imagewam.benchmark_runner import make_env, result_path, run_tasks, write_once
from ode_imagewam.benchmark_summary import summarize


@pytest.fixture
def panel(tmp_path):
    protocol = load_protocol("libero2000")
    protocol.update(suites=["s"], expected_tasks=3, trials_per_task=2, max_steps={"s": 8})
    tasks = [SimpleNamespace(name=f"task{i}", language=f"pick object {i}", problem_folder="s",
                             bddl_file=f"task{i}.bddl") for i in range(3)]
    (tmp_path / "s").mkdir()
    for task in tasks:
        (tmp_path / "s" / task.bddl_file).write_text(task.name)
    class Suite:
        n_tasks = 3
        def get_task(self, index):
            return tasks[index]
        def get_task_init_states(self, index):
            return [np.array([index, trial], dtype=np.float64) for trial in range(2)]
    factories = {"s": Suite}
    manifest = build_manifest(protocol, factories, tmp_path)
    return manifest, factories


def test_public_protocols_have_exact_full_panels():
    standard, plus = load_protocol("libero2000"), load_protocol("libero-plus10030")
    assert standard["expected_tasks"] * standard["trials_per_task"] == 2000
    assert plus["expected_tasks"] == sum(plus["category_counts"].values()) == 10030
    assert standard["seed"] == plus["seed"] == 42
    assert standard["wait_steps"] == plus["wait_steps"] == 30
    assert standard["max_steps"] == {"libero_spatial": 400, "libero_object": 400,
                                      "libero_goal": 400, "libero_10": 700}


@pytest.mark.parametrize("suffix", ["table 1", "tb 2", "light 3", "noise 2", "add 4", "level2 sample3",
                                       "view 0 0 100 0 0 initstate 11 noise 5"])
def test_generated_metadata_removed_without_replacing_language(suffix):
    assert instruction_text(f"Please grasp the little cup {suffix}") == "Please grasp the little cup"
    assert instruction_text("Gently lift the crimson vessel") == "Gently lift the crimson vessel"


def test_enumeration_reads_actual_counts_and_states(panel, tmp_path):
    manifest, factories = panel
    assert len(manifest["tasks"]) == 3 and manifest["expected_episodes"] == 6
    assert len(manifest["tasks"][0]["initial_state_sha256"]) == 2
    broken = dict(manifest["protocol"], expected_tasks=4)
    with pytest.raises(ValueError, match="enumerates 3"):
        build_manifest(broken, factories, tmp_path)
    with pytest.raises(ValueError, match="Not enough"):
        build_manifest(dict(broken, trials_per_task=3), factories, tmp_path)
    shards = [shard_tasks(manifest, i, 2) for i in range(2)]
    assert {t["task_id"] for t in shards[0]}.isdisjoint(t["task_id"] for t in shards[1])
    assert sorted(t["task_id"] for shard in shards for t in shard) == [0, 1, 2]
    with pytest.raises(ValueError, match="shard-index"):
        shard_tasks(manifest, 2, 2)


def exercise(panel, output, *, shard=0, shards=1, resume=False, fail=False):
    manifest, factories = panel
    counters = {"models": 0, "environments": 0, "closed": 0, "episodes": 0}
    def planner_factory():
        counters["models"] += 1
        return object()
    class Env:
        def close(self):
            counters["closed"] += 1
    def env_factory(*args):
        counters["environments"] += 1
        return Env()
    def episode(env, state, planner, instruction, **kwargs):
        counters["episodes"] += 1
        assert kwargs == dict(max_steps=8, wait_steps=30, replan_steps=12)
        if fail:
            raise RuntimeError("simulator execution failed")
        return {"success": bool(state[1]), "executed_steps": 3, "model_ms_per_chunk": [1.2]}
    binding = {"manifest_sha256": digest(manifest), "checkpoint_sha256": "checkpoint",
               "semantics_sha256": "cfg-and-assets"}
    write_once(output / "manifest.json", manifest)
    write_once(output / "run.json", {"binding": binding, "num_shards": shards})
    run_tasks(manifest, output, binding, shard_index=shard, num_shards=shards,
              planner_factory=planner_factory, suite_factories=factories, env_factory=env_factory,
              episode_runner=episode, resume=resume)
    return counters


def test_models_persist_per_worker_and_shards_aggregate(panel, tmp_path):
    output = tmp_path / "eval"
    first = exercise(panel, output, shard=0, shards=2)
    second = exercise(panel, output, shard=1, shards=2)
    assert first == dict(models=1, environments=2, closed=2, episodes=4)
    assert second == dict(models=1, environments=1, closed=1, episodes=2)
    result = summarize(output)
    assert result["episodes"] == 6 and result["successes"] == 3
    assert result["success_rate"] == 0.5
    assert exercise(panel, output, shard=0, shards=2, resume=True)["models"] == 0


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong-checkpoint", "wrong-category", "error"])
def test_summary_rejects_missing_duplicate_mixed_or_failed_execution(panel, tmp_path, change):
    output = tmp_path / "eval"
    exercise(panel, output)
    task = panel[0]["tasks"][0]
    path = result_path(output, task, 0)
    if change == "missing":
        path.unlink()
    elif change == "duplicate":
        shutil.copy2(path, path.with_name("duplicate.json"))
    elif change == "error":
        (output / "errors").mkdir()
        (output / "errors/failed.json").write_text("{}")
    else:
        row = json.loads(path.read_text())
        if change == "wrong-category":
            row["category"] = "another perturbation"
        else:
            row["binding"]["checkpoint_sha256"] = "different checkpoint"
        path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        summarize(output)
    assert not (output / "summary.json").exists()


def test_partial_task_does_not_claim_rng_resume(panel, tmp_path):
    output = tmp_path / "eval"
    exercise(panel, output)
    result_path(output, panel[0]["tasks"][0], 1).unlink()
    with pytest.raises(RuntimeError, match="RNG stream"):
        exercise(panel, output, resume=True)


def test_execution_error_is_preserved_and_not_counted_as_failure(panel, tmp_path):
    output = tmp_path / "eval"
    with pytest.raises(RuntimeError, match="execution failed"):
        exercise(panel, output, fail=True)
    assert len(list((output / "errors").glob("*.json"))) == 1
    with pytest.raises(ValueError, match="Execution errors"):
        summarize(output)


def test_plus_keeps_virtual_filename_and_string_type(monkeypatch, tmp_path):
    seen = {}
    class Env:
        def __init__(self, **kwargs):
            seen.update(kwargs)
        def seed(self, seed):
            seen["seed"] = seed
    monkeypatch.setitem(sys.modules, "libero", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "libero.libero", SimpleNamespace(get_libero_path=lambda key: str(tmp_path)))
    monkeypatch.setitem(sys.modules, "libero.libero.envs", SimpleNamespace(OffScreenRenderEnv=Env))
    task = SimpleNamespace(problem_folder="suite", bddl_file="task_view_0_0_100_0_0_initstate_11_noise_5.bddl")
    make_env(task, 256, 42)
    assert seen["bddl_file_name"] == str(tmp_path / "suite" / task.bddl_file)
    assert isinstance(seen["bddl_file_name"], str) and seen["seed"] == 42


def test_asset_binding_covers_cache_and_online_qwen(tmp_path):
    cfg = SimpleNamespace(paths=SimpleNamespace(text_cache_dir=tmp_path, qwen_model=tmp_path,
                                                autoencoder=tmp_path / "ae"),
                          model=SimpleNamespace(qwen_context_len=128))
    manifest = {"tasks": [{"instruction": "pick cup"}]}
    prompt = PROMPT_TEMPLATE.format(task="pick cup")
    cache = tmp_path / f"{hashlib.sha256(prompt.encode()).hexdigest()}.qwen3_flux2_len128.pt"
    cache.write_bytes(b"cached text A")
    cfg.paths.autoencoder.write_bytes(b"vae")
    first = asset_identity(cfg, manifest, False)
    cache.write_bytes(b"cached text B")
    assert first != asset_identity(cfg, manifest, False)
    (tmp_path / "config.json").write_text("{}")
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"weights A")
    online = text_identity(cfg, manifest, True)
    weights.write_bytes(b"weights B")
    assert online != text_identity(cfg, manifest, True)
    assert online["mode"] != first["text"]["mode"]
