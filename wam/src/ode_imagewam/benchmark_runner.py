"""One policy per worker, deterministic task shards and per-episode results."""
import fcntl
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from .benchmark_protocol import digest, file_sha, installed_manifest, load_protocol, shard_tasks, state_sha
from .libero_env import make_env, load_initial_states


def write_once(path, value):
    """Publish a complete JSON atomically; an existing file must match exactly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        temp = Path(stream.name)
    try:
        try:
            os.link(temp, path)
        except FileExistsError:
            if json.loads(path.read_text()) != value:
                raise ValueError(f"Existing result/config differs: {path}")
    finally:
        temp.unlink()


def result_path(output, task, trial):
    return Path(output) / "results" / task["suite"] / f"task-{task['task_id']:05d}" / f"trial-{trial:03d}.json"


def validate_result(row, binding, task, trial):
    if row.get("binding") != binding or row.get("suite") != task["suite"] or row.get("task_id") != task["task_id"]:
        raise ValueError("Result belongs to a different policy, protocol or task")
    if row.get("initial_state_index") != trial or row.get("initial_state_sha256") != task["initial_state_sha256"][trial]:
        raise ValueError("Result initial state differs from the benchmark manifest")
    for key in ("name", "category", "instruction"):
        if row.get(key) != task[key]:
            raise ValueError(f"Result {key} differs from the benchmark manifest")
    if row.get("status") != "complete" or type(row.get("success")) is not bool:
        raise ValueError("An errored/incomplete episode cannot be skipped or counted as a completed trial")


def run_tasks(manifest, output, binding, *, shard_index, num_shards, planner_factory,
              suite_factories, env_factory=make_env, episode_runner=None, resume=False, seed_fn=None):
    """Dependency injection keeps orchestration testable without model or simulator."""
    if episode_runner is None:
        from .evaluation import run_episode
        episode_runner = run_episode
    protocol = manifest["protocol"]
    assigned = shard_tasks(manifest, shard_index, num_shards)
    planner, completed = None, 0
    for task in assigned:
        pending = []
        for index in range(protocol["trials_per_task"]):
            path = result_path(output, task, index)
            if path.exists():
                if not resume:
                    raise FileExistsError(f"Use --resume to keep verified completed episodes: {path}")
                validate_result(json.loads(path.read_text()), binding, task, index)
            else:
                pending.append(index)
        if not pending:
            continue
        if len(pending) != protocol["trials_per_task"]:
            raise RuntimeError("Partially completed task: continuing with a fresh simulator changes its RNG stream; inspect this task first")
        if planner is None:
            planner = planner_factory()
        suite = suite_factories[task["suite"]]()
        actual = suite.get_task(task["task_id"])
        if (actual.name, actual.problem_folder, actual.bddl_file) != (task["name"], task["problem_folder"], task["bddl_file"]):
            raise ValueError("Installed task differs from the saved manifest")
        states = load_initial_states(suite, task["task_id"])
        for index in pending:
            if state_sha(states[index]) != task["initial_state_sha256"][index]:
                raise ValueError("Installed initial state changed since benchmark enumeration")
        env, current = None, pending[0]
        try:
            if seed_fn:
                seed_fn(protocol["seed"])
            env = env_factory(actual, protocol["resolution"], protocol["seed"])
            for current in pending:
                result = episode_runner(env, states[current], planner, task["instruction"],
                    max_steps=protocol["max_steps"][task["suite"]], wait_steps=protocol["wait_steps"],
                    replan_steps=protocol["replan_steps"])
                row = {"binding": binding, "suite": task["suite"], "task_id": task["task_id"],
                       "name": task["name"], "category": task["category"],
                       "instruction": task["instruction"], "initial_state_index": current,
                       "initial_state_sha256": task["initial_state_sha256"][current],
                       "status": "complete", **result}
                validate_result(row, binding, task, current)
                write_once(result_path(output, task, current), row)
                completed += 1
        except Exception as error:
            path = Path(output) / "errors" / f"shard-{shard_index:03d}-{task['suite']}-{task['task_id']}-{current}.json"
            write_once(path, {"binding": binding, "suite": task["suite"], "task_id": task["task_id"],
                              "trial": current, "error": type(error).__name__, "message": str(error)})
            raise
        finally:
            if env is not None:
                env.close()
    return {"shard_index": shard_index, "num_shards": num_shards, "new_episodes": completed}


def evaluate_benchmark(cfg, checkpoint, output, *, protocol, shard_index=0, num_shards=1,
                       online_text=False, resume=False):
    from .checkpoint import make_binding
    from .benchmark_identity import asset_identity
    from .config import resolve_user_paths, validate_config
    from .inference import OfflinePlanner
    from .sidecar import load_metadata
    cfg = resolve_user_paths(cfg)
    validate_config(cfg, runtime=True)
    from libero.libero import benchmark
    manifest = installed_manifest(load_protocol(protocol))
    if not 1 <= manifest["protocol"]["replan_steps"] <= int(cfg.model.ode.horizon):
        raise ValueError("Protocol replan_steps exceeds the trained horizon")
    shard_tasks(manifest, shard_index, num_shards)
    metadata = load_metadata(Path(str(cfg.paths.sidecar_dir)) / "metadata.json")
    semantics = {"model": make_binding(metadata, cfg.model), "mixed_precision": str(cfg.mixed_precision),
                 "shape_meta": OmegaConf.to_container(cfg.data.train.shape_meta, resolve=True),
                 "processor": OmegaConf.to_container(cfg.data.train.processor, resolve=True),
                 "assets": asset_identity(cfg, manifest, online_text)}
    binding = {"manifest_sha256": digest(manifest), "checkpoint_sha256": file_sha(checkpoint),
               "semantics_sha256": digest(semantics)}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_once(output / "manifest.json", manifest)
    write_once(output / "run.json", {"binding": binding, "num_shards": num_shards})
    with (output / f"shard-{shard_index:03d}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if list((output / "errors").glob("*.json")):
            raise RuntimeError("This run contains execution errors; inspect them before starting a new output directory")
        import torch
        def seed_fn(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        seed_fn(manifest["protocol"]["seed"])
        result = run_tasks(manifest, output, binding, shard_index=shard_index, num_shards=num_shards,
            planner_factory=lambda: OfflinePlanner(cfg, checkpoint, online_text=online_text),
            suite_factories=benchmark.get_benchmark_dict(), resume=resume, seed_fn=seed_fn)
    return {**result, "output": str(output), "binding": binding}
