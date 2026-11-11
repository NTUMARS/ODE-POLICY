"""Enumerate the installed benchmark, without loading a policy or simulator."""
import hashlib
import json
import re
from importlib.resources import files
from pathlib import Path

import numpy as np
import yaml

from .libero_env import load_initial_states


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def state_sha(state):
    value = np.asarray(state, dtype="<f8")
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("Benchmark initial state must be a finite vector")
    return hashlib.sha256(value.tobytes()).hexdigest()


def instruction_text(text):
    """Remove generated variant metadata, keeping the actual language perturbation."""
    patterns = (r"\s+view(?:\s+-?\d+(?:\.\d+)?)+\s+initstate\s+\d+(?:\s+noise\s+\d+)?$",
                r"\s+noise\s+\d+$", r"\s+add\s+\d+$", r"\s+level\d+\s+sample\d+$",
                r"\s+(?:table|tb|light)\s+\d+$")
    text = " ".join(str(text).split())
    while True:
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE).strip()
        if cleaned == text:
            if not text:
                raise ValueError("Empty benchmark instruction")
            return text
        text = cleaned


def load_protocol(name):
    path = Path(name)
    raw = path.read_text() if path.is_file() else files("ode_imagewam").joinpath(
        f"configs/{name}.yaml").read_text()
    protocol = yaml.safe_load(raw)
    expected_keys = {"name", "benchmark", "suites", "expected_tasks", "trials_per_task", "seed",
                     "wait_steps", "resolution", "replan_steps", "max_steps", "category_counts"}
    if set(protocol) != expected_keys or protocol["benchmark"] not in ("libero", "libero-plus"):
        raise ValueError("Unsupported benchmark protocol schema")
    if set(protocol["suites"]) != set(protocol["max_steps"]) or len(set(protocol["suites"])) != len(protocol["suites"]):
        raise ValueError("Each distinct suite needs an explicit step budget")
    for key in ("expected_tasks", "trials_per_task", "resolution", "replan_steps"):
        if type(protocol[key]) is not int or protocol[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if protocol["wait_steps"] < 0 or any(int(v) < 1 for v in protocol["max_steps"].values()):
        raise ValueError("Invalid simulator step budget")
    if protocol["benchmark"] == "libero-plus" and protocol["trials_per_task"] != 1:
        raise ValueError("LIBERO-Plus full enumeration uses state zero once per instance")
    return protocol


def build_manifest(protocol, suite_factories, bddl_root, categories=None):
    """Read actual tasks/state arrays; expected counts only validate this enumeration."""
    tasks, category_counts = [], {}
    for suite_name in protocol["suites"]:
        suite = suite_factories[suite_name]()
        names = set()
        for task_id in range(suite.n_tasks):
            task = suite.get_task(task_id)
            if task.name in names:
                raise ValueError(f"Duplicate task name in {suite_name}: {task.name}")
            names.add(task.name)
            states = load_initial_states(suite, task_id)
            count = protocol["trials_per_task"]
            if len(states) < count:
                raise ValueError(f"Not enough distinct initial states: {suite_name}/{task_id}")
            initial_hashes = [state_sha(states[i]) for i in range(count)]
            if len(set(initial_hashes)) != count:
                raise ValueError(f"Repeated initial state: {suite_name}/{task_id}")
            category = None
            if protocol["benchmark"] == "libero-plus":
                category = categories[suite_name][task.name]
                category_counts[category] = category_counts.get(category, 0) + 1
            bddl = Path(bddl_root) / task.problem_folder / task.bddl_file
            # Plus encodes view/initstate/noise in a virtual filename. Its own
            # environment parses that name; only the underlying BDDL is on disk.
            underlying = Path(str(bddl).split("_view_")[0] + ".bddl") if "_view_" in task.bddl_file else bddl
            tasks.append({"suite": suite_name, "task_id": task_id, "name": task.name,
                          "problem_folder": task.problem_folder, "bddl_file": task.bddl_file,
                          "bddl_sha256": file_sha(underlying), "category": category,
                          "raw_instruction": str(task.language),
                          "instruction": instruction_text(task.language) if categories is not None else str(task.language),
                          "initial_state_sha256": initial_hashes})
    if len(tasks) != protocol["expected_tasks"]:
        raise ValueError(f"Installed benchmark enumerates {len(tasks)} tasks, expected {protocol['expected_tasks']}")
    if category_counts != protocol["category_counts"]:
        raise ValueError(f"Installed Plus categories differ: {category_counts}")
    return {"schema": "ode_imagewam_benchmark_v1", "protocol": protocol, "tasks": tasks,
            "expected_episodes": len(tasks) * protocol["trials_per_task"]}


def installed_manifest(protocol):
    from libero.libero import benchmark, get_libero_path
    categories = None
    if protocol["benchmark"] == "libero-plus":
        classification = Path(benchmark.__file__).parent / "task_classification.json"
        raw = json.loads(classification.read_text())
        categories = {suite: {row["name"]: row["category"] for row in raw[suite]}
                      for suite in protocol["suites"]}
    manifest = build_manifest(protocol, benchmark.get_benchmark_dict(), get_libero_path("bddl_files"), categories)
    root = Path(benchmark.__file__).parent.parent
    manifest["benchmark_source_sha256"] = {
        str(path.relative_to(root)): file_sha(path) for path in
        (Path(benchmark.__file__), root / "envs/env_wrapper.py", root / "envs/bddl_utils.py")}
    if categories is not None:
        manifest["classification_sha256"] = file_sha(classification)
    return manifest


def shard_tasks(manifest, index, count):
    if not 0 <= index < count:
        raise ValueError("Use 0 <= shard-index < num-shards")
    return [task for i, task in enumerate(manifest["tasks"]) if i % count == index]
