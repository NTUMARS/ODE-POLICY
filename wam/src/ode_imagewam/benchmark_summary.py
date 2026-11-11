"""Aggregate only complete, unique episodes from one bound benchmark run."""
import json
from collections import Counter, defaultdict
from pathlib import Path

from .benchmark_protocol import digest
from .benchmark_runner import result_path, validate_result, write_once


def summarize(output, destination=None):
    output = Path(output)
    manifest = json.loads((output / "manifest.json").read_text())
    run = json.loads((output / "run.json").read_text())
    binding = run["binding"]
    if binding["manifest_sha256"] != digest(manifest):
        raise ValueError("Benchmark manifest changed after the run was created")
    errors = list((output / "errors").glob("*.json"))
    if errors:
        raise ValueError(f"Execution errors remain in {len(errors)} files; no success summary was produced")
    expected, missing, rows = set(), [], []
    for task in manifest["tasks"]:
        for trial in range(manifest["protocol"]["trials_per_task"]):
            path = result_path(output, task, trial)
            if path in expected:
                raise ValueError("Duplicate task/trial in manifest")
            expected.add(path)
            if not path.is_file():
                missing.append(str(path.relative_to(output)))
                continue
            row = json.loads(path.read_text())
            validate_result(row, binding, task, trial)
            rows.append(row)
    extra = set((output / "results").rglob("*.json")) - expected
    if extra or missing or len(rows) != manifest["expected_episodes"]:
        raise ValueError(f"Incomplete or duplicate results: missing={len(missing)}, unexpected={len(extra)}, "
                         f"completed={len(rows)}, expected={manifest['expected_episodes']}")
    groups = {"suites": defaultdict(Counter), "categories": defaultdict(Counter)}
    for row in rows:
        for grouping, key in (("suites", "suite"), ("categories", "category")):
            if row[key] is not None:
                groups[grouping][row[key]].update(episodes=1, successes=int(row["success"]))
    def rates(counts):
        return {key: {**value, "success_rate": value["successes"] / value["episodes"]}
                for key, value in counts.items()}
    successes = sum(row["success"] for row in rows)
    result = {"status": "complete", "protocol": manifest["protocol"], "binding": binding,
              "episodes": len(rows), "successes": successes, "success_rate": successes / len(rows),
              "suites": rates(groups["suites"]), "categories": rates(groups["categories"]),
              "weighting": "actual episode counts; no subset extrapolation",
              "latency_scope": "synchronized model.infer_action; excludes preprocessing/cache I/O/simulator"}
    write_once(destination or output / "summary.json", result)
    return result
