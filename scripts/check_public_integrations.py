#!/usr/bin/env python3
"""Check WAM/GR00T release files without importing ML or robot libraries."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import subprocess


PRIVATE_MARKERS = {
    "developer home directory": re.compile(r"/Users/(?!example(?:/|$))[^/\s]+/"),
    "developer Linux home directory": re.compile(r"/home/(?!example(?:/|$)|user(?:/|$))[^/\s]+/"),
    "cluster project directory": re.compile(r"/projects/faculty/"),
    "private network endpoint": re.compile(
        r"\b(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|"
        r"172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+|"
        r"100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d+\.\d+)\b"
    ),
    "private key": re.compile(r"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----"),
    "Hugging Face credential": re.compile(r"\bhf_[A-Za-z0-9]{25,}\b"),
    "GitHub credential": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
}
EXCLUDED_ASSETS = {
    ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".npz", ".npy",
    ".parquet", ".arrow", ".mp4", ".avi", ".zip", ".gz", ".tar",
}
SOURCE_ROOTS = ("wam", "groot", "docs/wam_libero", "docs/deploy_groot")
GENERATED_PARTS = {
    "__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv", "build",
    "dist", "cache", "checkpoints", "runs", "wandb", "outputs",
}


def check(root: Path) -> dict:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", *SOURCE_ROOTS],
        cwd=root, check=True, capture_output=True,
    )
    paths = sorted(set(result.stdout.decode().split("\0")) - {""})
    issues = []
    for package in ("wam", "groot"):
        for path in (root / package / "src").rglob("*.py"):
            relative = str(path.relative_to(root))
            if "__pycache__" not in path.parts and relative not in paths:
                issues.append({"path": relative, "reason": "Python source is excluded from the Git release"})
    python_files = shell_files = 0
    files_checked = 0
    for relative in paths:
        path = root / relative
        if not path.exists() and not path.is_symlink():  # Ignore tracked deletions, not broken symlinks.
            continue
        files_checked += 1
        if path.is_symlink():
            issues.append({"path": relative, "reason": "symlink in public integration source"})
            continue
        if any(part in GENERATED_PARTS or part.startswith(".venv-") or part.endswith(".egg-info")
               for part in Path(relative).parts):
            issues.append({"path": relative, "reason": "generated file or local environment in source release"})
            continue
        if path.suffix.lower() in EXCLUDED_ASSETS:
            issues.append({"path": relative, "reason": "model, dataset, cache, or archive asset in source release"})
            continue
        raw = path.read_bytes()
        if len(raw) > 1024 * 1024:
            issues.append({"path": relative, "reason": "source file exceeds 1 MiB"})
            continue
        try:
            source = raw.decode("utf-8")
        except UnicodeDecodeError:
            issues.append({"path": relative, "reason": "non-text file in integration source release"})
            continue
        for name, pattern in PRIVATE_MARKERS.items():
            match = pattern.search(source)
            if match:
                # Do not echo a credential or a private value into logs.
                issues.append({"path": relative, "line": source.count("\n", 0, match.start()) + 1, "reason": name})
        if path.suffix == ".py":
            python_files += 1
            try:
                ast.parse(source, filename=relative)
            except SyntaxError as error:
                issues.append({"path": relative, "line": error.lineno, "reason": "invalid Python syntax"})
        elif path.suffix == ".sh":
            shell_files += 1
            syntax = subprocess.run(["bash", "-n", str(path)], capture_output=True)
            if syntax.returncode:
                issues.append({"path": relative, "reason": "invalid shell syntax"})
    if not files_checked:
        issues.append({"path": ", ".join(SOURCE_ROOTS), "reason": "no public source files found"})
    return {"files_checked": files_checked, "python_files": python_files, "shell_files": shell_files, "issues": issues}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = check(args.root.resolve())
    print(json.dumps(report, indent=2))
    return int(bool(report["issues"]))


if __name__ == "__main__":
    raise SystemExit(main())
