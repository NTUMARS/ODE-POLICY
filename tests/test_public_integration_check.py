"""The release check follows what Git would publish, without echoing secret values."""

import importlib.util
from pathlib import Path
import subprocess


SPEC = importlib.util.spec_from_file_location(
    "public_integration_check", Path(__file__).resolve().parents[1] / "scripts/check_public_integrations.py"
)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def repository(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "wam").mkdir()
    (tmp_path / "groot").mkdir()
    return tmp_path


def test_ignored_local_environment_is_not_published_but_forced_asset_is_caught(tmp_path):
    root = repository(tmp_path)
    (root / ".gitignore").write_text(".venv/\n*.safetensors\n")
    (root / "wam/.venv").mkdir()
    (root / "wam/.venv/local.py").write_text("not valid python\n")
    (root / "wam/README.md").write_text("WAM source release\n")
    model = root / "wam/model.safetensors"
    model.write_bytes(b"not a real model")
    assert CHECKER.check(root)["issues"] == []
    subprocess.run(["git", "add", "-f", str(model)], cwd=root, check=True)
    assert CHECKER.check(root)["issues"] == [
        {"path": "wam/model.safetensors", "reason": "model, dataset, cache, or archive asset in source release"}
    ]


def test_private_path_and_credential_report_location_without_value(tmp_path):
    root = repository(tmp_path)
    path = root / "groot/config.py"
    sample = "hf_" + "a" * 35
    path.write_text("DATA = '/Users/example_user/private_data'\nTOKEN = " + repr(sample) + "\n")
    report = CHECKER.check(root)
    assert {(issue["reason"], issue["line"]) for issue in report["issues"]} == {
        ("developer home directory", 1), ("Hugging Face credential", 2)
    }
    assert sample not in str(report)
    assert "private_data" not in str(report)


def test_relative_configuration_and_python_syntax(tmp_path):
    root = repository(tmp_path)
    path = root / "wam/config.py"
    path.write_text("from pathlib import Path\nDATA = Path('data')\n")
    assert CHECKER.check(root)["issues"] == []
    path.write_text("def invalid(\n")
    assert CHECKER.check(root)["issues"][0]["reason"] == "invalid Python syntax"


def test_new_roots_are_checked_and_other_upstream_modules_are_outside_scope(tmp_path):
    root = repository(tmp_path)
    (root / "ode_policy").mkdir()
    (root / "ode_policy/existing.py").write_text("not part of this release check\n")
    (root / "wam/source.py").write_text("VALID = True\n")
    (root / "groot/source.py").write_text("def invalid(\n")
    result = CHECKER.check(root)
    assert result["files_checked"] == 2
    assert result["issues"] == [{"path": "groot/source.py", "line": 1, "reason": "invalid Python syntax"}]


def test_empty_release_fails_instead_of_reporting_success(tmp_path):
    root = repository(tmp_path)
    assert CHECKER.check(root)["issues"][0]["reason"] == "no public source files found"


def test_forced_generated_source_is_rejected(tmp_path):
    root = repository(tmp_path)
    (root / ".gitignore").write_text("build/\n")
    (root / "groot/build").mkdir()
    generated = root / "groot/build/stale.py"
    generated.write_text("OLD = True\n")
    subprocess.run(["git", "add", "-f", str(generated)], cwd=root, check=True)
    assert CHECKER.check(root)["issues"] == [
        {"path": "groot/build/stale.py", "reason": "generated file or local environment in source release"}
    ]


def test_broken_symlink_cannot_bypass_source_boundary(tmp_path):
    root = repository(tmp_path)
    (root / "wam/external.py").symlink_to("missing-private-file.py")
    assert CHECKER.check(root)["issues"] == [
        {"path": "wam/external.py", "reason": "symlink in public integration source"}
    ]


def test_dataset_source_cannot_be_silently_ignored_as_generated_data(tmp_path):
    root = repository(tmp_path)
    (root / "wam/README.md").write_text("WAM\n")
    (root / "wam/.gitignore").write_text("data/\n")
    source = root / "wam/src/ode_imagewam/data/processor.py"
    source.parent.mkdir(parents=True)
    source.write_text("NATIVE_SOURCE = True\n")
    assert CHECKER.check(root)["issues"] == [
        {"path": "wam/src/ode_imagewam/data/processor.py",
         "reason": "Python source is excluded from the Git release"}
    ]
    (root / "wam/.gitignore").write_text("/data/\n")
    assert CHECKER.check(root)["issues"] == []
