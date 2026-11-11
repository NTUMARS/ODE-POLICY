import subprocess
import sys

import pytest
from omegaconf.errors import ConfigAttributeError, ConfigKeyError

from ode_imagewam.config import load_config, resolve_user_paths, validate_config


@pytest.mark.parametrize("command", [[], ["train"], ["infer"], ["eval"], ["prepare"], ["inspect-sidecar"],
                                     ["cache-text"], ["eval-benchmark"], ["plan-eval"], ["summarize-eval"]])
def test_help_requires_no_upstream_or_weights(command):
    result = subprocess.run([sys.executable, "-m", "ode_imagewam", *command, "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_full_template_has_h12_defaults_and_typed_overrides(tmp_path):
    cfg = load_config(overrides=["batch_size=2", "persistent_workers=false", "paths.dataset_dirs=[one,two]",
                                 "paths.output_dir=./relative-output"])
    validate_config(cfg)
    assert cfg.batch_size == 2 and cfg.persistent_workers is False
    assert cfg.model.ode.horizon == 12
    assert cfg.model.action_dit_config.num_layers_double == 5
    assert cfg.model.action_dit_config.num_layers_single == 20
    assert cfg.model.loss.lambda_video == 0.5
    assert cfg.model.ode.layers == 4
    assert cfg.data.train.require_text_cache is False
    assert cfg.resume is None and cfg.max_steps is None and cfg.lr_scheduler_total_steps is None
    resolve_user_paths(cfg)
    assert all(item.startswith("/") for item in cfg.paths.dataset_dirs)
    with pytest.raises(ValueError, match="frames"):
        broken = load_config(overrides=["data.train.num_frames=17"])
        validate_config(broken)
    with pytest.raises(ValueError, match="DeepSpeed"):
        validate_config(load_config(overrides=["num_processes=2"]))
    with pytest.raises(ValueError, match="legacy Wan/T5"):
        validate_config(load_config(overrides=["data.train.require_text_cache=true"]))
    with pytest.raises((ConfigKeyError, ConfigAttributeError)):
        load_config(overrides=["batch_szie=4"])
    user_config = tmp_path / "config.yaml"
    user_config.write_text("paths:\n  output_dir: result\n")
    relocated = resolve_user_paths(load_config(user_config))
    assert relocated.paths.output_dir == str(tmp_path / "result")


def test_template_config_check_does_not_import_backbone():
    code = "from ode_imagewam.cli import main; main(['check-config']); import sys; assert 'imagewam' not in sys.modules; assert 'torch' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_init_config_exports_the_packaged_single_source(tmp_path):
    from importlib.resources import files
    output = tmp_path / "config.yaml"
    result = subprocess.run([sys.executable, "-m", "ode_imagewam", "init-config", str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert output.read_text() == files("ode_imagewam").joinpath("configs/libero_ode.yaml").read_text()
    validate_config(load_config(output))
