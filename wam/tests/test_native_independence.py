"""The public package must work without a second WAM framework checkout."""
import os
import subprocess
import sys


def test_native_components_import_with_external_wam_framework_blocked():
    code = '''
import importlib.abc
import sys

class BlockExternalWAM(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "imagewam" or fullname.startswith(("imagewam.", "experiments.libero")):
            raise AssertionError("External WAM framework import: " + fullname)

sys.meta_path.insert(0, BlockExternalWAM())
from ode_imagewam import runtime, model, dataset, trainer, inference, initialization, libero_env
from ode_imagewam.config import load_config, validate_config
assert runtime.ImageWAM.__module__.startswith("ode_imagewam.")
assert all(not cls.__module__.startswith("imagewam") for cls in trainer.PortableTrainer.__mro__)
config = load_config()
validate_config(config)
assert "imagewam_root" not in config
'''
    env = dict(os.environ)
    env.pop("IMAGEWAM_SOURCE", None)
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
