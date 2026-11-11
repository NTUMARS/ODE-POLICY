"""Resolve explicitly supplied official FLUX.2 source or an installed package."""
import importlib.util
import sys
from pathlib import Path


def ensure_flux2_importable(flux2_src_path=None):
    if flux2_src_path:
        root = Path(str(flux2_src_path)).expanduser().resolve(strict=True)
        candidates = [root / "src", root]
        path = next((p for p in candidates if (p / "flux2/model.py").is_file()), None)
        if path is None:
            raise ValueError("flux2_src must contain the official flux2/model.py package")
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    if importlib.util.find_spec("flux2") is None:
        raise ImportError("Install FLUX.2 or set paths.flux2_src to its source checkout")
