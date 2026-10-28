import pathlib
import sys

# the repository may not be pip-installed: make the repo root importable for the tests
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
