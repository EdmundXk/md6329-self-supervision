"""Environment bootstrap: torch's inductor cache-dir helper calls getpass.getuser(),
which falls back to the POSIX-only `pwd` module inside this sandbox. Setting the
cache dir and USERNAME explicitly before importing torch avoids the import error."""
import os, pathlib
os.environ.setdefault("USERNAME", "runner")
os.environ.setdefault("USER", "runner")
_c = pathlib.Path(".torchcache").absolute()
_c.mkdir(exist_ok=True)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(_c / "inductor"))
os.environ.setdefault("TRITON_CACHE_DIR", str(_c / "triton"))

# The interactive kernel places the conda prefix ahead of the project venv on
# sys.path, which picks up a conda Pillow whose DLLs do not load here. Put the
# venv's site-packages first and drop any half-imported PIL modules.
import sys
_venv = pathlib.Path(".venv/dl/Lib/site-packages").absolute()
if _venv.is_dir():
    s = str(_venv)
    if sys.path and sys.path[0] != s:
        sys.path.insert(0, s)
        for _m in [m for m in list(sys.modules) if m.split(".")[0] == "PIL"]:
            del sys.modules[_m]


def ensure_dirs(*names):
    """Create the output directories the pipeline writes to. Scripts call this so a
    fresh clone works without any manual mkdir."""
    for n in names or ("data", "ckpt", "results", "results/preds", "figs"):
        pathlib.Path(n).mkdir(parents=True, exist_ok=True)
