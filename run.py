"""Bootstrap wrapper that installs missing dependencies on the fly and delegates to main.py.

This script is designed for users who do not want to manage virtual environments
or install packages manually.  It will check for required Python modules,
install any that are missing using pip, warn about missing system tools
(FFmpeg), and then forward its command‑line arguments to `main.py`.
"""

"""
from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
import shutil
import json


# Map import names to pip package names.  If a module import fails, the
# corresponding package will be installed via pip.
REQUIRED_MODULES = {
    "yaml": "pyyaml",  # import name: pip package name
    "numpy": "numpy<2",
    "scipy": "scipy",
    "soundfile": "soundfile",
    "librosa": "librosa",
    #"musicnn": "musicnn",
    #"dejavu": "PyDejavu",
    "faiss": "faiss-cpu",
    "laion_clap": "laion-clap @ git+https://github.com/LAION-AI/CLAP.git#egg=laion_clap",
    "torch": "torch",
    "hdbscan": "hdbscan",
    "sklearn": "scikit-learn",
    "mutagen": "mutagen",
    "rich": "rich",
    # Note: PyTorch installation may need a specific CUDA wheel; fallback to CPU.
}

"""


# run.py — self-bootstrapping launcher
# - Ensures pip exists
# - Installs required packages (with safe pins)
# - Forces NumPy < 2 (ABI compatibility)
# - Fixes 'progressbar' -> 'progressbar2' before laion_clap
# - Installs laion_clap last, then optional deps
# - Delegates to main.py

import os
import sys
import subprocess
import importlib
import shutil
import json

# ---------------------- dependency bootstrap ----------------------

REQUIRED_BASE = {
    # import name -> pip spec
    "yaml": "pyyaml",
    "numpy": "numpy<2",                # pin to 1.x for torch/torchvision/laion_clap
    "scipy": "scipy>=1.10",
    "soundfile": "soundfile>=0.12",
    "librosa": "librosa==0.11.0",      # avoids numba headaches on Win+Py3.11
    "sklearn": "scikit-learn>=1.3",
    "mutagen": "mutagen>=1.46",
    "requests": "requests>=2.31",
    "faiss": "faiss-cpu",              # module 'faiss', package 'faiss-cpu'
    "uvicorn":   "uvicorn[standard]>=0.30",
    "fastapi":   "fastapi>=0.111",
    "multipart": "python-multipart>=0.0.9", 
    # laion_clap is installed separately after we fix progressbar
}

OPTIONAL_MODULES = [
    ("dejavu", "PyDejavu"),            # fingerprinting (optional on Win/Py3.11)
    ("musicnn", "musicnn"),            # tagging (skip on Py>=3.11)
]

def _have(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False

def _pip(*args: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", *args])

def _numpy_major() -> int | None:
    try:
        out = subprocess.check_output(
            [sys.executable, "-c", "import json, numpy as np; print(json.dumps(np.__version__))"]
        )
        ver = json.loads(out.decode().strip())
        return int(ver.split(".")[0])
    except Exception:
        return None

def _fix_progressbar_py3() -> None:
    """
    Ensure 'progressbar' resolves to a Python 3-compatible module.
    Many wheels named 'progressbar' are Python-2 era. 'progressbar2' provides the
    modern module under the same import name 'progressbar'.
    """
    try:
        import progressbar  # noqa: F401
        # try touching an attribute to trigger a SyntaxError early if it's the py2 one
        getattr(progressbar, "__version__", None)
        return
    except SyntaxError:
        pass
    except Exception:
        pass

    # Remove old 'progressbar' if present, then install progressbar2
    try:
        _pip("uninstall", "-y", "progressbar")
    except subprocess.CalledProcessError:
        pass
    _pip("install", "--no-cache-dir", "progressbar2>=4.4")

def _install_laion_clap() -> None:
    """
    Install laion_clap AFTER numpy<2 and progressbar2 are in place.
    """
    if _have("laion_clap"):
        return
    print("Installing laion_clap (after numpy<2 and progressbar2)…")
    _fix_progressbar_py3()
    _pip("install", "git+https://github.com/LAION-AI/CLAP.git#egg=laion_clap")
    importlib.invalidate_caches()
    if not _have("laion_clap"):
        print("Installed laion_clap, but still cannot import it. "
              "If this persists, ensure NumPy is <2 and try again.")
        sys.exit(1)

def ensure_dependencies() -> None:
    """Ensure required modules are importable. Install via pip if not."""
    # 0) pip
    try:
        import pip  # noqa: F401
    except Exception:
        try:
            subprocess.check_call([sys.executable, "-m", "ensurepip", "--default-pip"])
        except Exception as e:
            print("Failed to bootstrap pip for this Python interpreter.\n"
                  f"Interpreter: {sys.executable}\nError: {e}")
            sys.exit(1)

    # 1) Required base (without laion_clap yet)
    for mod, pkg in REQUIRED_BASE.items():
        if _have(mod):
            continue
        print(f"Dependency '{mod}' not found. Installing {pkg}…")
        try:
            _pip("install", pkg)
            importlib.invalidate_caches()
            if not _have(mod):
                print(f"Installed {pkg}, but still cannot import '{mod}'.")
                sys.exit(1)
        except subprocess.CalledProcessError as e:
            if mod == "faiss":
                print("If 'pip install faiss-cpu' fails on Windows, try once:\n"
                      "  conda install -c conda-forge faiss-cpu")
            print(f"Failed to install {pkg}: {e}")
            sys.exit(1)

    # 2) Force NumPy < 2 (and restart if needed)
    maj = _numpy_major()
    if maj is not None and maj >= 2:
        print("Detected NumPy 2.x; switching to NumPy<2 for laion_clap/torch compatibility…")
        try:
            _pip("install", "--force-reinstall", "--no-cache-dir", "numpy<2")
        except subprocess.CalledProcessError as e:
            print(f"Failed to switch NumPy to <2: {e}")
            sys.exit(1)
        print("NumPy changed. Restarting the script to use the new ABI…")
        os.execv(sys.executable, [sys.executable] + sys.argv)

    # 3) Progressbar fix FIRST, then laion_clap
    _fix_progressbar_py3()
    _install_laion_clap()

    # 4) Optional deps
    for mod, pkg in OPTIONAL_MODULES:
        if mod == "musicnn" and sys.version_info >= (3, 11):
            print("Optional 'musicnn' skipped on Python >= 3.11 (incompatible).")
            continue
        if _have(mod):
            continue
        try:
            print(f"Optional dependency '{mod}' missing. Trying to install {pkg}…")
            _pip("install", pkg)
            importlib.invalidate_caches()
            if not _have(mod):
                print(f"Optional '{mod}' still unavailable after install; continuing without it.")
        except subprocess.CalledProcessError:
            print(f"Optional '{mod}' could not be installed; continuing without it.")

    # 5) FFmpeg notice
    if shutil.which("ffmpeg") is None:
        print("Warning: FFmpeg not found on PATH. WAV/FLAC will work; MP3/M4A may fail. "
              "Install FFmpeg and restart the terminal.")

# Run the check *before* importing project modules
ensure_dependencies()

# ---------------------- delegate to main CLI ----------------------

# Ensure we can import local main.py when running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import traceback
def _run() -> None:
    try:
        from main import main as cli_main
    except Exception:
        print("Failed to import the project's main CLI. Full traceback:")
        traceback.print_exc()
        sys.exit(1)
    cli_main(sys.argv[1:])

if __name__ == "__main__":
    _run()
