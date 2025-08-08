"""Fingerprint your music library using Dejavu.

This script builds a SQLite database of audio fingerprints.  Given a root
directory of audio files, it will walk through all subdirectories and
fingerprint files with supported extensions (.mp3, .wav, .flac, .ogg, .m4a).
The resulting database can later be used to identify tracks from short
audio snippets.

Usage:
    python fingerprint.py --music-dir /path/to/music

The database location is taken from config.yaml (key: database_path).  If
the file does not exist, it will be created automatically.  Re‑running
the script will skip already fingerprinted tracks.
"""

# Optional fingerprinting via PyDejavu. Safe on Py3.11 (no crash if missing).
import argparse
import sys
from pathlib import Path
import yaml

DEJAVU_OK = True
DJV_ERR = None
try:
    from dejavu import Dejavu
    from dejavu.recognize import FileRecognizer
except Exception as e:
    DEJAVU_OK = False
    DJV_ERR = e

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    if not DEJAVU_OK:
        print("PyDejavu not available. Skipping fingerprinting.\n", DJV_ERR)
        return

    cfg = load_config(args.config)
    music_dir = Path(cfg.get("music_dir") or "")
    if not music_dir.exists():
        print("music_dir not set or missing.")
        return

    # Example minimal Dejavu usage with SQLite (adapt to your config if needed)
    db_path = Path(cfg.get("fingerprint_db", "data/fingerprints.db"))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    config = {"database": {"type": "sqlite", "connection": str(db_path)}}

    djv = Dejavu(config)
    print("Fingerprinting directory… this may take a while.")
    djv.fingerprint_directory(str(music_dir), [".mp3", ".flac", ".wav", ".m4a"])

if __name__ == "__main__":
    main()
