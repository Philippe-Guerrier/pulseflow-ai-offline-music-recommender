"""Top‑level command‑line interface for the music recommendation system.

This script centralises the various steps of the pipeline.  Run

    python main.py ingest

to fingerprint your library, extract features, compute embeddings and build
the FAISS index in one go.  To generate a playlist, run

    python main.py playlist --input-audio clip.wav --length 20 --policy smooth

All options default to values defined in config.yaml.  Use --help after
each subcommand for more details.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

"""
from scripts import fingerprint as fp
from scripts import extract_features as ef
from scripts import embed_clap as ec
from scripts import build_faiss_index as fi
from scripts import generate_playlist as gp
"""

import argparse
import json
import os
import sys
from pathlib import Path

import importlib.util as _ius

# lightweight deps (ensured by run.py)
import yaml
import numpy as np
import soundfile as sf

CONFIG_DEFAULT = "config.yaml"

def load_config(cfg_path: str | Path) -> dict:
    p = Path(cfg_path or CONFIG_DEFAULT)
    if not p.exists():
        print(f"Config file not found: {p}")
        sys.exit(1)
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # expand data_dir and join file names
    data_dir = Path(cfg.get("data_dir", "data"))
    cfg["_data_dir"] = data_dir
    cfg["features_file"] = str(data_dir / cfg.get("features_file", "features.json"))
    cfg["embeddings_file"] = str(data_dir / cfg.get("embeddings_file", "embeddings.npy"))
    cfg["mapping_file"] = str(data_dir / cfg.get("mapping_file", "embedding_mapping.json"))
    cfg["faiss_index_file"] = str(data_dir / cfg.get("faiss_index_file", "faiss.index"))
    cfg["fingerprint_db"] = str(data_dir / cfg.get("fingerprint_db", "fingerprints.db"))
    return cfg

def ensure_dirs(cfg: dict) -> None:
    Path(cfg["_data_dir"]).mkdir(parents=True, exist_ok=True)

def _make_snippet(src_path: Path, out_path: Path, seconds: float = 8.0) -> Path:
    """Create a short mono WAV snippet from a track for identification."""
    try:
        audio, sr = sf.read(str(src_path), always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        n = int(seconds * sr)
        clip = audio[:n]
        sf.write(str(out_path), clip, sr)
        return out_path
    except Exception as e:
        print(f"Failed to create snippet from {src_path}: {e}")
        return src_path  # fallback: use original

# ------------------------ Commands ------------------------

def ingest(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    music_dir = Path(cfg.get("music_dir") or "")
    if not music_dir.exists():
        print("Error: 'music_dir' not set or does not exist in config.yaml.")
        sys.exit(1)

    # Fingerprint only if dejavu is importable
    if _ius.find_spec("dejavu") is not None:
        print("Fingerprinting with PyDejavu…")
        from scripts import fingerprint as fp  # lazy
        fp.main(["--config", str(args.config)] + (["--force"] if args.force else []))
    else:
        print("PyDejavu not available; skipping fingerprinting (embedding-only recognition).")

    # Features -> Embeddings -> FAISS (always run)
    from scripts import extract_features as ef
    from scripts import embed_clap as ec
    from scripts import build_faiss_index as fi

    ef.main(["--config", str(args.config)] + (["--force"] if args.force else []))
    ec.main(["--config", str(args.config)] + (["--force"] if args.force else []))
    fi.main(["--config", str(args.config)])

    print("\nIngest pipeline completed.")

def playlist(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    input_audio = args.input_audio or cfg.get("default_input_audio", "") or ""
    start_path = args.start_path or cfg.get("default_start_path", "") or ""
    policy = args.policy or cfg.get("default_policy", "smooth")
    length = args.length or int(cfg.get("playlist_length", 20))

    if not input_audio and start_path:
        # auto-snippet from start track
        src = Path(start_path)
        out = Path(cfg["_data_dir"]) / "auto_snippet.wav"
        input_audio = str(_make_snippet(src, out, seconds=8.0))

    if not input_audio and not start_path:
        print("No --input-audio or --start-path, and no default_* in config.yaml.")
        sys.exit(2)

    from scripts import generate_playlist as gp  # lazy
    gp.main([
        "--config", str(args.config),
        "--length", str(length),
        "--policy", policy
    ] + (["--input-audio", input_audio] if input_audio else ["--start-path", start_path]))

def ui(args: argparse.Namespace) -> None:
    """Interactive file picker rooted at music_dir, with auto-ingest on first run."""
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    music_dir = Path(cfg.get("music_dir") or "")
    if not music_dir.exists():
        print("Error: 'music_dir' not set or does not exist in config.yaml.")
        sys.exit(1)

    # Auto-ingest if FAISS index or mapping is missing
    index_path = Path(cfg.get("faiss_index_file", "data/faiss.index"))
    map_path = Path(cfg.get("mapping_file", "data/embedding_mapping.json"))
    if not index_path.exists() or not map_path.exists():
        print("Index/mapping missing — running ingest steps now (features → embeddings → index)…")
        from scripts import extract_features as ef
        from scripts import embed_clap as ec
        from scripts import build_faiss_index as fi
        # Reuse any existing partial outputs (no --force)
        ef.main(["--config", str(args.config)])
        ec.main(["--config", str(args.config)])
        fi.main(["--config", str(args.config)])
        print("Ingest steps completed.\n")

    # Minimal GUI for selecting a start track
    try:
        import tkinter as tk
        from tkinter import filedialog, simpledialog
    except Exception as e:
        print(f"Tkinter not available: {e}")
        sys.exit(1)

    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select start track",
        initialdir=str(music_dir),
        filetypes=[("Audio", "*.wav *.flac *.mp3 *.m4a *.aac *.ogg"), ("All", "*.*")]
    )
    if not path:
        print("No file selected.")
        return

    # Ask for playlist params (defaults from config)
    length_default = int(cfg.get("playlist_length", 20))
    policy_default = cfg.get("default_policy", "smooth")
    try:
        length = simpledialog.askinteger("Playlist length", f"Length (default {length_default}):")
        if not length:
            length = length_default
    except Exception:
        length = length_default

    try:
        policy = simpledialog.askstring("Policy", f"Policy 'smooth' or 'contrast' (default {policy_default}):")
        if not policy:
            policy = policy_default
    except Exception:
        policy = policy_default

    # Auto-create an 8s snippet from the chosen track
    out = Path(cfg["_data_dir"]) / "auto_snippet.wav"
    input_audio = str(_make_snippet(Path(path), out, seconds=8.0))

    from scripts import generate_playlist as gp
    gp.main([
        "--config", str(args.config),
        "--length", str(length),
        "--policy", policy,
        "--input-audio", input_audio
    ])


# ------------------------ CLI ------------------------

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="music-rec", description="Local music recommender")
    p.add_argument("--config", type=str, default=CONFIG_DEFAULT, help="Path to config.yaml")

    sp = p.add_subparsers(dest="cmd", required=True)

    pi = sp.add_parser("digest", help="(alias) see 'ingest'")
    pi.set_defaults(func=ingest)
    pi.add_argument("--force", action="store_true", help="Recompute everything")

    p_ing = sp.add_parser("ingest", help="Build features/embeddings/index")
    p_ing.set_defaults(func=ingest)
    p_ing.add_argument("--force", action="store_true", help="Recompute everything")

    p_pl = sp.add_parser("playlist", help="Generate a playlist")
    g = p_pl.add_mutually_exclusive_group(required=False)
    g.add_argument("--input-audio", type=str, help="Seed clip (WAV/FLAC/MP3…)")
    g.add_argument("--start-path", type=str, help="Seed track path in library")
    p_pl.add_argument("--length", type=int, default=None, help="Playlist length (overrides config)")
    p_pl.add_argument("--policy", type=str, choices=["smooth", "contrast"], default=None)
    p_pl.set_defaults(func=playlist)

    p_ui = sp.add_parser("interface", help="Interactive picker UI")
    p_ui.set_defaults(func=ui)

    args = p.parse_args(argv)
    args.func(args)

if __name__ == "__main__":
    main()
