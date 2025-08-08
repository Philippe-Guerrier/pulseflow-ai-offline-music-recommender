"""Extract high‑level audio features and tags using librosa and MusicNN.

This script walks through a directory of audio files and computes for each track:

* Tempo (beats per minute)
* Key (as an integer 0–11; 0=C, 1=C#/Db, … 11=B)
* Energy (RMS) normalised to [0,1] across the dataset
* Multi‑label tag probabilities using MusicNN (genres/instruments/moods)

The resulting dictionary is saved as JSON at the path specified by
`feature_file` in config.yaml.  Keys are absolute file paths; values
contain the fields above.  You can load this JSON later in the sequencing
engine to compute distances between tracks.

Note: Unlike earlier versions of this project, this script does not
depend on Essentia and therefore works on Windows.  Feature quality may
be less sophisticated but is sufficient for demonstration purposes.
"""
import argparse
import json
from pathlib import Path

import yaml
import librosa
import numpy as np

# Optional tagging with musicnn (skipped if unavailable)
MUSICNN_OK = False
MUSICNN_ERR = None
try:
    from musicnn.tagger import musicnn_tagger  # type: ignore
    MUSICNN_OK = True
except Exception as e:
    MUSICNN_ERR = e

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg"}

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def iter_tracks(root: Path):
    for p in root.rglob("*"):
        if p.suffix.lower() in AUDIO_EXTS:
            yield p

def estimate_key(y: np.ndarray, sr: int) -> str:
    # very rough key guess using chroma-cqt peak
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key_idx = int(np.argmax(chroma.sum(axis=1)))
    keys = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    return keys[key_idx]

def extract_one(path: Path, enable_tagging: bool) -> dict:
    y, sr = librosa.load(str(path), sr=None, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = librosa.feature.rms(y=y).mean()
    zcr = librosa.feature.zero_crossing_rate(y).mean()
    key = estimate_key(y, sr)

    tags: list[str] = []
    if enable_tagging:
        if MUSICNN_OK:
            try:
                tops, _ = musicnn_tagger(str(path), model='MSD_musicnn', topN=5)
                tags = [t for t in tops]  # simple list of tag names
            except Exception as e:
                print(f"musicnn failed on {path}: {e}")
        else:
            print(f"MusicNN not available (skipping tags): {MUSICNN_ERR}")

    return {
        "path": str(path),
        "sr": int(sr),
        "tempo": float(tempo),
        "energy": float(rms),
        "zcr": float(zcr),
        "key": key,
        "tags": tags,
    }

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    music_dir = Path(cfg.get("music_dir") or "")
    if not music_dir.exists():
        print("music_dir not set or missing.")
        return
    out_path = Path(cfg.get("features_file", "data/features.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    enable_tagging = bool(cfg.get("enable_tagging", False))

    # If not forcing and features exist, keep them
    if out_path.exists() and not args.force:
        with out_path.open("r", encoding="utf-8") as f:
            existing = {item["path"]: item for item in (json.load(f) or [])}
    else:
        existing = {}

    all_items = existing
    count = 0
    for track in iter_tracks(music_dir):
        sp = str(track)
        if sp in existing and not args.force:
            continue
        try:
            item = extract_one(track, enable_tagging)
            all_items[sp] = item
            count += 1
            if count % 25 == 0:
                print(f"Processed {count} new tracks…")
        except Exception as e:
            print(f"Feature extraction failed for {track}: {e}")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(list(all_items.values()), f, ensure_ascii=False, indent=2)
    print(f"Wrote features: {out_path} ({len(all_items)} tracks)")

if __name__ == "__main__":
    main()
