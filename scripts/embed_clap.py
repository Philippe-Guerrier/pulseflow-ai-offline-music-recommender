"""Compute CLAP embeddings for your music library.

This script iterates over all audio files in a given directory (recursively),
computes a 512‑dimensional CLAP embedding for each track and writes two
artefacts:

* a NumPy array of shape (n_tracks, 512) containing embeddings
* a mapping file that associates each row of the embedding array with
  the absolute path of the corresponding audio file

Embeddings are stored in the location specified by `embedding_file` in
config.yaml.  The mapping file is stored in `mapping_file` in config.yaml.

You should run this script after fingerprinting and feature extraction.  It
requires a GPU for optimal performance but will fall back to CPU if none is
available.
"""
# scripts/embed_clap.py — build CLAP embeddings (CLAP_Module + PyTorch 2.6+ safe load allowlist + NumPy 2.x polyfill)

import argparse
import json
from pathlib import Path
from typing import Iterator

import yaml
import numpy as np
import soundfile as sf
import librosa


AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg"}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def iter_tracks(root: Path) -> Iterator[Path]:
    for p in root.rglob("*"):
        if p.suffix.lower() in AUDIO_EXTS:
            yield p


def read_mono(path: Path, target_sr: int | None = None):
    y, sr = sf.read(str(path), always_2d=False)
    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = y.mean(axis=1)
    if target_sr and sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y.astype("float32"), sr


def _polyfill_numpy_random_integers():
    """NumPy 2.x removed np.random.integers; some deps still call it."""
    try:
        if not hasattr(np.random, "integers"):
            np.random.integers = np.random.randint  # type: ignore[attr-defined]
    except Exception:
        pass


def _allowlist_torch_pickle_globals():
    """Work around torch.load(weights_only=True) by allowlisting NumPy types/classes."""
    try:
        from torch.serialization import add_safe_globals  # PyTorch 2.6+
    except Exception:
        try:
            from torch.serialization import safe_globals as add_safe_globals  # type: ignore
        except Exception:
            return

    to_allow = [
        np.dtype,
        getattr(np, "bool_", None),
        getattr(np, "float32", None), getattr(np, "float64", None),
        getattr(np, "int8", None), getattr(np, "int16", None),
        getattr(np, "int32", None), getattr(np, "int64", None),
        getattr(np, "uint8", None), getattr(np, "uint16", None),
        getattr(np, "uint32", None), getattr(np, "uint64", None),
        getattr(np, "complex64", None), getattr(np, "complex128", None),
    ]
    try:
        import numpy.dtypes as ndt  # NumPy 2.x classes
        for name in [
            "BoolDType",
            "Float32DType", "Float64DType",
            "Int8DType", "Int16DType", "Int32DType", "Int64DType",
            "UInt8DType", "UInt16DType", "UInt32DType", "UInt64DType",
            "Complex64DType", "Complex128DType",
        ]:
            cls = getattr(ndt, name, None)
            if cls is not None:
                to_allow.append(cls)
    except Exception:
        pass
    try:
        to_allow.append(np.core.multiarray.scalar)  # legacy scalar used in some ckpts
    except Exception:
        pass
    to_allow = [x for x in to_allow if x is not None]
    try:
        add_safe_globals(to_allow)  # type: ignore[misc]
    except Exception:
        pass


def _load_clap():
    """
    Use the stable pip API: laion_clap.CLAP_Module + load_ckpt().
    First run downloads weights to cache.
    """
    _polyfill_numpy_random_integers()
    _allowlist_torch_pickle_globals()
    import laion_clap
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    return model


def _embed_with_model(model, y: np.ndarray, sr: int) -> np.ndarray:
    """CLAP expects 48kHz audio and (N, T) float32. Return a 1D vector."""
    x = y.reshape(1, -1).astype("float32")
    emb = model.get_audio_embedding_from_data(x=x, use_tensor=False)
    emb = np.asarray(emb)
    if emb.ndim == 2:
        emb = emb[0]
    return emb.reshape(-1)


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

    emb_path = Path(cfg.get("embeddings_file", "data/embeddings.npy"))
    map_path = Path(cfg.get("mapping_file", "data/embedding_mapping.json"))
    emb_path.parent.mkdir(parents=True, exist_ok=True)

    if emb_path.exists() and map_path.exists() and not args.force:
        print("Embeddings already exist; use --force to recompute.")
        return

    print("Loading CLAP (htsat-tiny)…")
    model = _load_clap()

    embs = []
    mapping = []
    count = 0
    for p in iter_tracks(music_dir):
        try:
            y, sr = read_mono(p, target_sr=48000)
            emb = _embed_with_model(model, y, sr).astype("float32")
            embs.append(emb)
            mapping.append(str(p))
            count += 1
            if count % 25 == 0:
                print(f"Embedded {count} tracks…")
        except Exception as e:
            print(f"Embedding failed for {p}: {e}")

    if not embs:
        print("No embeddings computed.")
        return

    embs_np = np.vstack(embs).astype("float32")
    np.save(emb_path, embs_np)
    with map_path.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"Wrote embeddings: {emb_path} shape={embs_np.shape}")
    print(f"Wrote mapping: {map_path} ({len(mapping)} paths)")


if __name__ == "__main__":
    main()



