"""End‑to‑end playlist generation.

This script identifies the current track either from an audio snippet
(embedding match) or directly from a provided file path, selects
follow‑up tracks via the sequencing engine and optionally generates
natural‑language explanations via the LLM configured in `config.yaml`.

Usage:
    python generate_playlist.py --input-audio /path/to/clip.wav --length 20 --policy smooth
    python generate_playlist.py --start-path "/path/to/song.flac" --length 20 --policy contrast

Exactly one of --input-audio or --start-path must be provided.
"""
# scripts/generate_playlist.py — FAISS search + playlist (CLAP_Module + PyTorch 2.6+ allowlist + NumPy 2.x polyfill)

import argparse
import json
from pathlib import Path

import yaml
import numpy as np
import faiss
import soundfile as sf
import librosa


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_mono(path: Path, target_sr: int | None = None):
    y, sr = sf.read(str(path), always_2d=False)
    if isinstance(y, np.ndarray) and y.ndim > 1:
        y = y.mean(axis=1)
    if target_sr and sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y.astype("float32"), sr


def _polyfill_numpy_random_integers():
    try:
        if not hasattr(np.random, "integers"):
            np.random.integers = np.random.randint  # type: ignore[attr-defined]
    except Exception:
        pass


def _allowlist_torch_pickle_globals():
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
        import numpy.dtypes as ndt
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
        to_allow.append(np.core.multiarray.scalar)  # legacy helper
    except Exception:
        pass
    try:
        add_safe_globals([x for x in to_allow if x is not None])  # type: ignore[misc]
    except Exception:
        pass


def _load_clap():
    _polyfill_numpy_random_integers()
    _allowlist_torch_pickle_globals()
    import laion_clap
    model = laion_clap.CLAP_Module(enable_fusion=False)
    model.load_ckpt()
    return model


def _embed_with_model(model, y: np.ndarray, sr: int) -> np.ndarray:
    x = y.reshape(1, -1).astype("float32")
    emb = model.get_audio_embedding_from_data(x=x, use_tensor=False)
    emb = np.asarray(emb)
    if emb.ndim == 2:
        emb = emb[0]
    return emb.reshape(-1)


def _embed_clip(model, clip_path: Path) -> np.ndarray:
    y, sr = read_mono(clip_path, target_sr=48000)
    emb = _embed_with_model(model, y, sr)
    return emb.astype("float32")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input-audio", type=str, default="")
    ap.add_argument("--start-path", type=str, default="")
    ap.add_argument("--length", type=int, default=20)
    ap.add_argument("--policy", type=str, choices=["smooth", "contrast"], default="smooth")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    index_path = Path(cfg.get("faiss_index_file", "data/faiss.index"))
    map_path = Path(cfg.get("mapping_file", "data/embedding_mapping.json"))

    if not index_path.exists() or not map_path.exists():
        print("Index/mapping missing. Run ingest first.")
        return

    index = faiss.read_index(str(index_path))
    with map_path.open("r", encoding="utf-8") as f:
        mapping = list(json.load(f))

    print("Loading CLAP (htsat-tiny)…")
    model = _load_clap()

    # Seed embedding
    if args.input_audio:
        seed_emb = _embed_clip(model, Path(args.input_audio))
    elif args.start_path:
        seed_emb = _embed_clip(model, Path(args.start_path))
    else:
        print("Need --input-audio or --start-path.")
        return

    seed = seed_emb.reshape(1, -1).astype("float32")
    faiss.normalize_L2(seed)

    # Retrieve candidates
    k = min(max(args.length * 5, 50), len(mapping))
    D, I = index.search(seed, k)
    I = I[0].tolist()
    D = D[0].tolist()

    cand = []
    for idx, score in zip(I, D):
        if idx < 0:
            continue
        cand.append((mapping[idx], float(score)))

    out = []
    if args.policy == "smooth":
        seen = set()
        for p, s in sorted(cand, key=lambda x: x[1], reverse=True):
            if p in seen:
                continue
            seen.add(p)
            out.append((p, s))
            if len(out) >= args.length:
                break
    else:
        # contrast: alternate high/low similarity
        cand_sorted = sorted(cand, key=lambda x: x[1], reverse=True)
        lo_sorted = list(reversed(cand_sorted))
        hi_i = lo_i = 0
        seen = set()
        while len(out) < args.length and (hi_i < len(cand_sorted) or lo_i < len(lo_sorted)):
            if hi_i < len(cand_sorted):
                p, s = cand_sorted[hi_i]; hi_i += 1
                if p not in seen:
                    out.append((p, s)); seen.add(p)
                    if len(out) >= args.length: break
            if lo_i < len(lo_sorted):
                p, s = lo_sorted[lo_i]; lo_i += 1
                if p not in seen:
                    out.append((p, s)); seen.add(p)

    print("\nPlaylist:")
    for i, (p, s) in enumerate(out, 1):
        print(f"{i:2d}. {p}  (sim={s:.3f})")


if __name__ == "__main__":
    main()


