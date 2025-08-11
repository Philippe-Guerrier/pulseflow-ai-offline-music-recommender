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

from pathlib import Path
from scripts import explanation as ex
from main import load_config

import argparse
import json
from pathlib import Path

import yaml
import numpy as np
import faiss
import soundfile as sf
import librosa

import io, os, tempfile

# --- stringify safety for LLM fields ---
def _coerce_text(val):
    if isinstance(val, (str, type(None))):
        return val or ""
    try:
        # last-resort to readable JSON
        import json as _json
        return _json.dumps(val, ensure_ascii=False)
    except Exception:
        return str(val)

def _sanitize_payload(payload: dict):
    # summary strings
    s = payload.get("summary")
    if isinstance(s, dict):
        for k in ("top_pick", "overview", "reason", "note"):
            if k in s:
                s[k] = _coerce_text(s[k])
    # per-track strings
    for t in payload.get("tracks", []):
        if "title" in t: t["title"] = _coerce_text(t["title"])
        if "explanation" in t: t["explanation"] = _coerce_text(t["explanation"])



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


def _load_index_and_mapping(cfg):
    index_path = Path(cfg.get("faiss_index_file", "data/faiss.index"))
    map_path = Path(cfg.get("mapping_file", "data/embedding_mapping.txt"))
    if not index_path.exists() or not map_path.exists():
        raise FileNotFoundError("Index/mapping missing. Run ingest first.")
    index = faiss.read_index(str(index_path))
    # mapping: support .json or .txt
    try:
        if map_path.suffix.lower() == ".json":
            with map_path.open("r", encoding="utf-8") as f:
                mapping = list(json.load(f))
        else:
            with map_path.open("r", encoding="utf-8") as f:
                mapping = [ln.strip() for ln in f if ln.strip()]
    except Exception as e:
        raise RuntimeError(f"Failed to load mapping {map_path}: {e}")
    return index, mapping

def _embed_bytes(model, blob: bytes) -> np.ndarray:
    # Try direct decode (wav/flac/ogg)
    try:
        buf = io.BytesIO(blob)
        y, sr = sf.read(buf, always_2d=False)
        if isinstance(y, np.ndarray) and y.ndim > 1:
            y = y.mean(axis=1)
    except Exception:
        # Fallback for mp3/m4a: write temp then use librosa/audioread
        with tempfile.NamedTemporaryFile(suffix=".tmp", delete=False) as tmp:
            tmp.write(blob)
            tmp_path = tmp.name
        try:
            y, sr = librosa.load(tmp_path, sr=None, mono=True)
        finally:
            try: os.unlink(tmp_path)
            except Exception: pass
    if sr != 48000:
        y = librosa.resample(y, orig_sr=sr, target_sr=48000)
        sr = 48000
    return _embed_with_model(model, y.astype("float32"), sr).astype("float32")

def _select_tracks(index, mapping, seed_emb: np.ndarray, length: int, policy: str):
    seed = seed_emb.reshape(1, -1).astype("float32")
    faiss.normalize_L2(seed)
    k = min(max(length * 5, 50), len(mapping))
    D, I = index.search(seed, k)
    I = I[0].tolist(); D = D[0].tolist()
    cand = [(mapping[i], float(d)) for i, d in zip(I, D) if i >= 0]

    out = []
    if policy == "smooth":
        seen = set()
        for p, s in sorted(cand, key=lambda x: x[1], reverse=True):
            if p in seen: continue
            seen.add(p); out.append((p, s))
            if len(out) >= length: break
    else:
        hi = sorted(cand, key=lambda x: x[1], reverse=True)
        lo = list(reversed(hi))
        hi_i = lo_i = 0; seen = set()
        while len(out) < length and (hi_i < len(hi) or lo_i < len(lo)):
            if hi_i < len(hi):
                p, s = hi[hi_i]; hi_i += 1
                if p not in seen:
                    out.append((p, s)); seen.add(p)
                    if len(out) >= length: break
            if lo_i < len(lo):
                p, s = lo[lo_i]; lo_i += 1
                if p not in seen:
                    out.append((p, s)); seen.add(p)
    return out


def _attach_llm_explanations(playlist: list[dict], policy: str, cfg: dict) -> dict[str, str]:
    """Attach per-transition explanations and return a global summary dict."""
    summary: dict[str, str] = {}
    llm_cfg = cfg.get("llm", {})
    if not llm_cfg.get("enabled", True):
        return summary

    llm_model = llm_cfg.get("model", "qwen2.5:3b")
    maxt = int(llm_cfg.get("max_tokens", 160))
    temp = float(llm_cfg.get("temperature", 0.7))

    # Per-transition (attach to NEXT track)
    for i in range(len(playlist) - 1):
        stats = playlist[i + 1].get("stats", {}) or {}
        try:
            txt = ex.explain_transition(playlist[i], playlist[i + 1], stats, llm_model, maxt, temp)
        except Exception:
            txt = ""
        playlist[i + 1]["explanation"] = txt

    # Global summary (top pick + overview)
    try:
        summary = ex.explain_playlist_summary(playlist, policy, llm_model, maxt, temp)
    except Exception:
        summary = {}

    return summary


def generate_from_bytes(audio_bytes: bytes, length: int = 20, policy: str = "smooth",
                        explain: bool = False, cfg_path: str = "config.yaml") -> dict:
    cfg = load_config(cfg_path)
    index, mapping = _load_index_and_mapping(cfg)
    model = _load_clap()
    seed_emb = _embed_bytes(model, audio_bytes)
    pairs = _select_tracks(index, mapping, seed_emb, length, policy)
    playlist = [{"path": p, "score": s, "title": Path(p).stem, "stats": {}} for p, s in pairs]

    summary: Dict[str, str] = {}
    if explain:
        summary = _attach_llm_explanations(playlist, policy, cfg)

    result = {
        "seed": {"source": "clip"},
        "policy": policy,
        "count": len(playlist),
        "tracks": playlist,
        "summary": summary,
    }
    _sanitize_payload(result)
    return result


def generate_from_path(start_path: str, length: int = 20, policy: str = "smooth",
                       explain: bool = False, cfg_path: str = "config.yaml") -> dict:
    cfg = load_config(cfg_path)
    index, mapping = _load_index_and_mapping(cfg)
    model = _load_clap()
    y, sr = read_mono(Path(start_path), target_sr=48000)
    seed_emb = _embed_with_model(model, y, sr).astype("float32")
    pairs = _select_tracks(index, mapping, seed_emb, length, policy)
    playlist = [{"path": p, "score": s, "title": Path(p).stem, "stats": {}} for p, s in pairs]

    summary: Dict[str, str] = {}
    if explain:
        summary = _attach_llm_explanations(playlist, policy, cfg)

    result = {
        "seed": {"source": "path", "value": start_path},
        "policy": policy,
        "count": len(playlist),
        "tracks": playlist,
        "summary": summary,
    }
    _sanitize_payload(result)
    return result


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

    # Build playlist dicts from (path, score) pairs
    playlist = [
        {"path": p, "score": s, "title": Path(p).stem, "stats": {}}
        for (p, s) in out
    ]

    # LLM-forward output (per-transition + global summary)
    summary = {}
    try:
        llm_cfg = cfg.get("llm", {})
        if llm_cfg.get("enabled", True):
            model = llm_cfg.get("model", "qwen2.5:3b")
            maxt  = int(llm_cfg.get("max_tokens", 160))
            temp  = float(llm_cfg.get("temperature", 0.7))

            # Per-transition explanations (attach to NEXT track)
            for i in range(len(playlist) - 1):
                try:
                    stats = playlist[i+1].get("stats", {}) or {}
                    txt = ex.explain_transition(playlist[i], playlist[i+1], stats, model, maxt, temp)
                    playlist[i+1]["explanation"] = _coerce_text(txt)
                except Exception:
                    playlist[i+1]["explanation"] = ""

            # Global summary (top pick + overview)
            try:
                summary = ex.explain_playlist_summary(playlist, args.policy, model, maxt, temp)
            except Exception:
                summary = {}
    except Exception as e:
        print(f"LLM explanations skipped: {e}")

    # Pretty print to console
    if summary:
        print("\nTop pick:", summary.get("top_pick", ""))
        print("Overview:", summary.get("overview", ""))

    print("\nPlaylist:")
    for i, t in enumerate(playlist, 1):
        print(f"{i:2d}. {t['title']}  (sim={t['score']:.3f})")
        if t.get("explanation"):
            print(f"    → {t['explanation']}")

    # Also write JSON payload to disk for the UI / export
    out_json = {"tracks": playlist, "summary": summary}
    out_path = Path(cfg.get("last_playlist_file", "data/last_playlist.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    print(f"\nSaved JSON: {out_path}")


if __name__ == "__main__":
    main()


