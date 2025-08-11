"""Build a FAISS index from CLAP embeddings.

This script loads the embedding matrix saved by `embed_clap.py`, normalises
each vector to unit length, builds a FAISS inner-product index and writes
it to disk. You only need to run this script when embeddings change.

Paths come from `config.yaml` (keys: `embedding_file`, `faiss_index_file`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
import numpy as np

# Single import path for faiss with a helpful error if missing
try:
    import faiss
except ImportError:
    print(
        "faiss-cpu is required. Install it via:\n"
        "  pip install faiss-cpu\n"
        "or\n"
        "  conda install -c conda-forge faiss-cpu",
        file=sys.stderr,
    )
    raise


def _to_gpu_if_available(index_cpu):
    """Move a CPU FAISS index to GPU 0 if available; otherwise return CPU index."""
    try:
        if hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            dev = 0  # GPU 0
            index_gpu = faiss.index_cpu_to_gpu(res, dev, index_cpu)
            print("Using FAISS-GPU on device 0")
            return index_gpu, res
    except Exception as e:
        print(f"FAISS-GPU unavailable, falling back to CPU. Reason: {e}")
    return index_cpu, None


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS index from CLAP embeddings.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "config.yaml",
        help="Path to configuration YAML file",
    )
    args = parser.parse_args(argv)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    embedding_file = Path(cfg.get("embedding_file", "data/embeddings.npy"))
    index_file = Path(cfg.get("faiss_index_file", "data/faiss.index"))
    index_file.parent.mkdir(parents=True, exist_ok=True)

    if not embedding_file.exists():
        print(f"Embeddings file {embedding_file} does not exist. Run embed_clap.py first.")
        return

    print(f"Loading embeddings from {embedding_file}…")
    embeddings = np.load(embedding_file)

    # Ensure float32 for FAISS ops, then L2-normalize for cosine via inner product
    embeddings = embeddings.astype("float32", copy=False)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index_cpu = faiss.IndexFlatIP(dim)

    index, gpu_res = _to_gpu_if_available(index_cpu)
    print("Adding embeddings to index…")
    index.add(embeddings)

    # Persist a CPU copy so it can be loaded without GPU
    cpu_copy = faiss.index_gpu_to_cpu(index) if gpu_res is not None else index
    faiss.write_index(cpu_copy, str(index_file))
    print(f"FAISS index saved to {index_file}")


if __name__ == "__main__":
    main()
