"""Build a FAISS index from CLAP embeddings.

This script loads the embedding matrix saved by `embed_clap.py`, normalises
each vector to unit length, builds a FAISS inner‑product index and writes
it to disk.  You only need to run this script when embeddings change.

The path of the embeddings and the output index are controlled via
`config.yaml` (keys: `embedding_file`, `faiss_index_file`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml
import numpy as np

try:
    import faiss
except ImportError:
    print("faiss-cpu is required. Install it via pip install faiss-cpu or conda install faiss-cpu.", file=sys.stderr)
    raise


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
        cfg = yaml.safe_load(f)
    embedding_file = Path(cfg.get("embedding_file", "data/embeddings.npy"))
    index_file = Path(cfg.get("faiss_index_file", "data/faiss.index"))
    index_file.parent.mkdir(parents=True, exist_ok=True)

    if not embedding_file.exists():
        print(f"Embeddings file {embedding_file} does not exist. Run embed_clap.py first.")
        return
    print(f"Loading embeddings from {embedding_file}…")
    embeddings = np.load(embedding_file)
    # Normalise to unit length for cosine similarity
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    print("Adding embeddings to index…")
    index.add(embeddings)
    faiss.write_index(index, str(index_file))
    print(f"FAISS index saved to {index_file}")


if __name__ == "__main__":
    main()