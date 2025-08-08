"""Sequencing engine for playlist generation.

This module defines functions to build playlists that transition smoothly
between tracks or deliberately inject contrast according to a policy.  It
operates on precomputed embeddings and features.  The cost function is
parameterised by weights specified in `config.yaml`.

Usage examples (see generate_playlist.py for integration):

    from sequencer import Sequencer
    seq = Sequencer(cfg)
    playlist = seq.generate(start_idx=42, length=20, policy='smooth')
    # playlist is a list of track indices into the embedding matrix

The sequencer will automatically load features, embeddings and the FAISS
index when constructed.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import yaml

try:
    import faiss
except ImportError:
    print("faiss is required. Install faiss-cpu via pip or conda.", file=sys.stderr)
    raise


def circle_of_fifths_distance(k1: int, k2: int) -> int:
    """Return the distance between two keys on the circle of fifths (0–11)."""
    diff = abs(k1 - k2)
    return min(diff, 12 - diff)


@dataclass
class TrackInfo:
    style_vec: np.ndarray  # normalised style vector (embedding + tags)
    mood_vec: np.ndarray   # [valence, arousal]
    bpm: float
    key: int
    energy: float


class Sequencer:
    def __init__(self, config: dict) -> None:
        """Load embeddings, features and FAISS index according to config."""
        self.cfg = config
        root = Path(__file__).resolve().parent.parent
        # Load features
        feature_file = root / self.cfg.get("feature_file", "data/features.json")
        if not feature_file.exists():
            raise FileNotFoundError(f"Feature file {feature_file} not found. Run extract_features.py")
        with open(feature_file, "r", encoding="utf-8") as f:
            features_dict: Dict[str, dict] = json.load(f)
        # Load embeddings
        embedding_file = root / self.cfg.get("embedding_file", "data/embeddings.npy")
        if not embedding_file.exists():
            raise FileNotFoundError(f"Embedding file {embedding_file} not found. Run embed_clap.py")
        embeddings = np.load(embedding_file)
        faiss.normalize_L2(embeddings)
        # Load mapping file to map index to path and vice versa
        mapping_file = root / self.cfg.get("mapping_file", "data/embedding_mapping.txt")
        if not mapping_file.exists():
            raise FileNotFoundError(f"Mapping file {mapping_file} not found. Run embed_clap.py")
        with open(mapping_file, "r", encoding="utf-8") as f:
            mapping_list = [line.strip() for line in f if line.strip()]
        if len(mapping_list) != embeddings.shape[0]:
            raise ValueError("Number of embeddings does not match number of mapping lines")

        # Build tag vocabulary
        tags_set = set()
        for data in features_dict.values():
            tags_set.update(data["tags"].keys())
        self.tag_list = sorted(tags_set)
        tag_index = {t: i for i, t in enumerate(self.tag_list)}
        n_tags = len(self.tag_list)

        # Precompute track info list
        self.tracks: List[TrackInfo] = []
        for idx, path in enumerate(mapping_list):
            fdata = features_dict.get(path)
            if not fdata:
                # Skip tracks missing features
                continue
            # Build tag vector
            tag_vec = np.zeros(n_tags, dtype=np.float32)
            for tag, score in fdata["tags"].items():
                tag_vec[tag_index[tag]] = float(score)
            # Concatenate embedding and tag vector
            style = np.concatenate([embeddings[idx], tag_vec])
            # Normalise style vector
            norm = np.linalg.norm(style)
            if norm > 0:
                style = style / norm
            # Mood vector (valence, arousal).  Use defaults if missing
            valence = float(fdata.get("valence", 0.5))
            arousal = float(fdata.get("arousal", 0.5))
            mood = np.array([valence, arousal], dtype=np.float32)
            track_info = TrackInfo(
                style_vec=style,
                mood_vec=mood,
                bpm=float(fdata.get("bpm", 120.0)),
                key=int(fdata.get("key", 0)),
                energy=float(fdata.get("energy", 0.5)),
            )
            self.tracks.append(track_info)
        # Keep mapping list for later retrieval
        self.mapping_list = mapping_list

        # Load FAISS index
        index_file = root / self.cfg.get("faiss_index_file", "data/faiss.index")
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index {index_file} not found. Run build_faiss_index.py")
        self.faiss_index = faiss.read_index(str(index_file))

    def cost(self, i: int, j: int, weights: dict) -> float:
        """Compute the weighted cost from track i to track j."""
        a = self.tracks[i]
        b = self.tracks[j]
        # Style distance: 1 - cosine similarity of style vectors
        sim = float(np.dot(a.style_vec, b.style_vec))
        style_dist = 1.0 - sim
        # Mood difference (Euclidean)
        mood_diff = float(np.linalg.norm(a.mood_vec - b.mood_vec))
        # Tempo difference
        tempo_diff = abs(a.bpm - b.bpm)
        # Key difference (circle of fifths)
        key_diff = circle_of_fifths_distance(a.key, b.key)
        # Energy difference
        energy_diff = abs(a.energy - b.energy)
        return (
            weights["style"] * style_dist
            + weights["mood"] * mood_diff
            + weights["tempo"] * tempo_diff
            + weights["key"] * key_diff
            + weights["energy"] * energy_diff
        )

    def generate(
        self,
        start_idx: int,
        length: int,
        policy: str = "smooth",
    ) -> List[int]:
        """Generate a playlist starting from index `start_idx` of given length.

        Parameters:
            start_idx: Index of the starting track (0‑based into mapping_list)
            length: Number of tracks to include (including the starting track)
            policy: 'smooth' or 'contrast'

        Returns:
            List of track indices (relative to mapping_list)
        """
        if start_idx < 0 or start_idx >= len(self.tracks):
            raise ValueError("start_idx out of range")
        playlist = [start_idx]
        used = set(playlist)

        w_base = self.cfg.get("sequencing", {}).get("weights", {})
        # Convert to floats to avoid repeated lookups
        base_weights = {k: float(v) for k, v in w_base.items()}
        # Thresholds for contrast mode
        seq_cfg = self.cfg.get("sequencing", {})
        contrast_every = int(seq_cfg.get("contrast_every", 0))
        mood_thr = float(seq_cfg.get("mood_threshold", 1.0))
        tempo_thr = float(seq_cfg.get("tempo_threshold", 9999))
        key_thr = int(seq_cfg.get("key_threshold", 12))
        energy_thr = float(seq_cfg.get("energy_threshold", 1.0))

        for step in range(1, length):
            curr = playlist[-1]
            # Determine if we are in contrast slot
            in_contrast = (
                policy == "contrast" and contrast_every > 0 and (step % contrast_every == 0)
            )
            # Build candidate set: all tracks not used
            candidates = [i for i in range(len(self.tracks)) if i not in used]
            if not candidates:
                break
            if in_contrast:
                # In contrast mode, invert style weight to encourage dissimilarity
                weights = base_weights.copy()
                weights["style"] = -abs(weights["style"])
                # Filter candidates by thresholds
                filtered = []
                a = self.tracks[curr]
                for j in candidates:
                    b = self.tracks[j]
                    mood_diff = float(np.linalg.norm(a.mood_vec - b.mood_vec))
                    tempo_diff = abs(a.bpm - b.bpm)
                    key_diff = circle_of_fifths_distance(a.key, b.key)
                    energy_diff = abs(a.energy - b.energy)
                    if (
                        mood_diff <= mood_thr
                        and tempo_diff <= tempo_thr
                        and key_diff <= key_thr
                        and energy_diff <= energy_thr
                    ):
                        filtered.append(j)
                candidates = filtered if filtered else candidates
            else:
                weights = base_weights

            # Choose candidate with minimum cost
            best = None
            best_cost = math.inf
            for j in candidates:
                c = self.cost(curr, j, weights)
                if c < best_cost:
                    best = j
                    best_cost = c
            if best is None:
                break
            playlist.append(best)
            used.add(best)
        return playlist

    def find_nearest_by_path(self, path: str, top_k: int = 5) -> List[int]:
        """Given a track path, return indices of similar tracks via FAISS.

        Useful for selecting starting points when the user supplies a track
        path instead of an audio snippet.
        """
        try:
            idx = self.mapping_list.index(path)
        except ValueError:
            raise ValueError(f"Track {path} not found in embedding mapping")
        # The first result returned by FAISS will be the query itself (index idx), so skip it
        D, I = self.faiss_index.search(self.tracks[idx].style_vec[None, :], top_k + 1)
        return [int(i) for i in I[0] if i != idx]