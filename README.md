# Local Music Recommendation System - Offline, Smooth, and Customizable

This is my end-to-end **offline** music identification and recommendation system.  
No cloud, no hidden data uploads - it runs entirely on your machine.  
It’s designed to avoid the usual Windows install headaches (no **Essentia** builds) and works on both GPU and CPU, but obviously flies faster with an NVIDIA GPU.

---

## What It Does

- **Identify tracks** from your local library via audio fingerprinting (using [PyDejavu](https://github.com/worldveil/dejavu))  
- **Understand your music’s “DNA”** with 512-dimensional CLAP embeddings that capture style and timbre  
- **Extract extra info**: tempo, key, energy, genre/mood/instrument tags (via `librosa` + [MusicNN](https://github.com/jordipons/musicnn))  
- **Find similar or contrasting tracks** instantly using a FAISS index  
- **Build smart playlists** that either keep a smooth mood or intentionally throw in a well-timed curveball  
- **Explain transitions** with an optional local LLM ([Ollama](https://ollama.com/)) so you know *why* each track follows the last

---

## Folder Structure



```
music_rec_improved/
├── README.md            # This file
├── requirements.txt      # Pip dependencies (for pure‑pip installations)
├── environment.yml       # Conda environment definition (optional)
├── config.yaml           # Centralised configuration (paths, weights, LLM settings)
├── .gitignore            # Ignore virtualenvs, caches and large data files
├── data/                 # **Untracked** - stores fingerprints, embeddings, features, indices
│   └── README.md         # Explains the purpose of each file in data/
├── scripts/              # Individual pipeline components
│   ├── fingerprint.py       # Build the fingerprint database from your music library
│   ├── embed_clap.py        # Compute CLAP embeddings for each track
│   ├── extract_features.py  # Extract tempo/key/energy (librosa) and tags (MusicNN)
│   ├── build_faiss_index.py # Build a FAISS index from the embeddings
│   ├── sequencer.py         # Playlist sequencing engine and cost functions
│   ├── explanation.py       # Connect to Ollama to generate transition explanations
│   └── generate_playlist.py # End‑to‑end demo: recognise, sequence, explain
└── tests/                # Optional unit tests
```

## Installation

### Using Conda (recommended)

If you have Anaconda or Miniconda installed, you can create an isolated environment with all required packages.  The file `environment.yml` declares a `musicrec` environment that uses the **conda‑forge**, **pytorch** and **nvidia** channels.  Unlike earlier versions of this project, the environment file no longer depends on **Essentia**, which is not available on Windows; instead we rely on `librosa` for tempo/key/energy extraction.  To create the environment:


---

## Install It

### With Conda (recommended)

```sh
conda env create -f environment.yml
conda activate musicrec


Once activated, verify that PyTorch can see your GPU (optional):

```sh
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### Using pip

If you prefer a pure‑pip installation, use the provided `requirements.txt` after creating a virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # optional GPU support
pip install -r requirements.txt
```

### Configure your music library

Before running any scripts, open `config.yaml` and set the `music_dir` field to the absolute path of your music collection.  This path is not committed to the repository-leaving it blank in version control ensures you don’t leak personal file locations.  Once set, all scripts will use it by default, so you don’t need to pass `--music-dir` on the command line.

## Usage

All scripts use `config.yaml` for paths and model settings.  Edit it to match your environment, for example:

```yaml
database_path: "data/fingerprints.db"
embedding_file: "data/embeddings.npy"
mapping_file: "data/embedding_mapping.txt"
feature_file: "data/features.json"
faiss_index_file: "data/faiss.index"

sequencing:
  contrast_every: 3         # every N tracks, jump to a different style
  mood_threshold: 0.3       # allowable mood difference on contrast slots
  tempo_threshold: 10       # allowable BPM difference on contrast slots
  key_threshold: 2          # allowable key distance (circle of fifths) on contrast slots
  weights:
    style:  0.3
    mood:   1.0
    tempo:  0.5
    key:    0.4
    energy: 0.5

llm:
  model: "qwen2.5vl:3b"
  max_tokens: 160
  temperature: 0.7
```

## Install It

### With Conda (recommended)

```sh
conda env create -f environment.yml
conda activate musicrec
```

Check GPU:

```sh
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

### With pip

```sh
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # optional GPU build
pip install -r requirements.txt
```

---

### You’ll Also Need

- **FFmpeg** (audio decoding for PyDejavu & librosa)  
  - Windows: download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and add `bin` to `PATH`  
  - Linux: `sudo apt install ffmpeg`
- **Ollama** (optional, for LLM explanations)  
  - Install from [ollama.com](https://ollama.com/download)  
  - Example: `ollama pull qwen2.5vl:3b`

---

## Quick Config

Edit `config.yaml`:

```yaml
music_dir: "D:/MyMusic"

database_path: "data/fingerprints.db"
embedding_file: "data/embeddings.npy"
mapping_file: "data/embedding_mapping.txt"
feature_file: "data/features.json"
faiss_index_file: "data/faiss.index"

sequencing:
  contrast_every: 3
  mood_threshold: 0.3
  tempo_threshold: 10
  key_threshold: 2

llm:
  model: "qwen2.5vl:3b"
  max_tokens: 160
  temperature: 0.7
```

---

## How to Use

### 1. Ingest your library

```sh
python main.py ingest
```

This fingerprints your music, extracts features, makes embeddings, and builds the FAISS index.  
If you add new tracks later, just re-run - only new ones get processed.  
Use `--force` to rebuild everything.

---

### 2. Make a playlist

From a snippet in the previous version
```sh
C:\path\to\your\envs\musicrec\python.exe run.py web
```

From a snippet in the previous version
```sh
python main.py playlist --input-audio /path/to/snippet.wav --length 20 --policy smooth
```

From a known track:

```sh
python main.py playlist --start-path "/path/to/song.flac" --length 20 --policy contrast
```

---

## Why It’s Modular

- Swap in a different fingerprinting engine  
- Try other embedding models  
- Change the playlist cost function entirely  
- Still works offline

---

(Changing the tone: It is just a project so that I could practice broader usage of concepts and libraries)

