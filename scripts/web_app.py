# web/app_web.py
import io, json, os, pathlib
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Reuse your existing pipeline bits
from scripts import generate_playlist as gp
from scripts import embed_clap as ec
from scripts import build_faiss_index as bi
from main import load_config

app = FastAPI(title="PulseFlow AI – Web")

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = load_config(ROOT / "config.yaml")
MUSIC_DIR = pathlib.Path(CFG.get("music_dir", "")) if CFG.get("music_dir") else None
DATA_DIR = ROOT / "data"

app.mount("/static", StaticFiles(directory=str(ROOT / "web" / "static")), name="static")

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>PulseFlow AI — Web</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, Arial; margin: 24px; }
    .row { display:flex; gap:24px; align-items:flex-start; }
    .panel { border:1px solid #ddd; border-radius:12px; padding:16px; flex:1; }
    .drop { border:2px dashed #999; padding:24px; text-align:center; border-radius:12px; color:#666; }
    button { padding:10px 14px; border-radius:10px; border:1px solid #555; cursor:pointer; }
    input[type="number"]{ width:80px; }
    .small { color:#777; font-size:12px; }
    .item { padding:6px 8px; border-bottom:1px solid #eee; cursor:pointer;}
    .item:hover { background:#f6f6f6; }
    .tracks { max-height:360px; overflow:auto; border:1px solid #eee; border-radius:8px; }
    pre { white-space: pre-wrap; background:#111; color:#eee; padding:12px; border-radius:10px; }
  </style>
</head>
<body>
  <h1>PulseFlow AI — Offline Music Recommender & Recogniser</h1>

  <div class="row">
    <div class="panel">
      <h3>1) Start from a clip (drag & drop)</h3>
      <div id="drop" class="drop">Drop WAV/MP3/FLAC here or <input type="file" id="fileInput" /></div>
      <div style="margin-top:8px;">
        Length: <input id="lenA" type="number" value="20" min="3" max="100" />
        Policy:
        <select id="polA">
          <option value="smooth">smooth</option>
          <option value="contrast">contrast</option>
        </select>
        <label style="margin-left:12px;">
          <input id="llmA" type="checkbox" checked /> Explain transitions
        </label>
        <button onclick="fromClip()">Generate</button>
      </div>
      <div class="small">No files are uploaded anywhere; everything stays local.</div>
    </div>

    <div class="panel">
      <h3>2) Start from your library</h3>
      <div class="small">Root: <code id="root"></code></div>
      <div id="tracks" class="tracks"></div>
      <div style="margin-top:8px;">
        Length: <input id="lenB" type="number" value="20" min="3" max="100" />
        Policy:
        <select id="polB">
          <option value="smooth">smooth</option>
          <option value="contrast">contrast</option>
        </select>
        <label style="margin-left:12px;">
          <input id="llmB" type="checkbox" checked /> Explain transitions
        </label>
        <button onclick="fromPath()">Generate</button>
      </div>
    </div>
  </div>

  <div class="panel" style="margin-top:24px;">
    <h3>Result</h3>
    <div id="result"></div>
  </div>

<script>
const rootEl = document.getElementById('root');
const listEl = document.getElementById('tracks');
let selectedPath = null;
fetch('/library').then(r=>r.json()).then(d=>{
  rootEl.textContent = d.root || '(not configured)';
  listEl.innerHTML = d.tracks.map(t=>(
    `<div class="item" data-path="${t.path.replace(/"/g,'&quot;')}">${t.name}</div>`
  )).join('');
  listEl.onclick = (e)=>{
    const item = e.target.closest('.item');
    if(!item) return;
    [...listEl.querySelectorAll('.item')].forEach(x=>x.style.background='');
    item.style.background = '#e9f3ff';
    selectedPath = item.dataset.path;
  };
});

function renderPlaylist(payload){
  const div = document.getElementById('result');
  const lines = [];
  if(payload.summary){ lines.push(`<p><strong>Top pick:</strong> ${payload.summary.top_pick || ''}</p>`); }
  if(payload.summary && payload.summary.overview){ lines.push(`<p>${payload.summary.overview}</p>`); }
  lines.push('<ol>');
  for(const row of (payload.tracks||[])){
    const expl = row.explanation ? `<div class="small">${row.explanation}</div>` : '';
    const stats = row.stats ? `<div class="small">${row.stats}</div>` : '';
    lines.push(`<li>${row.title || row.path}${expl}${stats}</li>`);
  }
  lines.push('</ol>');
  if(payload.m3u_path){ lines.push(`<p><a href="/static/${payload.m3u_path}" download>Download M3U</a></p>`); }
  div.innerHTML = lines.join('\\n');
}

async function fromClip(){
  const f = document.getElementById('fileInput').files[0] || window._dropFile;
  if(!f){ alert('Please choose a clip file.'); return; }
  const fd = new FormData();
  fd.append('file', f);
  fd.append('length', document.getElementById('lenA').value);
  fd.append('policy', document.getElementById('polA').value);
  fd.append('explain', document.getElementById('llmA').checked ? '1':'0');
  const r = await fetch('/playlist/clip', {method:'POST', body:fd});
  renderPlaylist(await r.json());
}

async function fromPath(){
  if(!selectedPath){ alert('Select a track from the list.'); return; }
  const fd = new FormData();
  fd.append('path', selectedPath);
  fd.append('length', document.getElementById('lenB').value);
  fd.append('policy', document.getElementById('polB').value);
  fd.append('explain', document.getElementById('llmB').checked ? '1':'0');
  const r = await fetch('/playlist/path', {method:'POST', body:fd});
  renderPlaylist(await r.json());
}

const drop = document.getElementById('drop');
drop.ondragover = (e)=>{ e.preventDefault(); drop.style.background='#f7f7f7';};
drop.ondragleave = (e)=>{ drop.style.background='';};
drop.ondrop = (e)=>{
  e.preventDefault();
  drop.style.background='';
  window._dropFile = e.dataTransfer.files[0];
  drop.textContent = 'Ready: ' + window._dropFile.name;
};
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return INDEX_HTML

@app.get("/library")
def library():
    if not MUSIC_DIR or not MUSIC_DIR.exists():
        return JSONResponse({"root": None, "tracks": []})
    # Simple flat listing of audio files (no recursion for speed)
    exts = {".mp3",".flac",".wav",".m4a",".ogg",".aac"}
    files = []
    for p in sorted(MUSIC_DIR.glob("*")):
        if p.suffix.lower() in exts and p.is_file():
            files.append({"name": p.name, "path": str(p)})
    return JSONResponse({"root": str(MUSIC_DIR), "tracks": files})

@app.post("/playlist/clip")
async def playlist_clip(file: UploadFile = File(...),
                        length: int = Form(20),
                        policy: str = Form("smooth"),
                        explain: str = Form("1")):
    audio_bytes = await file.read()
    res = gp.generate_from_bytes(audio_bytes, length=length, policy=policy, explain=(explain=="1"))
    # optional: write m3u in /web/static for download
    static_dir = ROOT / "web" / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    m3u = static_dir / "playlist.m3u8"
    _write_m3u(m3u, [t.get("path","") for t in res.get("tracks",[])])
    res["m3u_path"] = f"playlist.m3u8"
    return JSONResponse(res)

@app.post("/playlist/path")
async def playlist_path(path: str = Form(...),
                        length: int = Form(20),
                        policy: str = Form("smooth"),
                        explain: str = Form("1")):
    res = gp.generate_from_path(path, length=length, policy=policy, explain=(explain=="1"))
    static_dir = ROOT / "web" / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    m3u = static_dir / "playlist.m3u8"
    _write_m3u(m3u, [t.get("path","") for t in res.get("tracks",[])])
    res["m3u_path"] = f"playlist.m3u8"
    return JSONResponse(res)

def _write_m3u(pathlib_path, paths: List[str]):
    with open(pathlib_path, "w", encoding="utf-8") as f:
        f.write("#EXTM3U\n")
        for p in paths:
            if p:
                f.write(p + "\n")
