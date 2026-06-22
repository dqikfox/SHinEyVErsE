import json, time, shutil, uuid
from pathlib import Path
import requests
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

COMFY_URL = "http://127.0.0.1:8188"
COMFY_OUTPUT = Path(r"C:\Users\Shadow\ComfyUI\output")
AI_IMAGES = Path(r"C:\Users\Shadow\AI_Output\Images")
AI_VIDEOS = Path(r"C:\Users\Shadow\AI_Output\Videos")
WAN_TEMPLATE = Path(r"C:\Users\Shadow\ComfyUI\user\default\workflows\video_wan2_2_5B_ti2v.json")

AI_IMAGES.mkdir(parents=True, exist_ok=True)
AI_VIDEOS.mkdir(parents=True, exist_ok=True)

def get_wan_negative_default():
    try:
        wf = json.loads(WAN_TEMPLATE.read_text(encoding="utf-8"))
        for n in wf["nodes"]:
            if n["id"] == 7:
                return n["widgets_values"][0]
    except Exception:
        pass
    return "low quality, blurry, distorted, watermark"

WAN_NEG_DEFAULT = get_wan_negative_default()

app = FastAPI(title="SHInEyVErSE")
app.mount("/files/Images", StaticFiles(directory=str(AI_IMAGES)), name="images")
app.mount("/files/Videos", StaticFiles(directory=str(AI_VIDEOS)), name="videos")

MODEL_INFO = {
    "sd15":         {"kind": "image",      "label": "Stable Diffusion 1.5 (fastest)",        "width": 512,  "height": 512,  "steps": 20, "cfg": 8.0, "sampler": "euler",  "scheduler": "normal"},
    "sdxl":         {"kind": "image",      "label": "SDXL (balanced)",                       "width": 1024, "height": 1024, "steps": 20, "cfg": 8.0, "sampler": "euler",  "scheduler": "normal"},
    "flux":         {"kind": "image",      "label": "Flux Schnell (best quality)",           "width": 1024, "height": 1024, "steps": 4,  "cfg": 1.0, "sampler": "euler",  "scheduler": "simple"},
    "wan22_5b":     {"kind": "video",      "label": "Wan2.2 5B Text-to-Video",               "width": 1280, "height": 704,  "steps": 20, "cfg": 5.0, "sampler": "uni_pc", "scheduler": "simple", "length": 121, "fps": 24},
    "wan22_14b_i2v":{"kind": "video_i2v",  "label": "Wan2.2 14B Image-to-Video (Lightning)", "width": 640,  "height": 640,  "steps": 4,  "cfg": 1.0, "sampler": "euler",  "scheduler": "simple", "length": 81,  "fps": 16},
}
def build_graph(model, prompt, negative, width, height, steps, cfg, seed, sampler, scheduler, length, fps, prefix, image_filename=None):
    if model in ("sd15", "sdxl"):
        ckpt = "v1-5-pruned-emaonly-fp16.safetensors" if model == "sd15" else "sd_xl_base_1.0.safetensors"
        graph = {
            "3": {"inputs": {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0, "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}, "class_type": "KSampler"},
            "4": {"inputs": {"ckpt_name": ckpt}, "class_type": "CheckpointLoaderSimple"},
            "5": {"inputs": {"width": width, "height": height, "batch_size": 1}, "class_type": "EmptyLatentImage"},
            "6": {"inputs": {"text": prompt, "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
            "7": {"inputs": {"text": negative or "text, watermark", "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
            "8": {"inputs": {"samples": ["3", 0], "vae": ["4", 2]}, "class_type": "VAEDecode"},
            "9": {"inputs": {"filename_prefix": prefix, "images": ["8", 0]}, "class_type": "SaveImage"},
        }
        return graph, "image"

    elif model == "flux":
        graph = {
            "38": {"inputs": {"unet_name": "flux1-schnell.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
            "40": {"inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5xxl_fp8_e4m3fn.safetensors", "type": "flux", "device": "default"}, "class_type": "DualCLIPLoader"},
            "39": {"inputs": {"vae_name": "ae.safetensors"}, "class_type": "VAELoader"},
            "27": {"inputs": {"width": width, "height": height, "batch_size": 1}, "class_type": "EmptySD3LatentImage"},
            "41": {"inputs": {"clip": ["40", 0], "clip_l": prompt, "t5xxl": prompt, "guidance": 3.5}, "class_type": "CLIPTextEncodeFlux"},
            "42": {"inputs": {"conditioning": ["41", 0]}, "class_type": "ConditioningZeroOut"},
            "31": {"inputs": {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0, "model": ["38", 0], "positive": ["41", 0], "negative": ["42", 0], "latent_image": ["27", 0]}, "class_type": "KSampler"},
            "8": {"inputs": {"samples": ["31", 0], "vae": ["39", 0]}, "class_type": "VAEDecode"},
            "9": {"inputs": {"filename_prefix": prefix, "images": ["8", 0]}, "class_type": "SaveImage"},
        }
        return graph, "image"

    elif model == "wan22_5b":
        graph = {
            "37": {"inputs": {"unet_name": "wan2.2_ti2v_5B_fp16.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
            "38": {"inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}, "class_type": "CLIPLoader"},
            "39": {"inputs": {"vae_name": "wan2.2_vae.safetensors"}, "class_type": "VAELoader"},
            "48": {"inputs": {"model": ["37", 0], "shift": 8}, "class_type": "ModelSamplingSD3"},
            "6": {"inputs": {"clip": ["38", 0], "text": prompt}, "class_type": "CLIPTextEncode"},
            "7": {"inputs": {"clip": ["38", 0], "text": negative or WAN_NEG_DEFAULT}, "class_type": "CLIPTextEncode"},
            "55": {"inputs": {"vae": ["39", 0], "width": width, "height": height, "length": length, "batch_size": 1}, "class_type": "Wan22ImageToVideoLatent"},
            "3": {"inputs": {"model": ["48", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["55", 0], "seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0}, "class_type": "KSampler"},
            "8": {"inputs": {"samples": ["3", 0], "vae": ["39", 0]}, "class_type": "VAEDecode"},
            "57": {"inputs": {"images": ["8", 0], "fps": fps}, "class_type": "CreateVideo"},
            "58": {"inputs": {"video": ["57", 0], "filename_prefix": f"video/{prefix}", "format": "auto", "codec": "auto"}, "class_type": "SaveVideo"},
        }
        return graph, "video"

    elif model == "wan22_14b_i2v":
        if not image_filename:
            raise ValueError("This model requires a starting image upload.")
        graph = {
            "1":  {"inputs": {"image": image_filename}, "class_type": "LoadImage"},
            "84": {"inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}, "class_type": "CLIPLoader"},
            "90": {"inputs": {"vae_name": "wan_2.1_vae.safetensors"}, "class_type": "VAELoader"},
            "95": {"inputs": {"unet_name": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
            "96": {"inputs": {"unet_name": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
            "101": {"inputs": {"model": ["95", 0], "lora_name": "Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors", "strength_model": 1.0}, "class_type": "LoraLoaderModelOnly"},
            "102": {"inputs": {"model": ["96", 0], "lora_name": "Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors", "strength_model": 1.0}, "class_type": "LoraLoaderModelOnly"},
            "104": {"inputs": {"model": ["101", 0], "shift": 5.0}, "class_type": "ModelSamplingSD3"},
            "103": {"inputs": {"model": ["102", 0], "shift": 5.0}, "class_type": "ModelSamplingSD3"},
            "93": {"inputs": {"clip": ["84", 0], "text": prompt}, "class_type": "CLIPTextEncode"},
            "89": {"inputs": {"clip": ["84", 0], "text": negative or WAN_NEG_DEFAULT}, "class_type": "CLIPTextEncode"},
            "98": {"inputs": {"positive": ["93", 0], "negative": ["89", 0], "vae": ["90", 0], "start_image": ["1", 0], "width": width, "height": height, "length": length, "batch_size": 1}, "class_type": "WanImageToVideo"},
            "86": {"inputs": {"model": ["104", 0], "positive": ["98", 0], "negative": ["98", 1], "latent_image": ["98", 2], "add_noise": "enable", "noise_seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler, "start_at_step": 0, "end_at_step": steps // 2, "return_with_leftover_noise": "enable"}, "class_type": "KSamplerAdvanced"},
            "85": {"inputs": {"model": ["103", 0], "positive": ["98", 0], "negative": ["98", 1], "latent_image": ["86", 0], "add_noise": "disable", "noise_seed": 0, "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": scheduler, "start_at_step": steps // 2, "end_at_step": steps, "return_with_leftover_noise": "disable"}, "class_type": "KSamplerAdvanced"},
            "87": {"inputs": {"samples": ["85", 0], "vae": ["90", 0]}, "class_type": "VAEDecode"},
            "94": {"inputs": {"images": ["87", 0], "fps": fps}, "class_type": "CreateVideo"},
            "108": {"inputs": {"video": ["94", 0], "filename_prefix": f"video/{prefix}", "format": "auto", "codec": "auto"}, "class_type": "SaveVideo"},
        }
        return graph, "video"

    raise ValueError(f"unknown model {model}")

def find_and_store_output(history_entry, kind, prefix):
    outputs = history_entry.get("outputs", {})
    found = None
    for node_id, out in outputs.items():
        for key in ("images", "gifs", "video"):
            if key in out and out[key]:
                found = out[key][0]
                break
        if found:
            break
    if not found:
        return None
    filename = found.get("filename")
    subfolder = found.get("subfolder", "")
    dest_dir = AI_VIDEOS if kind == "video" else AI_IMAGES
    dest_path = dest_dir / filename
    if dest_path.exists():
        return f"/files/{'Videos' if kind=='video' else 'Images'}/{filename}"
    src = COMFY_OUTPUT / subfolder / filename
    for _ in range(20):
        if src.exists():
            shutil.copy2(src, dest_path)
            return f"/files/{'Videos' if kind=='video' else 'Images'}/{filename}"
        time.sleep(1)
    return None

@app.post("/upload_image")
def upload_image(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix or ".png"
    name = f"shinyverse_upload_{uuid.uuid4().hex[:10]}{suffix}"
    files = {"image": (name, file.file, file.content_type)}
    r = requests.post(f"{COMFY_URL}/upload/image", files=files, timeout=30)
    r.raise_for_status()
    data = r.json()
    return {"filename": data.get("name", name)}

@app.post("/generate")
def generate(
    model: str = Form(...),
    prompt: str = Form(...),
    negative: str = Form(""),
    width: int = Form(...),
    height: int = Form(...),
    steps: int = Form(...),
    cfg: float = Form(...),
    seed: int = Form(...),
    sampler: str = Form(...),
    scheduler: str = Form(...),
    length: int = Form(121),
    fps: int = Form(24),
    image_filename: str = Form(""),
):
    prefix = f"SHInEyVErSE_{int(time.time())}"
    try:
        graph, kind = build_graph(model, prompt, negative, width, height, steps, cfg, seed, sampler, scheduler, length, fps, prefix, image_filename or None)
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)})

    r = requests.post(f"{COMFY_URL}/prompt", json={"prompt": graph}, timeout=15)
    if r.status_code != 200:
        return JSONResponse({"ok": False, "error": f"ComfyUI rejected the job: {r.text[:300]}"})
    prompt_id = r.json()["prompt_id"]

    timeout_s = 1800
    start = time.time()
    while time.time() - start < timeout_s:
        h = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=10).json()
        if prompt_id in h:
            entry = h[prompt_id]
            status = entry.get("status", {})
            if status.get("completed"):
                if status.get("status_str") == "success":
                    url = find_and_store_output(entry, kind, prefix)
                    if url:
                        return JSONResponse({"ok": True, "kind": kind, "url": url})
                    return JSONResponse({"ok": False, "error": "Generation succeeded but output file could not be located."})
                else:
                    msgs = status.get("messages", [])
                    err = next((m[1].get("exception_message") for m in msgs if m[0] == "execution_error"), "Unknown error")
                    return JSONResponse({"ok": False, "error": err})
        time.sleep(2)
    return JSONResponse({"ok": False, "error": "Timed out waiting for generation."})

@app.get("/models")
def models():
    return MODEL_INFO

@app.get("/gallery")
def gallery(limit: int = 24):
    items = []
    for folder, kind in ((AI_IMAGES, "image"), (AI_VIDEOS, "video")):
        for f in folder.glob("*"):
            if f.is_file():
                items.append({"url": f"/files/{'Videos' if kind=='video' else 'Images'}/{f.name}", "kind": kind, "name": f.name, "mtime": f.stat().st_mtime})
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return items[:limit]

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE

HTML_PAGE = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SHInEyVErSE</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@500;700;800&family=Quicksand:wght@500;600;700&display=swap');
  :root { color-scheme: light; }
  body {
    background: #fff0f6;
    background-image:
      radial-gradient(circle, #ffd6ea 2px, transparent 2.5px),
      radial-gradient(circle, #c9ecff 2px, transparent 2.5px);
    background-size: 38px 38px, 38px 38px;
    background-position: 0 0, 19px 19px;
    color: #5a3d52;
    font-family: 'Quicksand', 'Segoe UI', Arial, sans-serif;
    margin: 0; padding: 30px;
  }
  .wrap { max-width: 920px; margin: 0 auto; }
  h1 {
    font-family: 'Baloo 2', cursive;
    font-size: 38px; margin-bottom: 2px; text-align:center;
    color: #ff6fa5;
    text-shadow: 2px 2px 0 #fff, 4px 4px 0 #ffd1e6;
    letter-spacing: 1px;
  }
  h1::before { content: "\01F380 "; }
  h1::after { content: " \01F380"; }
  .sub { color:#a86b95; margin-bottom: 24px; font-size: 14px; text-align:center; font-weight:600; }
  .sub::before { content: "\01F43E "; }
  .sub::after { content: " \01F43E"; }
  .card {
    background:#fffafd;
    border: 3px dashed #ffb6d9;
    border-radius:22px;
    padding:24px;
    margin-bottom:22px;
    box-shadow: 6px 6px 0 #ffd1e6, 0 0 0 6px #fff inset;
  }
  label { display:block; font-size:13px; color:#b3568f; font-weight:700; margin-bottom:5px; margin-top:14px; }
  textarea, input, select {
    width:100%; box-sizing:border-box;
    background:#fff7fb; color:#5a3d52;
    border:2px solid #ffc9e2; border-radius:14px;
    padding:10px 12px; font-size:14px; font-family:inherit;
  }
  textarea:focus, input:focus, select:focus { outline:none; border-color:#ff8fc0; box-shadow:0 0 0 3px #ffe1ef; }
  textarea { resize:vertical; min-height:64px; }
  .row { display:flex; gap:14px; }
  .row > div { flex:1; }
  .check { display:flex; align-items:center; gap:8px; margin-top:14px; }
  .check input { width:auto; }
  button {
    margin-top:20px;
    background: linear-gradient(180deg,#ffa8d2,#ff7fb8);
    color:#fff; border:3px solid #fff; border-radius:18px;
    padding:13px 26px; font-size:16px; font-weight:800; font-family:'Baloo 2',cursive;
    cursor:pointer; box-shadow: 4px 4px 0 #d9477f;
    transition: transform 0.1s;
  }
  button:hover { transform: translateY(-2px); }
  button:active { transform: translateY(2px); box-shadow: 2px 2px 0 #d9477f; }
  button:disabled { background:#e8c4d6; box-shadow:none; cursor:not-allowed; }
  .status { margin-top:16px; font-size:13px; color:#e8a23c; font-weight:700; }
  .err { color:#ff5577; }
  .result { margin-top:20px; text-align:center; }
  .result img, .result video {
    max-width:100%; border-radius:18px; border:5px solid #fff;
    box-shadow: 0 0 0 4px #ffc9e2, 6px 6px 14px rgba(255,143,192,0.35);
  }
  .imgOnly, .videoOnly, .i2vOnly { display:none; }
  .hint { font-size:11px; color:#c98ab0; margin-top:3px; }
  .gallery { display:grid; grid-template-columns: repeat(auto-fill,minmax(140px,1fr)); gap:12px; }
  .gallery .item {
    position:relative; border-radius:14px; overflow:hidden;
    border:3px solid #ffd1e6; background:#fff; box-shadow:3px 3px 0 #ffd1e6;
  }
  .gallery img, .gallery video { width:100%; height:120px; object-fit:cover; display:block; }
  .gallery .name { font-size:9px; color:#b3568f; padding:4px; word-break:break-all; font-weight:600; }
  h2 { font-family:'Baloo 2',cursive; font-size:20px; color:#ff6fa5; }
  h2::before { content: "\02728 "; }
  .preview { max-width:200px; border-radius:14px; margin-top:8px; display:none; border:3px solid #ffc9e2; }
</style>
</head>
<body>
<div class="wrap">
  <h1>SHInEyVErSE</h1>
  <div class="sub">Local image &amp; video generation - powered by your ComfyUI install</div>

  <div class="card">
    <label>Prompt</label>
    <textarea id="prompt" placeholder="Describe what you want to create..."></textarea>

    <label>Negative prompt (optional - leave blank for sensible default)</label>
    <textarea id="negative" placeholder="Things to avoid in the result"></textarea>

    <div class="row">
      <div>
        <label>Mode</label>
        <select id="mode">
          <option value="image">Image</option>
          <option value="video">Video (text-to-video)</option>
          <option value="video_i2v">Video (image-to-video)</option>
        </select>
      </div>
      <div>
        <label>Model</label>
        <select id="model"></select>
      </div>
    </div>

    <div class="i2vOnly" id="imageUploadField">
      <label>Starting image</label>
      <input type="file" id="imageFile" accept="image/*">
      <img id="imagePreview" class="preview">
    </div>

    <div class="row">
      <div>
        <label>Width</label>
        <input type="number" id="width" step="8">
      </div>
      <div>
        <label>Height</label>
        <input type="number" id="height" step="8">
      </div>
    </div>

    <div class="row videoOnly" id="videoFields">
      <div>
        <label>Length (frames)</label>
        <input type="number" id="length">
        <div class="hint">at 24fps, 121 frames ~= 5 seconds</div>
      </div>
      <div>
        <label>FPS</label>
        <input type="number" id="fps">
      </div>
    </div>

    <div class="row">
      <div>
        <label>Steps</label>
        <input type="number" id="steps">
      </div>
      <div>
        <label>CFG</label>
        <input type="number" id="cfg" step="0.1">
      </div>
    </div>

    <div class="row">
      <div>
        <label>Sampler</label>
        <select id="sampler">
          <option>euler</option>
          <option>euler_ancestral</option>
          <option>dpmpp_2m</option>
          <option>uni_pc</option>
        </select>
      </div>
      <div>
        <label>Scheduler</label>
        <select id="scheduler">
          <option>normal</option>
          <option>simple</option>
          <option>karras</option>
        </select>
      </div>
    </div>

    <div class="row">
      <div>
        <label>Seed</label>
        <input type="number" id="seed" value="0">
      </div>
      <div class="check" style="margin-top:34px;">
        <input type="checkbox" id="randomize" checked>
        <label style="margin:0;">Randomize seed each run</label>
      </div>
    </div>

    <button id="goBtn" onclick="generate()">Generate</button>
    <div class="status" id="status"></div>
  </div>

  <div class="result" id="result"></div>

  <div class="card">
    <h2>Gallery</h2>
    <div class="gallery" id="gallery"></div>
  </div>
</div>
<script>
let MODELS = {};

async function loadModels() {
  const r = await fetch('/models');
  MODELS = await r.json();
  refreshModelOptions();
}

function refreshModelOptions() {
  const mode = document.getElementById('mode').value;
  const sel = document.getElementById('model');
  sel.innerHTML = '';
  for (const [key, info] of Object.entries(MODELS)) {
    if (info.kind === mode) {
      const opt = document.createElement('option');
      opt.value = key;
      opt.textContent = info.label;
      sel.appendChild(opt);
    }
  }
  applyDefaults();
}

function applyDefaults() {
  const key = document.getElementById('model').value;
  const info = MODELS[key];
  if (!info) return;
  document.getElementById('width').value = info.width;
  document.getElementById('height').value = info.height;
  document.getElementById('steps').value = info.steps;
  document.getElementById('cfg').value = info.cfg;
  document.getElementById('sampler').value = info.sampler;
  document.getElementById('scheduler').value = info.scheduler;
  const isVideo = (info.kind === 'video' || info.kind === 'video_i2v');
  document.getElementById('videoFields').style.display = isVideo ? 'flex' : 'none';
  document.getElementById('imageUploadField').style.display = info.kind === 'video_i2v' ? 'block' : 'none';
  if (isVideo) {
    document.getElementById('length').value = info.length;
    document.getElementById('fps').value = info.fps;
  }
}

document.getElementById('mode').addEventListener('change', refreshModelOptions);
document.getElementById('model').addEventListener('change', applyDefaults);

document.getElementById('imageFile').addEventListener('change', function(e) {
  const file = e.target.files[0];
  const preview = document.getElementById('imagePreview');
  if (file) {
    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';
  } else {
    preview.style.display = 'none';
  }
});

async function loadGallery() {
  const r = await fetch('/gallery');
  const items = await r.json();
  const g = document.getElementById('gallery');
  g.innerHTML = '';
  for (const it of items) {
    const div = document.createElement('div');
    div.className = 'item';
    if (it.kind === 'video') {
      div.innerHTML = '<video src="' + it.url + '" muted loop onmouseover="this.play()" onmouseout="this.pause()"></video><div class="name">' + it.name + '</div>';
    } else {
      div.innerHTML = '<img src="' + it.url + '"><div class="name">' + it.name + '</div>';
    }
    g.appendChild(div);
  }
}

async function generate() {
  const btn = document.getElementById('goBtn');
  const statusEl = document.getElementById('status');
  const resultEl = document.getElementById('result');
  resultEl.innerHTML = '';
  statusEl.className = 'status';
  let seed = document.getElementById('seed').value;
  if (document.getElementById('randomize').checked) {
    seed = Math.floor(Math.random() * 1000000000000);
    document.getElementById('seed').value = seed;
  }
  const modelKey = document.getElementById('model').value;
  const info = MODELS[modelKey];

  let imageFilename = '';
  if (info.kind === 'video_i2v') {
    const fileInput = document.getElementById('imageFile');
    if (!fileInput.files[0]) {
      statusEl.className = 'status err';
      statusEl.textContent = 'Please choose a starting image first.';
      return;
    }
    statusEl.textContent = 'Uploading image...';
    const ufd = new FormData();
    ufd.append('file', fileInput.files[0]);
    const ur = await fetch('/upload_image', { method: 'POST', body: ufd });
    const udata = await ur.json();
    imageFilename = udata.filename;
  }

  const fd = new FormData();
  fd.append('model', modelKey);
  fd.append('prompt', document.getElementById('prompt').value);
  fd.append('negative', document.getElementById('negative').value);
  fd.append('width', document.getElementById('width').value);
  fd.append('height', document.getElementById('height').value);
  fd.append('steps', document.getElementById('steps').value);
  fd.append('cfg', document.getElementById('cfg').value);
  fd.append('seed', seed);
  fd.append('sampler', document.getElementById('sampler').value);
  fd.append('scheduler', document.getElementById('scheduler').value);
  fd.append('length', document.getElementById('length').value || 121);
  fd.append('fps', document.getElementById('fps').value || 24);
  fd.append('image_filename', imageFilename);

  btn.disabled = true;
  const isVideo = (info.kind === 'video' || info.kind === 'video_i2v');
  statusEl.textContent = isVideo ? 'Generating video... this can take several minutes.' : 'Generating image...';
  const startTime = Date.now();
  const timer = setInterval(() => {
    const s = Math.floor((Date.now() - startTime) / 1000);
    statusEl.textContent = (isVideo ? 'Generating video... ' : 'Generating image... ') + s + 's elapsed';
  }, 1000);

  try {
    const r = await fetch('/generate', { method: 'POST', body: fd });
    const data = await r.json();
    clearInterval(timer);
    btn.disabled = false;
    if (data.ok) {
      statusEl.textContent = 'Done.';
      if (data.kind === 'video') {
        resultEl.innerHTML = '<video src="' + data.url + '" controls autoplay loop></video>';
      } else {
        resultEl.innerHTML = '<img src="' + data.url + '">';
      }
      loadGallery();
    } else {
      statusEl.className = 'status err';
      statusEl.textContent = 'Error: ' + data.error;
    }
  } catch (e) {
    clearInterval(timer);
    btn.disabled = false;
    statusEl.className = 'status err';
    statusEl.textContent = 'Request failed: ' + e;
  }
}

loadModels();
loadGallery();
</script>
</body>
</html>
"""