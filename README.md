# SHInEyVErSE

A local, single-file web GUI for generating images and videos through your own [ComfyUI](https://github.com/comfyanonymous/ComfyUI) install. No cloud, no subscriptions, no OneDrive - everything stays on your machine.

Built with FastAPI. One Python file, no build step, no frontend framework.

## Features

- Prompt + negative prompt
- Switch between **Image** and **Video** modes
- Five models, each wired to the real, verified ComfyUI node graph it needs:
  - **Stable Diffusion 1.5** - fastest
  - **SDXL** - balanced quality/speed
  - **Flux Schnell** - best image quality (split UNETLoader + DualCLIPLoader pipeline)
  - **Wan2.2 5B Text-to-Video** - pure text-to-video, no starting image needed
  - **Wan2.2 14B Image-to-Video (Lightning)** - upload a starting image, dual high/low-noise sampler with the 4-step Lightning LoRA for fast generation
- Adjustable width/height, steps, CFG, sampler, scheduler, seed (with one-click randomize)
- Video-specific controls: frame length, FPS
- Built-in gallery of everything you've generated
- All outputs saved straight to local folders (`AI_Output/Images`, `AI_Output/Videos`) - deliberately outside any OneDrive-synced directory

## How it works

The FastAPI backend builds the exact ComfyUI API graph for whichever model you pick, submits it to your running ComfyUI instance's `/prompt` endpoint, polls `/history` until it's done, then copies the result into the local output folders and serves it back to the page.

## Prerequisites

- A working [ComfyUI](https://github.com/comfyanonymous/ComfyUI) install, running on `http://127.0.0.1:8188`
- Python 3.10+
- The following model files, in ComfyUI's normal `models/` folders:

| Model | Files needed |
|---|---|
| SD 1.5 | `checkpoints/v1-5-pruned-emaonly-fp16.safetensors` |
| SDXL | `checkpoints/sd_xl_base_1.0.safetensors` |
| Flux Schnell | `diffusion_models/flux1-schnell.safetensors`, `text_encoders/clip_l.safetensors`, `text_encoders/t5xxl_fp8_e4m3fn.safetensors`, `vae/ae.safetensors` |
| Wan2.2 5B T2V | `diffusion_models/wan2.2_ti2v_5B_fp16.safetensors`, `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`, `vae/wan2.2_vae.safetensors` |
| Wan2.2 14B I2V | `diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors`, `diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors`, `loras/Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors`, `loras/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors`, `vae/wan_2.1_vae.safetensors`, `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors` (shared with the 5B model above)

You only need the model(s) for the modes you actually want to use - SD1.5 alone is enough to try the app out.

## Setup

```bash
git clone <this-repo-url>
cd SHInEyVErSE
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Edit `COMFY_URL` near the top of `app.py` if your ComfyUI isn't running on the default `127.0.0.1:8188`, and update the hardcoded output/template paths to match your own machine.

## Running it

Make sure ComfyUI is already running, then:

```bash
venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 7860
```

Open `http://127.0.0.1:7860`.

On Windows, `Launch_SHInEyVErSE.bat` does this for you and auto-opens the browser.

## Notes

- This is a personal/local tool, not hardened for any kind of public or multi-user deployment - there's no auth on the endpoints.
- Paths in `app.py` are currently hardcoded to one machine's folder layout; adjust `COMFY_OUTPUT`, `AI_IMAGES`, `AI_VIDEOS`, and `WAN_TEMPLATE` for your own setup.
- The Wan2.2 14B I2V graph is a flattened, faithful translation of ComfyUI's own official subgraph template, fixed to the fast 4-step Lightning-LoRA path (the official template also supports a slower non-LoRA mode via a toggle, which isn't exposed here).
