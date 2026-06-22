# SHInEyVErSE

A local, single-file web GUI for generating images and videos through your own [ComfyUI](https://github.com/comfyanonymous/ComfyUI) install. No cloud, no subscriptions, no OneDrive - everything stays local.

Built with FastAPI. One Python file, no build step, no frontend framework.

## Repository

- Source: https://github.com/dqikfox/SHinEyVErsE
- License: MIT

## Quick start

1. Install [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and make sure it is running at `http://127.0.0.1:8188`.
2. Install Python 3.10 or newer.
3. Download the model files you want to use and place them in ComfyUI's `models/` folders.
4. Update the local paths in `app.py` to match your machine.
5. Install dependencies and launch the app.

## Features

- Prompt and negative prompt inputs
- Image mode and video mode, including image-to-video
- Five prewired model presets that map to the correct ComfyUI node graphs
- Adjustable width, height, steps, CFG, sampler, scheduler, seed, frame length, and FPS
- One-click seed randomization
- Built-in gallery of generated outputs
- Local output folders for images and videos

## Model and asset references

Use these official or primary sources when downloading models and related assets:

- ComfyUI: https://github.com/comfyanonymous/ComfyUI
- Stable Diffusion 1.5: https://huggingface.co/runwayml/stable-diffusion-v1-5
- SDXL Base 1.0: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- FLUX.1 Schnell: https://huggingface.co/black-forest-labs/FLUX.1-schnell
- Wan-AI models and related assets: https://huggingface.co/Wan-AI

You only need the assets for the modes you plan to use. If a filename differs from the one listed in the app, update the path in `app.py`.

## Setup

```bash
git clone https://github.com/dqikfox/SHinEyVErsE.git
cd SHinEyVErsE
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Configuration

Edit the constants near the top of `app.py` to match your local setup:

- `COMFY_URL`
- `COMFY_OUTPUT`
- `AI_IMAGES`
- `AI_VIDEOS`
- `WAN_TEMPLATE`

The app assumes your ComfyUI output folder and local export folders are on your machine and not inside OneDrive-synced paths.

## Running it

Make sure ComfyUI is already running, then start the app with:

```bash
venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 7860
```

Open `http://127.0.0.1:7860` in your browser.

On Windows, `Launch_SHInEyVErSE.bat` starts the app and opens the browser for you.

## Notes

- This project is intended for local, single-user use.
- There is no authentication on the API endpoints.
- The Wan2.2 14B image-to-video graph follows the fast Lightning-LoRA path.
- If you change model filenames or ComfyUI workflow files, update the corresponding values in `app.py`.
