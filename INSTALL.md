# Installation Guide

## Requirements

- Python 3.10+
- A working [ComfyUI](https://github.com/comfyanonymous/ComfyUI) installation
- The model files required for the presets you want to use

## 1. Install ComfyUI

Install ComfyUI and confirm it runs locally at:

- `http://127.0.0.1:8188`

If ComfyUI is on a different host or port, update `COMFY_URL` in `app.py`.

## 2. Clone this repository

```bash
git clone https://github.com/dqikfox/SHinEyVErsE.git
cd SHinEyVErsE
```

## 3. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate          # Windows
```

## 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

## 5. Configure local paths

Open `app.py` and update these values so they match your system:

- `COMFY_URL`
- `COMFY_OUTPUT`
- `AI_IMAGES`
- `AI_VIDEOS`
- `WAN_TEMPLATE`

## 6. Start the app

```bash
venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 7860
```

Then open:

- `http://127.0.0.1:7860`

## Optional Windows launcher

If you prefer, use `Launch_SHInEyVErSE.bat` to start the app and open the browser automatically.
