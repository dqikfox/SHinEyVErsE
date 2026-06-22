# Model Reference Guide

This project uses ComfyUI workflows that expect specific model files in ComfyUI's standard `models/` directories.

## Image models

### Stable Diffusion 1.5

- Primary reference: https://huggingface.co/runwayml/stable-diffusion-v1-5
- Expected file:
  - `checkpoints/v1-5-pruned-emaonly-fp16.safetensors`

### SDXL Base 1.0

- Primary reference: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- Expected file:
  - `checkpoints/sd_xl_base_1.0.safetensors`

### FLUX.1 Schnell

- Primary reference: https://huggingface.co/black-forest-labs/FLUX.1-schnell
- Expected files:
  - `diffusion_models/flux1-schnell.safetensors`
  - `text_encoders/clip_l.safetensors`
  - `text_encoders/t5xxl_fp8_e4m3fn.safetensors`
  - `vae/ae.safetensors`

## Video models

### Wan2.2 5B Text-to-Video

- Primary reference: https://huggingface.co/Wan-AI
- Expected files:
  - `diffusion_models/wan2.2_ti2v_5B_fp16.safetensors`
  - `text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors`
  - `vae/wan2.2_vae.safetensors`

### Wan2.2 14B Image-to-Video

- Primary reference: https://huggingface.co/Wan-AI
- Expected files:
  - `diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors`
  - `diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors`
  - `loras/Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors`
  - `loras/Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors`
  - `vae/wan_2.1_vae.safetensors`

## Notes

- You only need the assets for the modes you plan to use.
- If the filenames in your ComfyUI installation differ, update the matching values in `app.py`.
- The repo's workflows assume ComfyUI's standard folder structure.
