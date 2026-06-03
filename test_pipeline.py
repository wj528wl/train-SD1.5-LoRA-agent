"""Inference smoke test for the local SD1.5 pipeline."""
import os

import torch

from src.pipeline import StableDiffusionPipeline


MODEL_PATH = "./models/anything-v5"
LORA_PATH = "./outputs//lora_full/final_lora2.safetensors"
USE_LORA = True
OUTPUT_BASE = "output.png"
OUTPUT_LORA = "output_lora.png"
PROMPT = (
    "leisai, neck ring, purple hair, green eyes, white shirt, "
    "upper body, smile, anime girl, soft light, best quality"
)
NEGATIVE_PROMPT = (
    "realistic, low quality, worst quality, bad anatomy, deformed face, "
    "distorted eyes, extra fingers, extra limbs, poorly drawn hands, messy background"
)


def _resolve_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_model_path():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")
    return MODEL_PATH


def _resolve_lora_path():
    if not USE_LORA:
        return None
    return LORA_PATH if os.path.exists(LORA_PATH) else None


def main():
    device = _resolve_device()
    model_path = _resolve_model_path()
    lora_path = _resolve_lora_path()

    print(f"Using device: {device}")
    print(f"Using local model: {model_path}")
    if lora_path:
        print(f"Using local LoRA: {lora_path}")
    else:
        print("LoRA not found. Running base model only.")

    dtype = torch.float16 if device == "cuda" else torch.float32
    pipeline = StableDiffusionPipeline(
        device=device,
        dtype=dtype,
        model_path=model_path,
        lora_path=lora_path,
        lora_strength=1.0,
        verbose=True,
    )

    image = pipeline(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=42,
    )

    output_path = OUTPUT_LORA if lora_path else OUTPUT_BASE
    image.save(output_path)
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError(f"Inference smoke test failed to create a valid image: {output_path}")

    print(f"Inference smoke test passed. Image saved to {output_path}")


if __name__ == "__main__":
    main()
