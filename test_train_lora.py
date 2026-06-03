"""Minimal smoke test for LoRA training."""
import os

import torch

from train import LoRAConfig, train


MODEL_PATH = "./models/anything-v5"
DATA_DIR = "./data"
OUTPUT_DIR = "./outputs/lora_test"
FINAL_LORA_PATH = os.path.join(OUTPUT_DIR, "final_lora.safetensors")
RESOLUTION = 512
CENTER_CROP = True
BATCH_SIZE = 1
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4
LORA_RANK = 4
LORA_ALPHA = 4.0
GRADIENT_ACCUMULATION_STEPS = 1
LOGGING_STEPS = 1
SAVE_STEPS = 0
SAVE_TOTAL_LIMIT = 2
NUM_WORKERS = 0
MIXED_PRECISION = "no"
SEED = 42


def _resolve_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _validate_paths():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model directory not found: {MODEL_PATH}")
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Training data directory not found: {DATA_DIR}")


def build_smoke_test_config():
    device = _resolve_device()
    return LoRAConfig(
        model_path=MODEL_PATH,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        device=device,
        resolution=RESOLUTION,
        center_crop=CENTER_CROP,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        lora_rank=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        num_workers=NUM_WORKERS,
        mixed_precision=MIXED_PRECISION,
        seed=SEED,
    )


def main():
    _validate_paths()
    config = build_smoke_test_config()

    print(f"Using device: {config.device}")
    print(f"Using local model: {config.model_path}")
    print(f"Using training data: {config.data_dir}")
    print("Starting LoRA training smoke test...")

    train(config)

    if not os.path.exists(FINAL_LORA_PATH) or os.path.getsize(FINAL_LORA_PATH) == 0:
        raise RuntimeError(f"Training smoke test failed to create a valid LoRA file: {FINAL_LORA_PATH}")

    print(f"LoRA training smoke test passed. Output: {FINAL_LORA_PATH}")


if __name__ == "__main__":
    main()
