import os

from huggingface_hub import snapshot_download


MODEL_ID = "runwayml/stable-diffusion-v1-5"
SAVE_PATH = "./models/stable-diffusion-v1-5"

ALLOW_PATTERNS = [
    "tokenizer/*",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "scheduler/scheduler_config.json",
]

IGNORE_PATTERNS = [
    "*.bin",
    "*.ckpt",
    "*.msgpack",
    "*.onnx*",
]

REQUIRED_FILES = [
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer/tokenizer_config.json",
    "tokenizer/special_tokens_map.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "unet/config.json",
    "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "scheduler/scheduler_config.json",
]


def validate_local_model(save_path=SAVE_PATH):
    missing_files = [
        relative_path
        for relative_path in REQUIRED_FILES
        if not os.path.exists(os.path.join(save_path, relative_path))
    ]
    return missing_files


def main():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.makedirs(SAVE_PATH, exist_ok=True)

    missing_before = validate_local_model()
    if not missing_before:
        print(f"All required SD1.5 files already exist under: {SAVE_PATH}")
        return

    print(f"Downloading required SD1.5 files to: {SAVE_PATH}")
    print("Files to fetch:")
    for pattern in ALLOW_PATTERNS:
        print(f"  - {pattern}")

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=SAVE_PATH,
        local_dir_use_symlinks=False,
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
    )

    missing_after = validate_local_model()
    if missing_after:
        raise RuntimeError(
            "Model download finished but some required files are still missing:\n"
            + "\n".join(f"  - {path}" for path in missing_after)
        )

    print("Download complete.")
    print("Verified local files:")
    for relative_path in REQUIRED_FILES:
        print(f"  - {relative_path}")


if __name__ == "__main__":
    main()
