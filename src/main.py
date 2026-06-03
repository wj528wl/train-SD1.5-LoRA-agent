import argparse
import os
import sys

import torch

if __package__ in {None, ""}:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from src.pipeline import StableDiffusionPipeline
else:
    from .pipeline import StableDiffusionPipeline


DEFAULTS = {
    "prompt": "leisai, purple hair, green eyes, smile, anime girl, best quality",
    "negative_prompt": "low quality, blurry, bad anatomy, worst quality",
    "model_path": "./models/anything-v5",
    "use_lora": False,
    "lora_path": None,
    "lora_strength": 1.0,
    "height": 512,
    "width": 512,
    "steps": 50,
    "guidance_scale": 7.5,
    "seed": 42,
    "output": "output_lora.png",
    "device": None,
    "verbose": True,
}


def build_parser():
    parser = argparse.ArgumentParser(description="Run local SD1.5 inference.")
    parser.add_argument("--prompt", type=str, help="Positive prompt.")
    parser.add_argument("--negative_prompt", type=str, help="Negative prompt.")
    parser.add_argument("--model_path", type=str, help="Local SD1.5 model directory.")
    parser.add_argument("--lora_path", type=str, help="Optional LoRA path.")
    parser.add_argument("--lora_strength", type=float, help="LoRA strength.")
    parser.add_argument("--height", type=int, help="Image height.")
    parser.add_argument("--width", type=int, help="Image width.")
    parser.add_argument("--steps", type=int, help="Number of inference steps.")
    parser.add_argument("--guidance_scale", type=float, help="CFG guidance scale.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    parser.add_argument("--output", type=str, help="Output image path.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], help="Execution device.")
    parser.add_argument("--verbose", action="store_true", help="Print pipeline progress.")
    parser.add_argument("--no_verbose", action="store_true", help="Disable pipeline progress output.")
    parser.add_argument("--use_lora", action="store_true", help="Force-enable LoRA loading.")
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA loading.")
    return parser


def _cli_mode_requested(args):
    return any(value is not None for key, value in vars(args).items() if key not in {"verbose", "no_verbose", "use_lora", "no_lora"}) or args.verbose or args.no_verbose or args.use_lora or args.no_lora


def _resolve_config(args):
    config = DEFAULTS.copy()

    if not _cli_mode_requested(args):
        return config

    # In CLI mode, prefer explicit behavior over file defaults:
    # do not load LoRA unless the user requests it or provides a LoRA path.
    config["use_lora"] = False

    for key in ("prompt", "negative_prompt", "model_path", "lora_path", "lora_strength", "height", "width", "steps", "guidance_scale", "seed", "output", "device"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    if args.lora_path is not None:
        config["use_lora"] = True
    if args.use_lora:
        config["use_lora"] = True
    if args.no_lora:
        config["use_lora"] = False
    if args.verbose:
        config["verbose"] = True
    if args.no_verbose:
        config["verbose"] = False

    return config


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = _resolve_config(args)

    device = config["device"] or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device == "cuda" else torch.float32

    model_path = config["model_path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    lora_path = None
    if config["use_lora"]:
        candidate_lora_path = config["lora_path"]
        if not candidate_lora_path:
            raise ValueError("LoRA is enabled but `lora_path` is empty.")
        if not os.path.exists(candidate_lora_path):
            raise FileNotFoundError(f"LoRA file not found: {candidate_lora_path}")
        lora_path = candidate_lora_path

    print(f"Using device: {device}")
    print(f"Using model: {model_path}")
    if lora_path:
        print(f"Using LoRA: {lora_path} (strength={config['lora_strength']})")
    else:
        print("Running without LoRA.")
    print(f"Prompt: {config['prompt']}")
    print(f"Negative prompt: {config['negative_prompt']}")

    pipeline = StableDiffusionPipeline(
        device=device,
        dtype=dtype,
        model_path=model_path,
        lora_path=lora_path,
        lora_strength=config["lora_strength"],
        verbose=config["verbose"],
    )
    image = pipeline(
        prompt=config["prompt"],
        negative_prompt=config["negative_prompt"],
        height=config["height"],
        width=config["width"],
        num_inference_steps=config["steps"],
        guidance_scale=config["guidance_scale"],
        seed=config["seed"],
    )
    image.save(config["output"])
    print(f"Saved image to {config['output']}")


if __name__ == "__main__":
    main()
