"""
Save LoRA weights in a ComfyUI-friendly format.
"""
import os

import torch
from safetensors.torch import load_file, save_file

try:
    from .lora import LoRALinear
except ImportError:
    from lora import LoRALinear


def extract_lora_weights(unet):
    """
    Extract LoRA tensors from a UNet with explicit LoRA wrappers.
    """
    lora_state_dict = {}

    for name, module in unet.named_modules():
        if not isinstance(module, LoRALinear):
            continue

        base_key = name.replace(".", "_")
        lora_state_dict[f"lora_unet_{base_key}.lora_down.weight"] = module.lora.lora_down.weight.detach().cpu()
        lora_state_dict[f"lora_unet_{base_key}.lora_up.weight"] = module.lora.lora_up.weight.detach().cpu()
        lora_state_dict[f"lora_unet_{base_key}.alpha"] = torch.tensor(module.lora.alpha, dtype=torch.float32)

    return lora_state_dict


def save_lora_weights(lora_state_dict, save_path, metadata=None):
    """
    Save LoRA weights to safetensors or PyTorch format.
    """
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    if metadata is None:
        metadata = {}

    metadata.update({
        "architecture": "stable-diffusion-v1",
        "format": "pt",
        "type": "lora",
    })

    if save_path.endswith(".safetensors"):
        save_file(lora_state_dict, save_path, metadata=metadata)
    else:
        torch.save(
            {
                "state_dict": lora_state_dict,
                "metadata": metadata,
            },
            save_path,
        )

    print(f"Saved LoRA weights to: {save_path}")
    print(f"  - Tensor count: {len(lora_state_dict)}")
    print(f"  - File size: {os.path.getsize(save_path) / (1024 * 1024):.2f} MB")


def save_lora_for_comfyui(unet, save_path, rank=4, alpha=4.0, metadata=None):
    """
    Save a UNet LoRA in a ComfyUI-friendly .safetensors file.
    """
    lora_state_dict = extract_lora_weights(unet)
    if not lora_state_dict:
        raise ValueError("No LoRA weights were found in the UNet.")

    if metadata is None:
        metadata = {}

    metadata.update({
        "format": "comfyui",
        "lora_alpha": str(alpha),
        "lora_rank": str(rank),
    })
    save_lora_weights(lora_state_dict, save_path, metadata)

    print("\nUsage:")
    print(f"1. Copy {os.path.basename(save_path)} into ComfyUI/models/loras/")
    print("2. Load it with the ComfyUI 'Load LoRA' node")


def convert_diffusers_to_comfyui_lora(diffusers_lora_path, output_path):
    """
    Convert a Diffusers LoRA checkpoint into the key format used here.
    """
    if diffusers_lora_path.endswith(".safetensors"):
        state_dict = load_file(diffusers_lora_path)
    else:
        loaded = torch.load(diffusers_lora_path, map_location="cpu")
        state_dict = loaded.get("state_dict", loaded)

    comfyui_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace(".processor.", ".").replace("_lora", "")
        comfyui_state_dict[new_key] = value.detach().cpu() if hasattr(value, "detach") else value

    save_lora_weights(comfyui_state_dict, output_path)
