import os

import torch
from diffusers import UNet2DConditionModel

from .lora import load_lora_into_unet, set_lora_strength, unload_lora


LOCAL_UNET_FILES = (
    "config.json",
    "diffusion_pytorch_model.safetensors",
)


def _resolve_dtype(device, dtype):
    if str(device) == "cpu" and dtype == torch.float16:
        return torch.float32
    return dtype


def _has_required_files(base_dir, required_files):
    return all(os.path.exists(os.path.join(base_dir, name)) for name in required_files)


def _resolve_unet_path(model_path):
    if model_path:
        local_path = os.path.join(model_path, "unet")
        if _has_required_files(local_path, LOCAL_UNET_FILES):
            return local_path, True
    return "runwayml/stable-diffusion-v1-5", False


class UNet:
    """Thin wrapper around the SD1.5 UNet with inference-time LoRA support."""

    def __init__(self, device="cuda", dtype=torch.float16, model_path=None):
        self.device = device
        self.dtype = _resolve_dtype(device, dtype)
        self.loaded_lora_path = None
        self.loaded_lora_strength = None

        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        unet_path, local_only = _resolve_unet_path(model_path)

        try:
            self.unet = UNet2DConditionModel.from_pretrained(
                unet_path,
                subfolder=None if local_only else "unet",
                local_files_only=local_only,
            )
        except Exception as exc:
            print(f"Failed to load UNet: {exc}")
            raise

        self.unet.to(device=device, dtype=self.dtype)
        self.unet.eval()

    def load_lora(self, lora_path, strength=1.0, adapter_name="default"):
        applied = load_lora_into_unet(
            unet_or_wrapper=self,
            lora_path=lora_path,
            strength=strength,
            adapter_name=adapter_name,
        )
        self.loaded_lora_path = lora_path
        self.loaded_lora_strength = strength
        print(f"Loaded LoRA: {lora_path} (layers={applied}, strength={strength})")
        return applied

    def unload_lora(self, adapter_name=None):
        removed = unload_lora(self, adapter_name=adapter_name)
        if removed > 0 and adapter_name is None:
            self.loaded_lora_path = None
            self.loaded_lora_strength = None
        print(f"Unloaded LoRA adapters: {removed}")
        return removed

    def set_lora_strength(self, strength, adapter_name=None):
        updated = set_lora_strength(self, strength=strength, adapter_name=adapter_name)
        if updated > 0 and adapter_name is None:
            self.loaded_lora_strength = strength
        print(f"Updated LoRA strength to {strength} on {updated} wrapped layers")
        return updated

    def forward(self, latents, timestep, text_embeddings):
        latents = latents.to(device=self.device, dtype=self.dtype)
        text_embeddings = text_embeddings.to(device=self.device, dtype=self.dtype)

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=self.device, dtype=torch.long)
        else:
            timestep = timestep.to(device=self.device, dtype=torch.long)

        with torch.no_grad():
            noise_pred = self.unet(
                latents,
                timestep,
                encoder_hidden_states=text_embeddings,
            ).sample
        return noise_pred
