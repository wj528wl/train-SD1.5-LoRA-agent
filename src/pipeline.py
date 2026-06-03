import os
import numpy as np
import torch
from PIL import Image
from diffusers import DDIMScheduler

from .clip import CLIPTextEncoder
from .unet import UNet
from .vae import VAE


def _resolve_dtype(device, dtype):
    if str(device) == "cpu" and dtype == torch.float16:
        return torch.float32
    return dtype


class StableDiffusionPipeline:
    """Minimal SD1.5 text-to-image pipeline."""

    def __init__(
        self,
        device="cuda",
        dtype=torch.float16,
        model_path=None,
        lora_path=None,
        lora_strength=1.0,
        verbose=True,
    ):
        self.device = device
        self.dtype = _resolve_dtype(device, dtype)
        self.verbose = verbose

        self._log("Loading CLIP...")
        self.clip = CLIPTextEncoder(device=device, dtype=self.dtype, model_path=model_path)

        self._log("Loading UNet...")
        self.unet = UNet(device=device, dtype=self.dtype, model_path=model_path)

        self._log("Loading VAE...")
        self.vae = VAE(device=device, dtype=self.dtype, model_path=model_path)

        self._log("Loading scheduler...")
        # 优先从本地加载官方 DDIM，本地没有则从 HuggingFace 下载
        if model_path and os.path.isdir(os.path.join(model_path, "scheduler")):
            self.scheduler = DDIMScheduler.from_pretrained(
                model_path, subfolder="scheduler"
            )
        else:
            self.scheduler = DDIMScheduler.from_pretrained(
                "runwayml/stable-diffusion-v1-5", subfolder="scheduler"
            )

        if lora_path:
            self._log(f"Loading LoRA from {lora_path}...")
            self.unet.load_lora(lora_path, strength=lora_strength)

        self._log("Pipeline ready.")

    def _log(self, message):
        if self.verbose:
            print(message)

    def load_lora(self, lora_path, strength=1.0, adapter_name="default"):
        return self.unet.load_lora(lora_path, strength=strength, adapter_name=adapter_name)

    def unload_lora(self, adapter_name=None):
        return self.unet.unload_lora(adapter_name=adapter_name)

    def set_lora_strength(self, strength, adapter_name=None):
        return self.unet.set_lora_strength(strength=strength, adapter_name=adapter_name)

    def _validate_image_size(self, height, width):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError("`height` and `width` must both be divisible by 8.")
        if height <= 0 or width <= 0:
            raise ValueError("`height` and `width` must both be positive.")

    def _prepare_latents(self, height, width, seed=None):
        latent_height = height // 8
        latent_width = width // 8
        generator = None

        if seed is not None:
            np.random.seed(seed)
            generator = torch.Generator(device=self.device)
            generator.manual_seed(seed)

        return torch.randn(
            (1, 4, latent_height, latent_width),
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        seed=None,
    ):
        self._validate_image_size(height, width)
        positive_embeds, negative_embeds = self.clip.encode(prompt, negative_prompt)
        latents = self._prepare_latents(height, width, seed=seed)

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        self._log(f"Running denoising for {num_inference_steps} steps...")

        for step_index, timestep in enumerate(self.scheduler.timesteps, start=1):
            latent_model_input = self.scheduler.scale_model_input(latents, timestep)

            if guidance_scale > 1.0:
                latent_model_input = torch.cat([latent_model_input, latent_model_input], dim=0)
                text_embeddings = torch.cat([negative_embeds, positive_embeds], dim=0)
                noise_pred = self.unet.forward(latent_model_input, timestep, text_embeddings)
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred = self.unet.forward(latent_model_input, timestep, positive_embeds)

            step_out = self.scheduler.step(noise_pred, timestep, latents)
            # diffusers 0.17+ 返回 DDIMSchedulerOutput，兼容 tuple
            latents = step_out.prev_sample if hasattr(step_out, "prev_sample") else step_out[0]

            if self.verbose and (step_index == num_inference_steps or step_index % 10 == 0):
                print(f"  step {step_index}/{num_inference_steps}")

        image = self.vae.decode(latents)
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
        return Image.fromarray(image)
