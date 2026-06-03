import torch
import torch.nn as nn
from diffusers import AutoencoderKL
import os


class VAE:
    """变分自编码器，用于图像和潜在空间之间的转换"""
    
    def __init__(self, device="cuda", dtype=torch.float16, model_path=None):
        self.device = device
        self.dtype = dtype
        
        # 设置镜像源
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        
        if model_path and os.path.exists(model_path):
            vae_path = os.path.join(model_path, "vae")
            local_only = True
        else:
            vae_path = "runwayml/stable-diffusion-v1-5"
            local_only = False
        
        try:
            self.vae = AutoencoderKL.from_pretrained(
                vae_path,
                subfolder="vae" if not local_only else None,
                local_files_only=local_only
            )
        except Exception as e:
            print(f"加载VAE失败: {e}")
            raise
        
        self.vae.to(device=device, dtype=dtype)
        self.vae.eval()
        self.scale_factor = 0.18215
    
    def encode(self, image):
        """
        将图像编码到潜在空间
        Args:
            image: [B, C, H, W] 范围 [-1, 1]
        Returns:
            latents: [B, 4, H//8, W//8]
        """
        with torch.no_grad():
            latents = self.vae.encode(image.to(self.dtype)).latent_dist.sample()
            latents = latents * self.scale_factor
        return latents
    
    def decode(self, latents):
        """
        将潜在空间解码为图像
        Args:
            latents: [B, 4, H//8, W//8]
        Returns:
            image: [B, C, H, W] 范围 [-1, 1]
        """
        latents = latents / self.scale_factor
        with torch.no_grad():
            image = self.vae.decode(latents.to(self.dtype)).sample
        return image
