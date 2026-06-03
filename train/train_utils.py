"""
训练工具函数
"""
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math
import os
import random
import numpy as np


def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_optimizer(params, config):
    """创建优化器"""
    if config.optimizer.lower() == "adamw":
        optimizer = AdamW(
            params,
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.adam_weight_decay,
            eps=config.adam_epsilon
        )
    else:
        raise ValueError(f"不支持的优化器: {config.optimizer}")
    
    return optimizer


def get_scheduler(optimizer, config, num_training_steps):
    """创建学习率调度器"""
    if config.lr_scheduler == "constant":
        def lr_lambda(step):
            if step < config.lr_warmup_steps:
                return step / max(1, config.lr_warmup_steps)
            return 1.0
    
    elif config.lr_scheduler == "linear":
        def lr_lambda(step):
            if step < config.lr_warmup_steps:
                return step / max(1, config.lr_warmup_steps)
            return max(0.0, (num_training_steps - step) / (num_training_steps - config.lr_warmup_steps))
    
    elif config.lr_scheduler == "cosine":
        def lr_lambda(step):
            if step < config.lr_warmup_steps:
                return step / max(1, config.lr_warmup_steps)
            progress = (step - config.lr_warmup_steps) / (num_training_steps - config.lr_warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    else:
        raise ValueError(f"不支持的调度器: {config.lr_scheduler}")
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler


def compute_snr(timesteps, noise_scheduler):
    """
    计算信噪比 (Signal-to-Noise Ratio)
    用于 Min-SNR 加权
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5
    
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    
    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    
    snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
    return snr


def save_checkpoint(model, optimizer, scheduler, epoch, step, output_dir):
    """保存训练检查点"""
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None
    }
    
    checkpoint_path = os.path.join(output_dir, f"checkpoint-{step}.pt")
    torch.save(checkpoint, checkpoint_path)
    print(f"保存检查点: {checkpoint_path}")
    
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """加载训练检查点"""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    step = checkpoint.get('step', 0)
    
    print(f"加载检查点: epoch={epoch}, step={step}")
    return epoch, step


class AverageMeter:
    """计算并存储平均值和当前值"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
