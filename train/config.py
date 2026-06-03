"""
训练配置
"""
from dataclasses import dataclass


@dataclass
class LoRAConfig:
    """LoRA训练配置"""
    
    # 模型配置
    model_path: str = "./models/stable-diffusion-v1-5"
    lora_rank: int = 8
    lora_alpha: float = 8.0
    
    # 数据配置
    data_dir: str = "./data"
    resolution: int = 512
    center_crop: bool = True
    
    # 训练配置
    batch_size: int = 1
    num_epochs: int = 20
    learning_rate: float = 1e-4
    lr_scheduler: str = "constant"  # constant, linear, cosine
    lr_warmup_steps: int = 0
    
    # 优化器配置
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    
    # 梯度配置
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # 保存配置
    output_dir: str = "./outputs/lora_full"
    save_steps: int = 500
    save_total_limit: int = 5
    
    # 日志配置
    logging_steps: int = 10
    
    # 设备配置
    device: str = "cuda"
    mixed_precision: str = "fp16"  # no, fp16, bf16
    
    # 其他
    seed: int = 42
    num_workers: int = 4
    
    def __post_init__(self):
        """验证配置"""
        assert self.lora_rank > 0, "lora_rank 必须大于 0"
        assert self.batch_size > 0, "batch_size 必须大于 0"
        assert self.learning_rate > 0, "learning_rate 必须大于 0"


@dataclass
class DreamBoothConfig(LoRAConfig):
    """DreamBooth训练配置"""
    
    # DreamBooth特定配置
    instance_dir: str = "./data/instance"
    class_dir: str = ""
    instance_prompt: str = "a photo of sks person"
    class_prompt: str = "a photo of person"
    prior_loss_weight: float = 1.0
    
    # 覆盖默认值
    num_epochs: int = 100
    learning_rate: float = 5e-6
