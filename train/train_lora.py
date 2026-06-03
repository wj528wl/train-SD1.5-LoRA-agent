"""
Core LoRA training utilities for SD1.5 models.
"""
import json
import os
from contextlib import nullcontext
from dataclasses import asdict

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer

try:
    from .config import DreamBoothConfig, LoRAConfig
    from .dataset import DreamBoothDataset, ImageTextDataset
    from .lora import get_lora_state_dict, inject_lora_to_unet
    from .save_lora import save_lora_for_comfyui
    from .train_utils import AverageMeter, get_optimizer, get_scheduler, set_seed
except ImportError:
    from config import DreamBoothConfig, LoRAConfig
    from dataset import DreamBoothDataset, ImageTextDataset
    from lora import get_lora_state_dict, inject_lora_to_unet
    from save_lora import save_lora_for_comfyui
    from train_utils import AverageMeter, get_optimizer, get_scheduler, set_seed


LOCAL_TOKENIZER_FILES = (
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
)
LOCAL_TEXT_ENCODER_FILES = (
    "config.json",
    "model.safetensors",
)
LOCAL_COMPONENT_FILES = {
    "vae": ("config.json", "diffusion_pytorch_model.safetensors"),
    "unet": ("config.json", "diffusion_pytorch_model.safetensors"),
    "scheduler": ("scheduler_config.json",),
}


def _has_required_files(base_dir, required_files):
    return all(os.path.exists(os.path.join(base_dir, file_name)) for file_name in required_files)


def _resolve_model_paths(model_path):
    if model_path:
        tokenizer_path = os.path.join(model_path, "tokenizer")
        text_encoder_path = os.path.join(model_path, "text_encoder")
        vae_path = os.path.join(model_path, "vae")
        unet_path = os.path.join(model_path, "unet")
        scheduler_path = os.path.join(model_path, "scheduler")

        has_local_model = all(
            [
                _has_required_files(tokenizer_path, LOCAL_TOKENIZER_FILES),
                _has_required_files(text_encoder_path, LOCAL_TEXT_ENCODER_FILES),
                _has_required_files(vae_path, LOCAL_COMPONENT_FILES["vae"]),
                _has_required_files(unet_path, LOCAL_COMPONENT_FILES["unet"]),
                _has_required_files(scheduler_path, LOCAL_COMPONENT_FILES["scheduler"]),
            ]
        )
        if has_local_model:
            return {
                "tokenizer": tokenizer_path,
                "text_encoder": text_encoder_path,
                "vae": vae_path,
                "unet": unet_path,
                "scheduler": scheduler_path,
                "local_files_only": True,
            }

    return {
        "tokenizer": "openai/clip-vit-large-patch14",
        "text_encoder": "openai/clip-vit-large-patch14",
        "vae": "runwayml/stable-diffusion-v1-5",
        "unet": "runwayml/stable-diffusion-v1-5",
        "scheduler": "runwayml/stable-diffusion-v1-5",
        "local_files_only": False,
    }


def _get_device(requested_device):
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested_device)


def _get_weight_dtype(device, mixed_precision):
    if device.type != "cuda":
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def _get_autocast_context(device, mixed_precision):
    if device.type != "cuda":
        return nullcontext()
    if mixed_precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if mixed_precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _tokenize_captions(tokenizer, captions):
    return tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids


def collate_image_text(examples):
    pixel_values = torch.stack([example["image"] for example in examples])
    captions = [example["caption"] for example in examples]
    return {
        "pixel_values": pixel_values.contiguous(),
        "captions": captions,
    }


def collate_dreambooth(examples):
    batch = {
        "instance_pixel_values": torch.stack([example["instance_image"] for example in examples]).contiguous(),
        "instance_prompts": [example["instance_prompt"] for example in examples],
    }

    if "class_image" in examples[0]:
        batch["class_pixel_values"] = torch.stack([example["class_image"] for example in examples]).contiguous()
        batch["class_prompts"] = [example["class_prompt"] for example in examples]

    return batch


def build_dataset_and_loader(config):
    if isinstance(config, DreamBoothConfig):
        dataset = DreamBoothDataset(
            instance_dir=config.instance_dir,
            class_dir=config.class_dir or None,
            size=config.resolution,
            instance_prompt=config.instance_prompt,
            class_prompt=config.class_prompt,
        )
        collate_fn = collate_dreambooth
    else:
        dataset = ImageTextDataset(
            data_dir=config.data_dir,
            size=config.resolution,
            center_crop=config.center_crop,
        )
        collate_fn = collate_image_text

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn,
    )
    return dataset, loader


def load_models(config, device, weight_dtype):
    paths = _resolve_model_paths(config.model_path)
    local_only = paths["local_files_only"]

    tokenizer = CLIPTokenizer.from_pretrained(paths["tokenizer"], local_files_only=local_only)
    text_encoder = CLIPTextModel.from_pretrained(
        paths["text_encoder"],
        local_files_only=local_only,
    ).to(device=device, dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(
        paths["vae"],
        subfolder=None if local_only else "vae",
        local_files_only=local_only,
    ).to(device=device, dtype=weight_dtype)
    unet = UNet2DConditionModel.from_pretrained(
        paths["unet"],
        subfolder=None if local_only else "unet",
        local_files_only=local_only,
    ).to(device=device, dtype=weight_dtype)
    noise_scheduler = DDPMScheduler.from_pretrained(
        paths["scheduler"],
        subfolder=None if local_only else "scheduler",
        local_files_only=local_only,
    )

    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    text_encoder.eval()
    vae.eval()
    unet.train()
    return tokenizer, text_encoder, vae, unet, noise_scheduler


def _compute_standard_lora_loss(batch, tokenizer, text_encoder, vae, unet, noise_scheduler, device, weight_dtype):
    pixel_values = batch["pixel_values"].to(device=device, dtype=weight_dtype)
    input_ids = _tokenize_captions(tokenizer, batch["captions"]).to(device)

    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        encoder_hidden_states = text_encoder(input_ids)[0]

    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (latents.shape[0],),
        device=device,
        dtype=torch.long,
    )
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample

    target = noise
    if noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)

    return F.mse_loss(model_pred.float(), target.float(), reduction="mean")


def _compute_prior_batch_loss(pixel_values, prompts, tokenizer, text_encoder, vae, unet, noise_scheduler, device, weight_dtype):
    pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
    input_ids = _tokenize_captions(tokenizer, prompts).to(device)

    with torch.no_grad():
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        hidden_states = text_encoder(input_ids)[0]

    noise = torch.randn_like(latents)
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (latents.shape[0],),
        device=device,
        dtype=torch.long,
    )
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=hidden_states).sample

    target = noise
    if noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)

    return F.mse_loss(model_pred.float(), target.float(), reduction="mean")


def compute_loss(batch, tokenizer, text_encoder, vae, unet, noise_scheduler, device, weight_dtype, config):
    if "pixel_values" in batch:
        return _compute_standard_lora_loss(
            batch,
            tokenizer,
            text_encoder,
            vae,
            unet,
            noise_scheduler,
            device,
            weight_dtype,
        )

    instance_loss = _compute_prior_batch_loss(
        batch["instance_pixel_values"],
        batch["instance_prompts"],
        tokenizer,
        text_encoder,
        vae,
        unet,
        noise_scheduler,
        device,
        weight_dtype,
    )
    if "class_pixel_values" not in batch:
        return instance_loss

    prior_loss = _compute_prior_batch_loss(
        batch["class_pixel_values"],
        batch["class_prompts"],
        tokenizer,
        text_encoder,
        vae,
        unet,
        noise_scheduler,
        device,
        weight_dtype,
    )
    return instance_loss + config.prior_loss_weight * prior_loss


def save_training_snapshot(unet, optimizer, scheduler, config, epoch, global_step, suffix):
    os.makedirs(config.output_dir, exist_ok=True)
    lora_state_dict = get_lora_state_dict(unet)

    lora_path = os.path.join(config.output_dir, f"{suffix}.safetensors")
    save_lora_for_comfyui(
        unet=unet,
        save_path=lora_path,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        metadata={
            "epoch": str(epoch),
            "step": str(global_step),
            "base_model": config.model_path,
        },
    )

    trainer_state_path = os.path.join(config.output_dir, f"{suffix}_trainer_state.pt")
    torch.save(
        {
            "epoch": epoch,
            "step": global_step,
            "lora_state_dict": {k: v.detach().cpu() for k, v in lora_state_dict.items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "config": asdict(config),
        },
        trainer_state_path,
    )


def cleanup_old_checkpoints(output_dir, save_total_limit):
    if save_total_limit <= 0 or not os.path.exists(output_dir):
        return

    checkpoint_files = []
    for file_name in os.listdir(output_dir):
        if file_name.startswith("step-") and file_name.endswith(".safetensors"):
            full_path = os.path.join(output_dir, file_name)
            checkpoint_files.append((os.path.getmtime(full_path), full_path))

    checkpoint_files.sort(key=lambda item: item[0])
    stale = checkpoint_files[:-save_total_limit]
    for _, path in stale:
        trainer_state_path = path.replace(".safetensors", "_trainer_state.pt")
        if os.path.exists(path):
            os.remove(path)
        if os.path.exists(trainer_state_path):
            os.remove(trainer_state_path)


def train(config):
    set_seed(config.seed)
    device = _get_device(config.device)
    weight_dtype = _get_weight_dtype(device, config.mixed_precision)

    print(f"Using device: {device}")
    print(f"Weight dtype: {weight_dtype}")
    print(f"Model path: {config.model_path}")

    tokenizer, text_encoder, vae, unet, noise_scheduler = load_models(config, device, weight_dtype)
    dataset, train_dataloader = build_dataset_and_loader(config)

    lora_params = inject_lora_to_unet(unet, rank=config.lora_rank, alpha=config.lora_alpha)
    if not lora_params:
        raise RuntimeError("No LoRA layers were injected into the UNet.")

    optimizer = get_optimizer(lora_params, config)
    total_update_steps = max(
        1,
        (len(train_dataloader) * config.num_epochs) // config.gradient_accumulation_steps,
    )
    scheduler = get_scheduler(optimizer, config, total_update_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and config.mixed_precision == "fp16")

    global_step = 0
    print(f"Training samples: {len(dataset)}")
    print(f"Batches per epoch: {len(train_dataloader)}")
    print(f"Total update steps: {total_update_steps}")

    for epoch in range(config.num_epochs):
        loss_meter = AverageMeter()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_dataloader):
            with _get_autocast_context(device, config.mixed_precision):
                loss = compute_loss(
                    batch=batch,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    vae=vae,
                    unet=unet,
                    noise_scheduler=noise_scheduler,
                    device=device,
                    weight_dtype=weight_dtype,
                    config=config,
                )
                loss = loss / config.gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            should_step = (step + 1) % config.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader)
            loss_meter.update(loss.item() * config.gradient_accumulation_steps, n=1)
            if not should_step:
                continue

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, config.max_grad_norm)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % config.logging_steps == 0:
                lr = scheduler.get_last_lr()[0] if scheduler else config.learning_rate
                print(
                    f"epoch={epoch + 1}/{config.num_epochs} "
                    f"step={global_step} "
                    f"loss={loss_meter.avg:.6f} "
                    f"lr={lr:.8f}"
                )

            if config.save_steps > 0 and global_step % config.save_steps == 0:
                suffix = f"step-{global_step}"
                save_training_snapshot(unet, optimizer, scheduler, config, epoch + 1, global_step, suffix)
                cleanup_old_checkpoints(config.output_dir, config.save_total_limit)

        print(f"Epoch {epoch + 1} finished, average loss: {loss_meter.avg:.6f}")

    save_training_snapshot(unet, optimizer, scheduler, config, config.num_epochs, global_step, "final_lora")
    print("Training complete.")


def add_training_arguments(parser):
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file.")
    parser.add_argument("--dreambooth", action="store_true", help="Enable DreamBooth-style training.")

    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--center_crop", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--lr_scheduler", type=str, default=None)
    parser.add_argument("--lr_warmup_steps", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--adam_beta1", type=float, default=None)
    parser.add_argument("--adam_beta2", type=float, default=None)
    parser.add_argument("--adam_weight_decay", type=float, default=None)
    parser.add_argument("--adam_epsilon", type=float, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--mixed_precision", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=float, default=None)

    parser.add_argument("--instance_dir", type=str, default=None)
    parser.add_argument("--class_dir", type=str, default=None)
    parser.add_argument("--instance_prompt", type=str, default=None)
    parser.add_argument("--class_prompt", type=str, default=None)
    parser.add_argument("--prior_loss_weight", type=float, default=None)
    return parser


def build_config_from_args(args):
    config_cls = DreamBoothConfig if args.dreambooth else LoRAConfig
    config_kwargs = {}

    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            config_kwargs.update(json.load(handle))

    for key, value in vars(args).items():
        if key in {"config", "dreambooth"} or value is None:
            continue
        config_kwargs[key] = value

    return config_cls(**config_kwargs)
