from importlib import import_module


_EXPORTS = {
    "LoRAConfig": ".config",
    "DreamBoothConfig": ".config",
    "ImageTextDataset": ".dataset",
    "DreamBoothDataset": ".dataset",
    "LoRALayer": ".lora",
    "inject_lora_to_linear": ".lora",
    "inject_lora_to_unet": ".lora",
    "get_lora_state_dict": ".lora",
    "save_lora_weights": ".save_lora",
    "save_lora_for_comfyui": ".save_lora",
    "convert_diffusers_to_comfyui_lora": ".save_lora",
    "set_seed": ".train_utils",
    "get_optimizer": ".train_utils",
    "get_scheduler": ".train_utils",
    "compute_snr": ".train_utils",
    "save_checkpoint": ".train_utils",
    "load_checkpoint": ".train_utils",
    "AverageMeter": ".train_utils",
    "train": ".train_lora",
    "build_config_from_args": ".train_lora",
    "add_training_arguments": ".train_lora",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_EXPORTS[name], __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
