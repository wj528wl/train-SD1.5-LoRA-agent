"""
Inference-side LoRA loading utilities for the local SD1.5 pipeline.
"""
import os

import torch
import torch.nn as nn
from safetensors.torch import load_file


class InferenceLoRALayer(nn.Module):
    """A frozen LoRA branch used only during inference."""

    def __init__(self, in_features, out_features, rank, alpha=1.0, strength=1.0, dtype=None, device=None):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.strength = strength
        self.scaling = alpha / rank if rank > 0 else 1.0

        self.lora_down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.lora_up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        self.requires_grad_(False)

    def forward(self, x):
        return self.lora_up(self.lora_down(x)) * self.scaling * self.strength


class LoRAInjectedLinear(nn.Module):
    """Wrap a Linear layer and apply one or more inference LoRA adapters."""

    def __init__(self, base_layer):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRAInjectedLinear can only wrap nn.Linear layers.")

        self.base_layer = base_layer
        self.adapters = nn.ModuleDict()

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    @property
    def in_features(self):
        return self.base_layer.in_features

    @property
    def out_features(self):
        return self.base_layer.out_features

    def add_adapter(self, name, adapter):
        self.adapters[name] = adapter

    def remove_adapter(self, name):
        if name in self.adapters:
            del self.adapters[name]

    def clear_adapters(self):
        self.adapters = nn.ModuleDict()

    def set_strength(self, strength, adapter_name=None):
        if adapter_name is None:
            for adapter in self.adapters.values():
                adapter.strength = strength
            return

        if adapter_name in self.adapters:
            self.adapters[adapter_name].strength = strength

    def forward(self, x):
        output = self.base_layer(x)
        for adapter in self.adapters.values():
            output = output + adapter(x)
        return output


def load_lora_weights(lora_path):
    """Load a LoRA checkpoint from a safetensors or torch checkpoint file."""
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA file not found: {lora_path}")

    if lora_path.endswith(".safetensors"):
        return load_file(lora_path)

    loaded = torch.load(lora_path, map_location="cpu", weights_only=True)
    return loaded.get("state_dict", loaded)


def _combine_tokens(tokens):
    combined = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            pair = (tokens[i], tokens[i + 1])
            if pair in {
                ("down", "blocks"),
                ("up", "blocks"),
                ("transformer", "blocks"),
                ("mid", "block"),
                ("to", "q"),
                ("to", "k"),
                ("to", "v"),
                ("to", "out"),
            }:
                combined.append(f"{tokens[i]}_{tokens[i + 1]}")
                i += 2
                continue
        combined.append(tokens[i])
        i += 1
    return combined


def convert_lora_key_to_module_name(key):
    """
    Convert keys like:
    lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q.lora_down.weight
    into:
    down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q
    """
    if not key.startswith("lora_unet_"):
        raise ValueError(f"Unsupported LoRA key: {key}")

    base = key[len("lora_unet_"):]
    for suffix in (".lora_down.weight", ".lora_up.weight", ".alpha"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    tokens = _combine_tokens(base.split("_"))
    return ".".join(tokens)


def _get_target_model(unet_or_wrapper):
    return unet_or_wrapper.unet if hasattr(unet_or_wrapper, "unet") else unet_or_wrapper


def _get_submodule(root_module, module_name):
    module = root_module
    for part in module_name.split("."):
        if module is None:
            raise AttributeError(f"遍历模块路径 '{module_name}' 时在 '{part}' 之前遇到 None（上一级模块不存在）")
        if part.isdigit():
            try:
                module = module[int(part)]
            except (IndexError, TypeError) as e:
                raise AttributeError(f"无法索引模块 '{module_name}' 中的 [{part}]: {e}")
        else:
            try:
                module = getattr(module, part)
            except AttributeError:
                raise AttributeError(f"模块路径 '{module_name}' 中不存在 '{part}'（当前模块类型: {type(module).__name__}）")
    return module


def _set_submodule(root_module, module_name, new_module):
    parts = module_name.split(".")
    parent = root_module
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def _ensure_injected_linear(root_module, module_name):
    module = _get_submodule(root_module, module_name)
    if isinstance(module, LoRAInjectedLinear):
        return module
    if not isinstance(module, nn.Linear):
        raise TypeError(f"Target module is not nn.Linear: {module_name}")

    wrapped = LoRAInjectedLinear(module)
    _set_submodule(root_module, module_name, wrapped)
    return wrapped


def _group_lora_tensors(lora_state_dict):
    grouped = {}
    for key, value in lora_state_dict.items():
        if not key.startswith("lora_unet_"):
            continue

        module_name = convert_lora_key_to_module_name(key)
        entry = grouped.setdefault(module_name, {})

        if key.endswith(".lora_down.weight"):
            entry["lora_down.weight"] = value
        elif key.endswith(".lora_up.weight"):
            entry["lora_up.weight"] = value
        elif key.endswith(".alpha"):
            entry["alpha"] = value

    return grouped


def apply_lora_to_unet(unet_or_wrapper, lora_state_dict, strength=1.0, adapter_name="default"):
    """
    Dynamically apply a LoRA checkpoint to a UNet model.
    """
    target_model = _get_target_model(unet_or_wrapper)
    grouped = _group_lora_tensors(lora_state_dict)
    applied = 0

    for module_name, tensors in grouped.items():
        if "lora_down.weight" not in tensors or "lora_up.weight" not in tensors:
            continue

        target_module = _ensure_injected_linear(target_model, module_name)
        down_weight = tensors["lora_down.weight"]
        up_weight = tensors["lora_up.weight"]
        alpha_tensor = tensors.get("alpha", torch.tensor(down_weight.shape[0], dtype=torch.float32))
        alpha = float(alpha_tensor.item()) if torch.is_tensor(alpha_tensor) else float(alpha_tensor)

        adapter = InferenceLoRALayer(
            in_features=target_module.in_features,
            out_features=target_module.out_features,
            rank=down_weight.shape[0],
            alpha=alpha,
            strength=strength,
            dtype=target_module.weight.dtype,
            device=target_module.weight.device,
        )
        adapter.lora_down.weight.data.copy_(down_weight.to(device=target_module.weight.device, dtype=target_module.weight.dtype))
        adapter.lora_up.weight.data.copy_(up_weight.to(device=target_module.weight.device, dtype=target_module.weight.dtype))
        adapter.requires_grad_(False)

        target_module.add_adapter(adapter_name, adapter)
        applied += 1

    if applied == 0:
        raise ValueError("No compatible UNet LoRA weights were applied.")

    return applied


def load_lora_into_unet(unet_or_wrapper, lora_path, strength=1.0, adapter_name="default"):
    """Load a LoRA file and apply it to the UNet."""
    lora_state_dict = load_lora_weights(lora_path)
    return apply_lora_to_unet(
        unet_or_wrapper=unet_or_wrapper,
        lora_state_dict=lora_state_dict,
        strength=strength,
        adapter_name=adapter_name,
    )


def set_lora_strength(unet_or_wrapper, strength, adapter_name=None):
    """Update LoRA strength for one adapter or all adapters."""
    target_model = _get_target_model(unet_or_wrapper)
    updated = 0

    for module in target_model.modules():
        if not isinstance(module, LoRAInjectedLinear):
            continue
        module.set_strength(strength, adapter_name=adapter_name)
        updated += 1

    return updated


def unload_lora(unet_or_wrapper, adapter_name=None):
    """
    Remove LoRA adapters from the injected linear wrappers.
    If adapter_name is None, all adapters are removed.
    """
    target_model = _get_target_model(unet_or_wrapper)
    removed = 0

    for module in target_model.modules():
        if not isinstance(module, LoRAInjectedLinear):
            continue

        if adapter_name is None:
            removed += len(module.adapters)
            module.clear_adapters()
        elif adapter_name in module.adapters:
            module.remove_adapter(adapter_name)
            removed += 1

    return removed
