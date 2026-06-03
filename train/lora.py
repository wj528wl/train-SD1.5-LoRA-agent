"""
LoRA layers and injection helpers.
"""
import math

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """Low-rank adaptation branch."""

    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        lora_dtype = self.lora_down.weight.dtype
        x = x.to(dtype=lora_dtype)
        return self.lora_up(self.lora_down(x)) * self.scaling


class LoRALinear(nn.Module):
    """Wrap a frozen Linear layer with a trainable LoRA branch."""

    def __init__(self, base_layer, rank=4, alpha=1.0):
        super().__init__()
        if not isinstance(base_layer, nn.Linear):
            raise TypeError("LoRALinear can only wrap nn.Linear modules.")

        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.lora = LoRALayer(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=rank,
            alpha=alpha,
        ).to(device=base_layer.weight.device, dtype=torch.float32)

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = self.lora(x).to(dtype=base_out.dtype)
        return base_out + lora_out


def _replace_module(root_module, module_name, new_module):
    parts = module_name.split(".")
    parent = root_module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _collect_linear_targets(model, target_modules):
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            continue
        if isinstance(module, nn.Linear) and any(target in name for target in target_modules):
            targets.append((name, module))
    return targets


def inject_lora_to_linear(model, rank=4, alpha=1.0, target_modules=None):
    """
    Inject LoRA wrappers into target Linear layers.
    """
    if target_modules is None:
        target_modules = ("to_q", "to_k", "to_v", "to_out")

    lora_layers = []
    for name, module in _collect_linear_targets(model, target_modules):
        wrapped = LoRALinear(module, rank=rank, alpha=alpha)
        _replace_module(model, name, wrapped)
        lora_layers.append((name, wrapped))
        print(f"Injected LoRA into: {name}")

    return lora_layers


def inject_lora_to_unet(unet, rank=4, alpha=1.0):
    """
    Inject LoRA into the UNet attention projections used by SD1.5.
    """
    target_modules = ("to_q", "to_k", "to_v", "to_out")
    lora_params = []
    wrapped_count = 0

    for name, module in list(unet.named_modules()):
        if isinstance(module, LoRALinear):
            continue
        if not isinstance(module, nn.Linear):
            continue
        if "attn" not in name.lower():
            continue
        if not any(target in name for target in target_modules):
            continue

        wrapped = LoRALinear(module, rank=rank, alpha=alpha)
        _replace_module(unet, name, wrapped)
        lora_params.extend(list(wrapped.lora.parameters()))
        wrapped_count += 1
        print(f"Injected LoRA into UNet: {name}")

    print(f"\nInjected {wrapped_count} LoRA layers into the UNet.")
    return lora_params


def get_lora_state_dict(unet):
    """Extract only the LoRA weights from a model."""
    lora_state_dict = {}

    for name, module in unet.named_modules():
        if isinstance(module, LoRALinear):
            lora_state_dict[f"{name}.lora_down.weight"] = module.lora.lora_down.weight.detach().clone()
            lora_state_dict[f"{name}.lora_up.weight"] = module.lora.lora_up.weight.detach().clone()
            lora_state_dict[f"{name}.alpha"] = torch.tensor(module.lora.alpha, device=module.weight.device)

    return lora_state_dict
