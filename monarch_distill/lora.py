from collections.abc import Iterator
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .monarch import is_monarch_linear


LORA_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def iter_lora_targets(model) -> Iterator[tuple[str, torch.nn.Module]]:
    layers = model.model.language_model.layers
    compressed_layers = sorted(int(index) for index in model.config.monarch_compressed_layers)
    for layer_index in compressed_layers:
        mlp = layers[layer_index].mlp
        for projection_name in LORA_PROJECTIONS:
            module = getattr(mlp, projection_name)
            if not is_monarch_linear(module):
                raise TypeError(
                    f"layer {layer_index} {projection_name} is not a MonarchLinear"
                )
            yield (
                f"model.language_model.layers.{layer_index}.mlp.{projection_name}",
                module,
            )


def enable_lora_adapters(model, *, rank: int, alpha: float, dropout: float) -> list[str]:
    enabled = []
    for path, module in iter_lora_targets(model):
        if not hasattr(module, "enable_lora"):
            raise TypeError(f"{path} does not support native Monarch LoRA")
        module.enable_lora(rank, alpha, dropout)
        enabled.append(path)

    model.config.monarch_lora_rank = int(rank)
    model.config.monarch_lora_alpha = float(alpha)
    model.config.monarch_lora_dropout = float(dropout)
    model.config.monarch_lora_target_projections = list(LORA_PROJECTIONS)
    model.config.monarch_lora_format_version = 1
    return enabled


def freeze_except_lora(model) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    trainable = []
    for _, module in iter_lora_targets(model):
        if not int(getattr(module, "lora_rank", 0)):
            raise RuntimeError("LoRA adapters must be enabled before freezing the model")
        module.lora_A.requires_grad = True
        module.lora_B.requires_grad = True
        trainable.extend((module.lora_A, module.lora_B))
    return trainable


def lora_state_dict(model) -> dict[str, torch.Tensor]:
    state = {}
    for name, parameter in model.named_parameters():
        if name.endswith(".lora_A") or name.endswith(".lora_B"):
            state[name] = parameter.detach().cpu().contiguous()
    return state


@torch.no_grad()
def load_lora_state_dict(model, state: dict[str, torch.Tensor]) -> None:
    expected = set(lora_state_dict(model))
    if set(state) != expected:
        difference = sorted(set(state) ^ expected)
        raise RuntimeError(f"LoRA state key mismatch: {difference[:10]}")

    named_parameters = dict(model.named_parameters())
    for name, tensor in state.items():
        parameter = named_parameters[name]
        if tuple(parameter.shape) != tuple(tensor.shape):
            raise RuntimeError(
                f"LoRA state shape mismatch for {name}: "
                f"{tuple(tensor.shape)} != {tuple(parameter.shape)}"
            )
        parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


def lora_inventory(model) -> dict[str, int]:
    modules = list(iter_lora_targets(model))
    parameters = sum(
        module.lora_A.numel() + module.lora_B.numel()
        for _, module in modules
        if int(getattr(module, "lora_rank", 0))
    )
    return {
        "module_count": len(modules),
        "tensor_count": len(lora_state_dict(model)),
        "parameter_count": parameters,
    }


def save_adapter_file(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(lora_state_dict(model), temporary)
    os.replace(temporary, path)


def load_adapter_file(model, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    load_lora_state_dict(model, load_file(path, device="cpu"))
