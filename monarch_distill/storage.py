from __future__ import annotations

from typing import Any


def is_torchao_tensor(value: Any) -> bool:
    return any(cls.__module__.startswith("torchao") for cls in type(value).mro())


def tensor_storage_bytes(value: Any) -> int:
    if is_torchao_tensor(value) and hasattr(value, "tensor_data_names"):
        names = list(value.tensor_data_names)
        names.extend(getattr(value, "optional_tensor_data_names", ()))
        return sum(
            tensor_storage_bytes(component)
            for name in names
            if (component := getattr(value, name, None)) is not None
        )
    return int(value.numel() * value.element_size())


def model_storage_bytes(model: Any, *, include_buffers: bool = True) -> int:
    total = sum(tensor_storage_bytes(value) for value in model.parameters())
    if include_buffers:
        total += sum(tensor_storage_bytes(value) for value in model.buffers())
    return total
