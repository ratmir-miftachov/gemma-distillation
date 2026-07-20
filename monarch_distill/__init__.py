"""Gemma Monarch distillation package."""

from .config import (
    CompressionConfig,
    LoRARecoveryConfig,
    default_config,
    default_lora_recovery_config,
)

__all__ = [
    "CompressionConfig",
    "LoRARecoveryConfig",
    "MonarchCompressor",
    "default_config",
    "default_lora_recovery_config",
]


def __getattr__(name):
    if name == "MonarchCompressor":
        from .trainer import MonarchCompressor

        return MonarchCompressor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
