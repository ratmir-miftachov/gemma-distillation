"""Gemma Monarch distillation package."""

from .config import CompressionConfig, default_config

__all__ = ["CompressionConfig", "MonarchCompressor", "default_config"]


def __getattr__(name):
    if name == "MonarchCompressor":
        from .trainer import MonarchCompressor

        return MonarchCompressor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
