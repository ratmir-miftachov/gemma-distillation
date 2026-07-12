from transformers import Gemma4ForConditionalGeneration

from .configuration_monarch_gemma4 import MonarchGemma4Config
from .monarch import replace_linear_with_monarch


class MonarchGemma4ForConditionalGeneration(Gemma4ForConditionalGeneration):
    """Gemma 4 with configured language-model MLPs stored as Monarch factors."""

    config_class = MonarchGemma4Config

    def __init__(self, config: MonarchGemma4Config):
        super().__init__(config)
        layers = self.model.language_model.layers
        for layer_index in config.monarch_compressed_layers:
            if layer_index < 0 or layer_index >= len(layers):
                raise ValueError(
                    f"compressed layer index {layer_index} is outside the language model's "
                    f"0..{len(layers) - 1} range"
                )
            replace_linear_with_monarch(
                layers[layer_index].mlp,
                config.monarch_blocks_weights,
                init_method="identity_noise",
                module_path=f"model.language_model.layers.{layer_index}.mlp",
            )
