from transformers import Gemma4Config


class MonarchGemma4Config(Gemma4Config):
    """Gemma 4 configuration with selected MLPs replaced by Monarch factors."""

    model_type = "monarch_gemma4"

    def __init__(
        self,
        monarch_compressed_layers=None,
        monarch_blocks_weights=128,
        monarch_factor_count=2,
        monarch_format_version=1,
        monarch_base_model="google/gemma-4-E2B-it",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.monarch_compressed_layers = list(monarch_compressed_layers or [])
        self.monarch_blocks_weights = int(monarch_blocks_weights)
        self.monarch_factor_count = int(monarch_factor_count)
        self.monarch_format_version = int(monarch_format_version)
        self.monarch_base_model = str(monarch_base_model)

        if self.monarch_factor_count != 2:
            raise ValueError("this model format supports exactly two Monarch factors")
        if len(set(self.monarch_compressed_layers)) != len(self.monarch_compressed_layers):
            raise ValueError("monarch_compressed_layers contains duplicate layer indices")
