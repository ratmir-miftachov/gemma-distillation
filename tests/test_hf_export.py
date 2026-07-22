import unittest

try:
    from monarch_distill.configuration_monarch_gemma4 import MonarchGemma4Config
    from scripts.export_hf import checkpoint_tensor_names, replaced_dense_weight_names
except ImportError as exc:
    raise unittest.SkipTest(f"Gemma 4 requires a newer Transformers environment: {exc}")


class HuggingFaceExportTest(unittest.TestCase):
    def test_expected_checkpoint_contract(self):
        layers = [34, 33, 32, 31, 30, 29, 28, 27]
        self.assertEqual(len(checkpoint_tensor_names(layers)), 48)
        self.assertEqual(len(replaced_dense_weight_names(layers)), 24)

    def test_subset_checkpoint_contract(self):
        layers = [34, 33, 32, 31]
        self.assertEqual(len(checkpoint_tensor_names(layers)), 24)
        self.assertEqual(len(replaced_dense_weight_names(layers)), 12)

    def test_custom_config_round_trip(self):
        config = MonarchGemma4Config(
            monarch_compressed_layers=[34, 33],
            monarch_blocks_weights=128,
        )
        restored = MonarchGemma4Config.from_dict(config.to_dict())
        self.assertEqual(restored.monarch_compressed_layers, [34, 33])
        self.assertEqual(restored.monarch_blocks_weights, 128)
        self.assertEqual(restored.monarch_factor_count, 2)

    def test_config_rejects_unsupported_factor_count(self):
        with self.assertRaises(ValueError):
            MonarchGemma4Config(monarch_factor_count=3)


if __name__ == "__main__":
    unittest.main()
