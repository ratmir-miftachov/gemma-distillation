import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from quantize_hf import (
    TORCHAO_INT8_CONFIG_VERSION,
    audit_quantized_model,
    file_manifest,
    is_monarch_factor_name,
    is_torchao_tensor,
    select_model_loader,
)


class FakeTorchAoTensor:
    pass


FakeTorchAoTensor.__module__ = "torchao.testing"


class Int8QuantizationTest(unittest.TestCase):
    def test_serializable_torchao_config_version_is_pinned(self):
        self.assertEqual(TORCHAO_INT8_CONFIG_VERSION, 2)

    def test_loader_detection_matches_multimodal_model(self):
        config = SimpleNamespace(
            auto_map={"AutoModelForImageTextToText": "modeling.MonarchModel"},
            is_encoder_decoder=False,
        )
        self.assertEqual(select_model_loader(config), "AutoModelForImageTextToText")

    def test_monarch_factor_names_are_exact(self):
        self.assertTrue(is_monarch_factor_name("layer.mlp.gate_proj.blk1"))
        self.assertTrue(is_monarch_factor_name("layer.mlp.down_proj.blk2"))
        self.assertFalse(is_monarch_factor_name("layer.mlp.gate_proj.weight"))
        self.assertFalse(is_monarch_factor_name("layer.blk1.scale"))

    def test_torchao_tensor_detection_uses_class_lineage(self):
        self.assertTrue(is_torchao_tensor(FakeTorchAoTensor()))
        self.assertFalse(is_torchao_tensor(object()))

    def test_file_manifest_is_sorted_and_hashes_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.txt").write_text("second", encoding="utf-8")
            (root / "a.txt").write_text("first", encoding="utf-8")
            records = file_manifest(root)
        self.assertEqual([record["path"] for record in records], ["a.txt", "b.txt"])
        self.assertEqual(
            records[0]["sha256"], hashlib.sha256(b"first").hexdigest()
        )

    def test_audit_allows_only_a_linear_tied_to_an_embedding(self):
        import torch

        class TiedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(8, 4)
                self.lm_head = torch.nn.Linear(4, 8, bias=False)
                self.lm_head.weight = self.embed.weight

            def get_memory_footprint(self, return_buffers=True):
                return sum(value.numel() * value.element_size() for value in self.parameters())

        model = TiedModel()
        with self.assertRaisesRegex(RuntimeError, "expected 210 Monarch factors"):
            audit_quantized_model(model, torch)


if __name__ == "__main__":
    unittest.main()
