import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from quantize_hf import (
    file_manifest,
    is_monarch_factor_name,
    is_torchao_tensor,
    select_model_loader,
)


class FakeTorchAoTensor:
    pass


FakeTorchAoTensor.__module__ = "torchao.testing"


class Int8QuantizationTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
