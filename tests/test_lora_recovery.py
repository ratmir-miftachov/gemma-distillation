import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from monarch_distill.config import LoRARecoveryConfig
from monarch_distill.lora import (
    enable_lora_adapters,
    freeze_except_lora,
    load_lora_state_dict,
    load_adapter_file,
    lora_inventory,
    lora_state_dict,
    save_adapter_file,
)
from monarch_distill.losses import (
    compute_distillation_metric_sums,
    normalize_distillation_metric_sums,
)
from monarch_distill.monarch import MonarchLinear

try:
    from monarch_distill.configuration_monarch_gemma4 import MonarchGemma4Config
except ImportError:
    MonarchGemma4Config = None


class ToyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = MonarchLinear(8, 16, 2, bias=False)
        self.up_proj = MonarchLinear(8, 16, 2, bias=False)
        self.down_proj = MonarchLinear(16, 8, 2, bias=False)


class ToyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = ToyMLP()


class ToyLanguageModel(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)


class ToyRoot(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.language_model = ToyLanguageModel(layers)


class ToyModel(torch.nn.Module):
    def __init__(self, layer_count=2):
        super().__init__()
        self.model = ToyRoot([ToyLayer() for _ in range(layer_count)])
        self.config = SimpleNamespace(monarch_compressed_layers=list(range(layer_count)))


class MonarchLoRATest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def test_zero_initialized_adapter_is_exact_noop(self):
        layer = MonarchLinear(8, 16, 2, bias=True)
        inputs = torch.randn(4, 8)
        expected = layer(inputs)
        base_keys = set(layer.state_dict())

        layer.enable_lora(rank=4, alpha=8, dropout=0)
        actual = layer(inputs)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(base_keys, {"blk1", "blk2", "bias"})
        self.assertTrue(torch.count_nonzero(layer.lora_B).item() == 0)

    def test_adapter_receives_gradients_and_materializes(self):
        layer = MonarchLinear(8, 16, 2, bias=False, lora_rank=4, lora_alpha=8)
        with torch.no_grad():
            layer.lora_B.normal_()
        inputs = torch.randn(5, 8)
        outputs = layer(inputs)
        outputs.square().mean().backward()
        self.assertGreater(layer.lora_A.grad.norm().item(), 0)
        self.assertGreater(layer.lora_B.grad.norm().item(), 0)

        dense = layer.to_dense_linear()
        torch.testing.assert_close(dense(inputs), outputs, rtol=1e-5, atol=1e-5)

    def test_meta_device_construction_with_lora(self):
        with torch.device("meta"):
            layer = MonarchLinear(8, 16, 2, lora_rank=4, lora_alpha=8)
        self.assertTrue(layer.lora_A.is_meta)
        self.assertTrue(layer.lora_B.is_meta)

    def test_inventory_freezing_and_exact_state_round_trip(self):
        source = ToyModel(layer_count=2)
        enabled = enable_lora_adapters(source, rank=4, alpha=8, dropout=0)
        trainable = freeze_except_lora(source)
        inventory = lora_inventory(source)
        self.assertEqual(len(enabled), 6)
        self.assertEqual(inventory["module_count"], 6)
        self.assertEqual(inventory["tensor_count"], 12)
        self.assertEqual(sum(parameter.numel() for parameter in trainable), 576)
        self.assertTrue(all(parameter.requires_grad for parameter in trainable))
        self.assertTrue(
            all(
                parameter.requires_grad == (name.endswith("lora_A") or name.endswith("lora_B"))
                for name, parameter in source.named_parameters()
            )
        )

        with torch.no_grad():
            for name, parameter in source.named_parameters():
                if name.endswith(".lora_A") or name.endswith(".lora_B"):
                    parameter.normal_()
        expected = lora_state_dict(source)
        restored = ToyModel(layer_count=2)
        enable_lora_adapters(restored, rank=4, alpha=8, dropout=0)
        load_lora_state_dict(restored, expected)
        for name, tensor in lora_state_dict(restored).items():
            torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "adapter_model.safetensors"
            save_adapter_file(source, path)
            load_adapter_file(restored, path)
            for name, tensor in lora_state_dict(restored).items():
                torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)

    def test_production_parameter_budget(self):
        rank = 8
        per_layer = rank * (
            (1536 + 6144) + (1536 + 6144) + (6144 + 1536)
        )
        self.assertEqual(per_layer * 35, 6_451_200)


class RecoveryMetricsTest(unittest.TestCase):
    def test_identical_logits_have_zero_kl_and_full_agreement(self):
        logits = torch.randn(2, 4, 11, requires_grad=True)
        weights = torch.tensor([[1.0, 1.0, 0.2, 0.0], [1.0, 0.0, 1.0, 1.0]])
        sums = compute_distillation_metric_sums(logits, logits.detach(), weights, 3)
        metrics = normalize_distillation_metric_sums(sums)
        self.assertLess(metrics["true_kl"].item(), 1e-6)
        self.assertEqual(metrics["top1_agreement"].item(), 1.0)
        metrics["cross_entropy"].backward()
        self.assertIsNotNone(logits.grad)

    def test_recovery_config_and_model_config_defaults(self):
        recovery = LoRARecoveryConfig()
        self.assertEqual(recovery.lora_rank, 8)
        self.assertEqual(recovery.recovery_steps, 2000)
        if MonarchGemma4Config is None:
            self.skipTest("installed Transformers does not include Gemma 4")
        config = MonarchGemma4Config()
        self.assertEqual(config.monarch_lora_rank, 0)
        self.assertEqual(
            config.monarch_lora_target_projections,
            ["gate_proj", "up_proj", "down_proj"],
        )


if __name__ == "__main__":
    unittest.main()
