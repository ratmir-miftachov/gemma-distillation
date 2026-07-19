import unittest

import torch
import torch.nn as nn

from monarch_distill.monarch import (
    MonarchLinear,
    densify_monarch_linears,
    is_monarch_linear,
    materialize_monarch_linear,
)


def dense_weight(layer: MonarchLinear) -> torch.Tensor:
    inputs = torch.eye(layer.in_features, dtype=layer.blk1.dtype, device=layer.blk1.device)
    outputs = layer(inputs)
    if layer.bias is not None:
        outputs = outputs - layer.bias
    return outputs.transpose(0, 1).contiguous()


class DenseProjectionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)

    def assert_exact_projection(self, in_features: int, out_features: int, n_blocks: int):
        source = MonarchLinear(in_features, out_features, n_blocks, bias=True)
        with torch.no_grad():
            source.blk1.normal_()
            source.blk2.normal_()
            source.bias.normal_()

        dense = nn.Linear(in_features, out_features, bias=True)
        with torch.no_grad():
            dense.weight.copy_(dense_weight(source))
            dense.bias.copy_(source.bias)

        projected = MonarchLinear(in_features, out_features, n_blocks, bias=True)
        relative_error = projected.initialize_from_dense(dense)

        self.assertLess(relative_error, 1e-5)
        torch.testing.assert_close(dense_weight(projected), dense.weight, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(projected.bias, dense.bias)

    def test_exact_projection_expansion(self):
        self.assert_exact_projection(in_features=12, out_features=24, n_blocks=3)

    def test_exact_projection_contraction(self):
        self.assert_exact_projection(in_features=24, out_features=12, n_blocks=3)

    def test_projection_improves_over_identity_noise(self):
        dense = nn.Linear(12, 24, bias=False)
        identity_noise = MonarchLinear(12, 24, 3, bias=False)
        projected = MonarchLinear(12, 24, 3, bias=False)

        baseline_error = torch.linalg.vector_norm(dense.weight - dense_weight(identity_noise))
        reported_error = projected.initialize_from_dense(dense)
        projected_error = torch.linalg.vector_norm(dense.weight - dense_weight(projected))
        measured_relative_error = projected_error / torch.linalg.vector_norm(dense.weight)

        self.assertLessEqual(projected_error.item(), baseline_error.item())
        self.assertAlmostEqual(reported_error, measured_relative_error.item(), places=5)

    def test_meta_device_construction(self):
        with torch.device("meta"):
            layer = MonarchLinear(12, 24, 3, bias=False)
        self.assertTrue(layer.blk1.is_meta)
        self.assertTrue(layer.blk2.is_meta)

    def test_direct_dense_materialization_matches_expansion_forward(self):
        self.assert_dense_materialization(8, 12, 2)

    def test_direct_dense_materialization_matches_contraction_forward(self):
        self.assert_dense_materialization(12, 8, 2)

    def assert_dense_materialization(self, in_features, out_features, n_blocks):
        layer = MonarchLinear(in_features, out_features, n_blocks, bias=True).double()
        with torch.no_grad():
            layer.blk1.normal_()
            layer.blk2.normal_()
            layer.bias.normal_()
        dense = layer.to_dense_linear()
        inputs = torch.randn(7, in_features, dtype=torch.float64)
        torch.testing.assert_close(dense(inputs), layer(inputs), rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(
            dense.weight,
            layer.materialize_dense_weight(),
            rtol=0,
            atol=0,
        )

    def test_structural_materializer_supports_remote_code_class(self):
        class MonarchLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                source = globals()["MonarchLinear"](8, 12, 2, bias=False)
                self.in_features = source.in_features
                self.out_features = source.out_features
                self.blk1 = torch.nn.Parameter(source.blk1.detach().clone())
                self.blk2 = torch.nn.Parameter(source.blk2.detach().clone())
                self.bias = None

        remote = MonarchLinear()
        self.assertTrue(is_monarch_linear(remote))
        dense = materialize_monarch_linear(remote)
        self.assertEqual(tuple(dense.weight.shape), (12, 8))

    def test_recursive_densification_reports_exact_paths(self):
        model = torch.nn.Sequential(
            MonarchLinear(8, 12, 2),
            torch.nn.Sequential(MonarchLinear(12, 8, 2)),
        )
        inputs = torch.randn(3, 8)
        expected = model(inputs)
        replaced = densify_monarch_linears(model)
        self.assertEqual(replaced, ["0", "1.0"])
        self.assertIsInstance(model[0], torch.nn.Linear)
        self.assertIsInstance(model[1][0], torch.nn.Linear)
        torch.testing.assert_close(model(inputs), expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
