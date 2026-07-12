import unittest

import torch
import torch.nn as nn

from monarch_distill.monarch import MonarchLinear


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


if __name__ == "__main__":
    unittest.main()
