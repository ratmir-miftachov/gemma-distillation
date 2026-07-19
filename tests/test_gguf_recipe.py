import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmark_gguf import benchmark_model
from densify_hf import EXPECTED_MONARCH_LINEAR_COUNT, expected_monarch_paths
from gguf_recipe import regex_escape_tensor_name, write_dynamic_recipe
from gguf_logit_fidelity import distribution_metrics, percentile

import torch


class GgufRecipeTest(unittest.TestCase):
    def test_expected_dense_inventory_covers_all_35_mlps(self):
        paths = expected_monarch_paths()
        self.assertEqual(len(paths), EXPECTED_MONARCH_LINEAR_COUNT)
        self.assertIn("model.language_model.layers.0.mlp.gate_proj", paths)
        self.assertIn("model.language_model.layers.34.mlp.down_proj", paths)

    def test_tensor_name_regex_is_exact_and_escaped(self):
        self.assertEqual(
            regex_escape_tensor_name("blk.0.ffn_up.weight"),
            r"^blk\.0\.ffn_up\.weight$",
        )

    def test_dynamic_recipe_replays_every_official_tensor_type(self):
        official = {
            "token_embd.weight": "Q6_K",
            "output.weight": "Q8_0",
            "blk.0.ffn_up.weight": "Q4_K",
            "blk.0.ffn_down.weight": "Q5_K",
        }
        candidate = {name: "BF16" for name in official}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "types.txt"
            with mock.patch(
                "gguf_recipe.load_tensor_type_map",
                side_effect=[official, candidate],
            ):
                recipe = write_dynamic_recipe(
                    official_dynamic=Path("official.gguf"),
                    candidate_bf16=Path("candidate.gguf"),
                    llama_cpp_dir=Path("llama.cpp"),
                    output_path=output,
                )
            lines = output.read_text(encoding="utf-8").splitlines()
        self.assertEqual(recipe["tensor_count"], 4)
        self.assertEqual(recipe["override_count"], 2)
        self.assertEqual(recipe["output_tensor_type"], "q8_0")
        self.assertEqual(recipe["token_embedding_type"], "q6_k")
        self.assertEqual(
            lines,
            [
                r"^blk\.0\.ffn_down\.weight$=q5_k",
                r"^blk\.0\.ffn_up\.weight$=q4_k",
            ],
        )

    def test_recipe_rejects_different_tensor_names(self):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch(
                "gguf_recipe.load_tensor_type_map",
                side_effect=[{"a": "Q4_K"}, {"b": "BF16"}],
            ):
                with self.assertRaisesRegex(RuntimeError, "tensor names differ"):
                    write_dynamic_recipe(
                        official_dynamic=Path("official.gguf"),
                        candidate_bf16=Path("candidate.gguf"),
                        llama_cpp_dir=Path("llama.cpp"),
                        output_path=Path(directory) / "types.txt",
                    )

    def test_llama_bench_uses_controlled_five_repeat_protocol(self):
        payload = [{"n_prompt": 512, "n_gen": 0}, {"n_prompt": 0, "n_gen": 128}]
        completed = mock.Mock(stdout=json.dumps(payload))
        with mock.patch("benchmark_gguf.subprocess.run", return_value=completed) as run:
            self.assertEqual(
                benchmark_model(Path("llama-bench"), Path("model.gguf"), 5),
                payload,
            )
        command = run.call_args.args[0]
        self.assertIn("512", command)
        self.assertIn("128", command)
        self.assertIn("5", command)
        self.assertEqual(command[-1], "json")

    def test_full_vocabulary_kl_reorders_server_token_ids(self):
        probabilities = torch.tensor([0.5, 0.3, 0.2])
        reference = probabilities.log()
        metrics = distribution_metrics(
            reference,
            [2, 0, 1],
            [math.log(0.2), math.log(0.5), math.log(0.3)],
        )
        self.assertAlmostEqual(metrics["token_kl"], 0.0, places=6)
        self.assertTrue(metrics["top_token_agreement"])

    def test_kl_percentile_reports_tail(self):
        self.assertEqual(percentile([0.0, 1.0, 2.0, 3.0, 4.0], 0.95), 3.8)


if __name__ == "__main__":
    unittest.main()
