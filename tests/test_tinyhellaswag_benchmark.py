import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from tinyhellaswag_benchmark import (
    LoadedModel,
    build_lm_eval_model,
    build_prompt_bundle,
    compare_results,
    default_output_dir,
    ensure_tinybenchmarks_data,
    exact_mcnemar_p_value,
    extract_items,
    load_model,
    resolve_hf_token,
    select_model_loader,
    validate_prompt_bundle,
    working_directory,
    write_json,
)


class FakeModel:
    def __init__(self):
        self.moved_to = None
        self.eval_called = False

    def to(self, device):
        self.moved_to = device
        return self

    def eval(self):
        self.eval_called = True
        return self


class FakeAutoClass:
    calls = []
    result = FakeModel()
    error = None

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        cls.calls.append((model_id, kwargs))
        if cls.error is not None:
            raise cls.error
        return cls.result


class FakeTokenizerClass:
    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        return SimpleNamespace(model_id=model_id, kwargs=kwargs)


class FakeCuda:
    @staticmethod
    def is_available():
        return False


class FakeTorch:
    bfloat16 = "bf16"
    float16 = "fp16"
    float32 = "fp32"
    cuda = FakeCuda()


class TinyHellaSwagBenchmarkTest(unittest.TestCase):
    def tearDown(self):
        FakeAutoClass.calls = []
        FakeAutoClass.result = FakeModel()
        FakeAutoClass.error = None

    def test_model_loader_detection(self):
        dense = SimpleNamespace(auto_map=None, is_encoder_decoder=False)
        monarch = SimpleNamespace(
            auto_map={
                "AutoConfig": "configuration.MonarchConfig",
                "AutoModelForImageTextToText": "modeling.MonarchModel",
            },
            is_encoder_decoder=False,
        )
        self.assertEqual(select_model_loader(dense), "AutoModelForCausalLM")
        self.assertEqual(select_model_loader(monarch), "AutoModelForImageTextToText")

    def test_encoder_decoder_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Encoder-decoder"):
            select_model_loader(SimpleNamespace(auto_map=None, is_encoder_decoder=True))

    def test_selected_loader_error_is_not_swallowed(self):
        config = SimpleNamespace(
            auto_map={"AutoModelForImageTextToText": "modeling.MonarchModel"},
            is_encoder_decoder=False,
        )
        config_loader = SimpleNamespace(from_pretrained=lambda *args, **kwargs: config)
        image_loader = type("ImageLoader", (FakeAutoClass,), {})
        causal_loader = type("CausalLoader", (FakeAutoClass,), {})
        image_loader.calls = []
        causal_loader.calls = []
        image_loader.error = OSError("private repository denied")
        transformers = SimpleNamespace(
            AutoConfig=config_loader,
            AutoModelForImageTextToText=image_loader,
            AutoModelForCausalLM=causal_loader,
            AutoTokenizer=FakeTokenizerClass,
        )

        with self.assertRaisesRegex(OSError, "private repository denied"):
            load_model(
                "private/model",
                revision="main",
                dtype_name="float32",
                device="cpu",
                token="secret",
                transformers_module=transformers,
                torch_module=FakeTorch,
            )
        self.assertEqual(len(image_loader.calls), 1)
        self.assertEqual(causal_loader.calls, [])

    def test_transformers_five_uses_dtype_keyword(self):
        config = SimpleNamespace(auto_map=None, is_encoder_decoder=False)
        config_loader = SimpleNamespace(from_pretrained=lambda *args, **kwargs: config)
        causal_loader = type("CausalLoader", (FakeAutoClass,), {})
        causal_loader.calls = []
        causal_loader.error = None
        causal_loader.result = FakeModel()
        transformers = SimpleNamespace(
            __version__="5.12.1",
            AutoConfig=config_loader,
            AutoModelForCausalLM=causal_loader,
            AutoTokenizer=FakeTokenizerClass,
        )

        loaded = load_model(
            "public/model",
            revision="main",
            dtype_name="bfloat16",
            device="cpu",
            token=None,
            transformers_module=transformers,
            torch_module=FakeTorch,
        )
        kwargs = causal_loader.calls[0][1]
        self.assertEqual(kwargs["dtype"], "bf16")
        self.assertNotIn("torch_dtype", kwargs)
        self.assertEqual(loaded.model.moved_to, "cpu")

    def test_dense_and_monarch_models_share_causal_lm_eval_adapter(self):
        calls = []

        class FakeHFLM:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        dense = LoadedModel(FakeModel(), object(), object(), "AutoModelForCausalLM")
        monarch = LoadedModel(
            FakeModel(), object(), object(), "AutoModelForImageTextToText"
        )
        build_lm_eval_model(
            dense, batch_size="auto", max_batch_size=32, hflm_class=FakeHFLM
        )
        build_lm_eval_model(
            monarch, batch_size="auto", max_batch_size=32, hflm_class=FakeHFLM
        )

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["backend"], "causal")
        self.assertEqual(calls[1]["backend"], "causal")
        self.assertEqual(calls[0]["batch_size"], calls[1]["batch_size"])

    def test_prompt_bundle_pins_strings_hashes_labels_and_token_ids(self):
        samples = []
        for doc_id in range(100):
            prompt = f"few-shot prompt {doc_id}:"
            continuations = [" a", " b", " c", " d"]
            samples.append(
                {
                    "doc_id": doc_id,
                    "doc_hash": f"doc-{doc_id}",
                    "prompt_hash": f"prompt-{doc_id}",
                    "target": doc_id % 4,
                    "doc": {"choices": continuations},
                    "arguments": [[prompt, value] for value in continuations],
                }
            )
        tokenize = lambda text: list(text.encode("utf-8"))
        bundle = build_prompt_bundle(samples, tokenize=tokenize, seed=1234)
        self.assertEqual(len(bundle["records"]), 100)
        self.assertEqual(bundle["records"][0]["prompt_token_ids"], tokenize("few-shot prompt 0:"))
        self.assertEqual(
            bundle["records"][0]["request_token_ids"][0],
            tokenize("few-shot prompt 0: a"),
        )
        validate_prompt_bundle(bundle, json.loads(json.dumps(bundle)))
        changed = json.loads(json.dumps(bundle))
        changed["records"][0]["prompt_token_ids"][0] += 1
        with self.assertRaisesRegex(ValueError, "first differing"):
            validate_prompt_bundle(changed, bundle)

    def test_item_scores_match_official_character_normalization(self):
        sample = {
            "doc_id": 7,
            "doc": {
                "query": "context",
                "choices": ["a", "bbbb", "cc", "ddd"],
            },
            "target": 1,
            "filtered_resps": [
                (-1.0, False),
                (-2.0, False),
                (-2.0, False),
                (-3.0, False),
            ],
            "acc_norm": 1.0,
            "doc_hash": "doc",
            "prompt_hash": "prompt",
        }
        item = extract_items([sample])[0]
        self.assertEqual(item["choice_normalized_scores"], [-1.0, -0.5, -1.0, -1.0])
        self.assertEqual(item["selected_choice"], 1)
        self.assertTrue(item["correct"])

    def test_token_resolution_prefers_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "token"
            path.write_text("file-token\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"HF_TOKEN": "env-token"}):
                self.assertEqual(resolve_hf_token(path), "env-token")
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertEqual(resolve_hf_token(path), "file-token")

    def test_default_output_path_is_model_and_timestamp_scoped(self):
        from datetime import datetime, timezone

        path = default_output_dir(
            "owner/private model",
            datetime(2026, 7, 13, 10, 20, 30, 123456, tzinfo=timezone.utc),
        )
        self.assertEqual(
            str(path),
            "benchmark_results/tinyhellaswag/owner-private-model/"
            "20260713T102030.123456Z",
        )

    def test_working_directory_is_restored(self):
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "cache"
            with working_directory(target):
                self.assertEqual(Path.cwd(), target.resolve())
            self.assertEqual(Path.cwd(), original)

    def test_tinybenchmarks_data_is_downloaded_and_hash_verified(self):
        payload = b"pinned tinyBenchmarks data"
        expected_hash = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory)
            with mock.patch(
                "tinyhellaswag_benchmark.urllib.request.urlopen",
                return_value=io.BytesIO(payload),
            ) as urlopen:
                path = ensure_tinybenchmarks_data(
                    cache_dir,
                    expected_sha256=expected_hash,
                    url="https://example.test/tinyBenchmarks.pkl",
                )
            self.assertEqual(path.read_bytes(), payload)
            urlopen.assert_called_once()

            with mock.patch(
                "tinyhellaswag_benchmark.urllib.request.urlopen",
                side_effect=AssertionError("cache should avoid another download"),
            ):
                self.assertEqual(
                    ensure_tinybenchmarks_data(
                        cache_dir,
                        expected_sha256=expected_hash,
                    ),
                    path,
                )

    def test_json_serialization_is_deterministic(self):
        payload = {
            "z": 1,
            "a": {"value": True},
            "dtype": torch.float32,
            "callable": select_model_loader,
        }
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.json"
            second = Path(directory) / "second.json"
            write_json(first, payload)
            write_json(second, payload)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            restored = json.loads(first.read_text())
            self.assertEqual(restored["a"], payload["a"])
            self.assertEqual(restored["dtype"], "torch.float32")
            self.assertTrue(restored["callable"].endswith(".select_model_loader"))

    def test_paired_comparison_and_mcnemar(self):
        baseline = self._synthetic_result(
            "baseline", [True, True, False, False], [0, 0, 1, 1], 0.5
        )
        candidate = self._synthetic_result(
            "candidate", [True, False, True, False], [0, 1, 0, 1], 0.6
        )
        comparison = compare_results(
            baseline, candidate, bootstrap_iterations=200, seed=1234
        )
        self.assertEqual(comparison["disagreement_count"], 2)
        self.assertEqual(comparison["mcnemar_exact"]["baseline_only_correct"], 1)
        self.assertEqual(comparison["mcnemar_exact"]["candidate_only_correct"], 1)
        self.assertEqual(comparison["mcnemar_exact"]["p_value"], 1.0)
        self.assertAlmostEqual(comparison["official_gp_irt"]["delta"], 0.1)
        self.assertEqual(exact_mcnemar_p_value(0, 0), 1.0)

    @staticmethod
    def _synthetic_result(model, correctness, selected, gpirt):
        items = [
            {
                "doc_id": index,
                "gold_choice": 0,
                "selected_choice": choice,
                "correct": is_correct,
            }
            for index, (is_correct, choice) in enumerate(zip(correctness, selected))
        ]
        return {
            "benchmark": {
                "task": "tinyHellaswag",
                "num_examples": len(items),
                "num_fewshot": 10,
                "scoring": "character-length-normalized continuation log likelihood",
                "chat_template_applied": False,
                "raw_anchor_accuracy": sum(correctness) / len(correctness),
                "official_gp_irt_accuracy": gpirt,
                "prompt_bundle_sha256": "canonical-bundle",
            },
            "model": {"requested_id": model},
            "runtime": {"seed": 1234},
            "items": items,
        }


if __name__ == "__main__":
    unittest.main()
