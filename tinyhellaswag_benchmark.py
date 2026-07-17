from __future__ import annotations

import importlib.metadata
import hashlib
import json
import math
import os
import random
import re
import time
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


TASK_NAME = "tinyHellaswag"
NUM_EXAMPLES = 100
NUM_FEWSHOT = 10
DEFAULT_SEED = 1234
LM_EVAL_COMMIT = "97a5e2c710e2b56b9dd48f367bb6fe87bbb2c176"
TINYBENCHMARKS_COMMIT = "e9a8b1031b0340571beb6c9ca3a27891be09a8fd"
TINYBENCHMARKS_DATA_SHA256 = (
    "c3b6e426dfe7b100fe6d0ee960398e10a8763254bcead3be80cc6bc15abca284"
)
DEFAULT_TOKEN_FILE = Path.home() / ".config/nebius-gemma/hf_read_token"
TINYBENCHMARKS_CACHE = Path.home() / ".cache/gemma-distillation/tinybenchmarks"


@dataclass(frozen=True)
class LoadedModel:
    model: Any
    tokenizer: Any
    config: Any
    loader_class: str


def resolve_hf_token(token_file: Path = DEFAULT_TOKEN_FILE) -> str | None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    if token_file.is_file():
        token = token_file.read_text(encoding="utf-8").strip()
        return token or None
    return None


def select_model_loader(config: Any) -> str:
    auto_map = getattr(config, "auto_map", None) or {}
    if "AutoModelForImageTextToText" in auto_map:
        return "AutoModelForImageTextToText"
    if getattr(config, "is_encoder_decoder", False):
        raise ValueError("Encoder-decoder models are outside this benchmark's scope")
    return "AutoModelForCausalLM"


def _torch_dtype(dtype_name: str, torch_module: Any) -> Any:
    names = {
        "bfloat16": torch_module.bfloat16,
        "float16": torch_module.float16,
        "float32": torch_module.float32,
    }
    try:
        return names[dtype_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype_name}") from exc


def load_model(
    model_id: str,
    *,
    revision: str,
    dtype_name: str,
    device: str,
    token: str | None,
    transformers_module: Any | None = None,
    torch_module: Any | None = None,
) -> LoadedModel:
    if transformers_module is None:
        import transformers as transformers_module
    if torch_module is None:
        import torch as torch_module

    if device.startswith("cuda") and not torch_module.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device}")

    common_kwargs = {
        "revision": revision,
        "token": token,
        "trust_remote_code": True,
    }
    config = transformers_module.AutoConfig.from_pretrained(model_id, **common_kwargs)
    loader_name = select_model_loader(config)
    loader = getattr(transformers_module, loader_name, None)
    if loader is None:
        raise RuntimeError(
            f"Installed transformers does not provide {loader_name}; upgrade transformers"
        )

    model_kwargs = {
        **common_kwargs,
        "low_cpu_mem_usage": True,
    }
    transformers_major = int(
        str(getattr(transformers_module, "__version__", "4")).split(".", 1)[0]
    )
    dtype_key = "dtype" if transformers_major >= 5 else "torch_dtype"
    model_kwargs[dtype_key] = _torch_dtype(dtype_name, torch_module)
    if device.startswith("cuda"):
        model_kwargs["device_map"] = {"": device}

    # The selected loader is deliberate. Authentication and download errors must
    # propagate instead of being mistaken for a model-class mismatch.
    model = loader.from_pretrained(model_id, **model_kwargs)
    if not device.startswith("cuda"):
        model.to(device)
    model.eval()
    tokenizer = transformers_module.AutoTokenizer.from_pretrained(
        model_id, **common_kwargs
    )
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        config=config,
        loader_class=loader_name,
    )


def build_lm_eval_model(
    loaded: LoadedModel,
    *,
    batch_size: str | int,
    max_batch_size: int,
    hflm_class: Any | None = None,
) -> Any:
    if hflm_class is None:
        from lm_eval.models.huggingface import HFLM as hflm_class

    return hflm_class(
        pretrained=loaded.model,
        tokenizer=loaded.tokenizer,
        backend="causal",
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        trust_remote_code=True,
    )


def _as_float(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def _metric_value(task_results: Mapping[str, Any], metric: str) -> float:
    matches = [key for key in task_results if key.split(",", 1)[0] == metric]
    if len(matches) != 1:
        raise ValueError(f"Expected one {metric} metric, found {matches}")
    return _as_float(task_results[matches[0]])


def _gold_index(sample: Mapping[str, Any], choices: Sequence[str]) -> int:
    gold = sample.get("target")
    if isinstance(gold, str):
        if gold in choices:
            return choices.index(gold)
        try:
            return int(gold)
        except ValueError as exc:
            raise ValueError(f"Unrecognized gold choice: {gold!r}") from exc
    return int(gold)


def extract_items(samples: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for sample in sorted(samples, key=lambda row: int(row["doc_id"])):
        choices = list(sample["doc"]["choices"])
        responses = sample["filtered_resps"]
        if len(choices) != 4 or len(responses) != 4:
            raise ValueError(
                f"Expected four choices for doc {sample['doc_id']}, got "
                f"{len(choices)} choices and {len(responses)} responses"
            )

        raw_scores = [_as_float(response[0]) for response in responses]
        lengths = [len(choice) for choice in choices]
        if any(length == 0 for length in lengths):
            raise ValueError(f"Empty choice for doc {sample['doc_id']}")
        normalized_scores = [
            score / length for score, length in zip(raw_scores, lengths)
        ]
        selected = max(range(4), key=normalized_scores.__getitem__)
        gold = _gold_index(sample, choices)
        correct = selected == gold

        harness_correct = bool(_as_float(sample["acc_norm"]))
        if harness_correct != correct:
            raise ValueError(
                f"Reconstructed decision disagrees with lm-eval for doc {sample['doc_id']}"
            )

        items.append(
            {
                "doc_id": int(sample["doc_id"]),
                "query": sample["doc"].get("query"),
                "choices": choices,
                "choice_character_lengths": lengths,
                "choice_log_likelihoods": raw_scores,
                "choice_normalized_scores": normalized_scores,
                "selected_choice": selected,
                "gold_choice": gold,
                "correct": correct,
                "doc_hash": sample.get("doc_hash"),
                "prompt_hash": sample.get("prompt_hash"),
            }
        )
    return items


def dependency_versions() -> dict[str, str]:
    versions = {
        "lm_eval_commit": LM_EVAL_COMMIT,
        "tinyBenchmarks_commit": TINYBENCHMARKS_COMMIT,
        "tinyBenchmarks_data_sha256": TINYBENCHMARKS_DATA_SHA256,
    }
    for package in (
        "lm_eval",
        "tinyBenchmarks",
        "torch",
        "torchao",
        "transformers",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def model_metadata(
    model_id: str,
    revision: str,
    loaded: LoadedModel,
) -> dict[str, Any]:
    model = loaded.model
    quantization_config = getattr(loaded.config, "quantization_config", None)
    if hasattr(quantization_config, "to_dict"):
        quantization_config = quantization_config.to_dict()
    return {
        "requested_id": model_id,
        "requested_revision": revision,
        "resolved_revision": getattr(loaded.config, "_commit_hash", None) or revision,
        "config_class": type(loaded.config).__name__,
        "model_class": type(model).__name__,
        "loader_class": loaded.loader_class,
        "architectures": list(getattr(loaded.config, "architectures", None) or []),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "loaded_model_footprint_bytes": int(
            model.get_memory_footprint(return_buffers=True)
        )
        if hasattr(model, "get_memory_footprint")
        else None,
        "quantization_config": quantization_config,
    }


def runtime_metadata(
    *,
    torch_module: Any,
    seed: int,
    dtype_name: str,
    device: str,
    batch_size: str | int,
    max_batch_size: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    cuda = device.startswith("cuda") and torch_module.cuda.is_available()
    return {
        "seed": seed,
        "dtype": dtype_name,
        "device": device,
        "batch_size": batch_size,
        "max_batch_size": max_batch_size,
        "gpu": torch_module.cuda.get_device_name(device) if cuda else None,
        "peak_memory_bytes": torch_module.cuda.max_memory_allocated(device)
        if cuda
        else 0,
        "elapsed_seconds": elapsed_seconds,
    }


def build_result(
    *,
    evaluation: Mapping[str, Any],
    model_info: Mapping[str, Any],
    runtime_info: Mapping[str, Any],
) -> dict[str, Any]:
    task_results = evaluation["results"][TASK_NAME]
    items = extract_items(evaluation["samples"][TASK_NAME])
    if len(items) != NUM_EXAMPLES:
        raise ValueError(f"Expected {NUM_EXAMPLES} examples, got {len(items)}")
    raw_accuracy = sum(item["correct"] for item in items) / len(items)
    return {
        "schema_version": 1,
        "benchmark": {
            "task": TASK_NAME,
            "num_examples": NUM_EXAMPLES,
            "num_fewshot": NUM_FEWSHOT,
            "scoring": "character-length-normalized continuation log likelihood",
            "chat_template_applied": False,
            "official_gp_irt_accuracy": _metric_value(task_results, "acc_norm"),
            "raw_anchor_accuracy": raw_accuracy,
        },
        "model": dict(model_info),
        "runtime": dict(runtime_info),
        "dependencies": dependency_versions(),
        "items": items,
    }


def json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if callable(value):
        module = getattr(value, "__module__", type(value).__module__)
        name = getattr(value, "__qualname__", type(value).__qualname__)
        return f"{module}.{name}"
    if type(value).__module__.startswith("torch"):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=False, default=json_default
        )
        + "\n",
        encoding="utf-8",
    )


def model_slug(model_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-.")
    return slug or "model"


def default_output_dir(model_id: str, now: datetime | None = None) -> Path:
    now = now or datetime.now(timezone.utc)
    timestamp = now.strftime("%Y%m%dT%H%M%S.%fZ")
    return Path("benchmark_results/tinyhellaswag") / model_slug(model_id) / timestamp


@contextmanager
def working_directory(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_tinybenchmarks_data(
    cache_dir: Path = TINYBENCHMARKS_CACHE,
    *,
    expected_sha256: str = TINYBENCHMARKS_DATA_SHA256,
    url: str | None = None,
) -> Path:
    destination = cache_dir / "tinyBenchmarks.pkl"
    if destination.is_file() and _file_sha256(destination) == expected_sha256:
        return destination

    url = url or (
        "https://raw.githubusercontent.com/felipemaiapolo/tinyBenchmarks/"
        f"{TINYBENCHMARKS_COMMIT}/tinyBenchmarks/tinyBenchmarks.pkl"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_dir / f"tinyBenchmarks.pkl.tmp-{os.getpid()}"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            temporary.write_bytes(response.read())
        actual_sha256 = _file_sha256(temporary)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "Pinned tinyBenchmarks data hash mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def run_benchmark(
    *,
    model_id: str,
    revision: str = "main",
    dtype_name: str = "bfloat16",
    device: str = "cuda",
    batch_size: str | int = "auto",
    max_batch_size: int = 64,
    seed: int = DEFAULT_SEED,
    token_file: Path = DEFAULT_TOKEN_FILE,
    output_dir: Path | None = None,
) -> Path:
    import torch
    from lm_eval import evaluator

    output_dir = output_dir or default_output_dir(model_id)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    token = resolve_hf_token(token_file)
    loaded = load_model(
        model_id,
        revision=revision,
        dtype_name=dtype_name,
        device=device,
        token=token,
    )
    lm_model = build_lm_eval_model(
        loaded,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
    )

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    ensure_tinybenchmarks_data()
    with working_directory(TINYBENCHMARKS_CACHE):
        evaluation = evaluator.simple_evaluate(
            model=lm_model,
            tasks=[TASK_NAME],
            num_fewshot=NUM_FEWSHOT,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            device=device,
            bootstrap_iters=0,
            log_samples=True,
            apply_chat_template=False,
            fewshot_as_multiturn=False,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
            fewshot_random_seed=seed,
        )
    if evaluation is None:
        raise RuntimeError("lm-eval returned no results on the primary process")
    elapsed = time.perf_counter() - started

    runtime_info = runtime_metadata(
        torch_module=torch,
        seed=seed,
        dtype_name=dtype_name,
        device=device,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        elapsed_seconds=elapsed,
    )
    runtime_info["detected_batch_sizes"] = evaluation.get("config", {}).get(
        "batch_sizes", []
    )
    result = build_result(
        evaluation=evaluation,
        model_info=model_metadata(model_id, revision, loaded),
        runtime_info=runtime_info,
    )
    write_json(output_dir / "result.json", result)
    write_json(output_dir / "lm_eval_results.json", evaluation)
    return output_dir / "result.json"


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_bootstrap_interval(
    baseline: Sequence[bool],
    candidate: Sequence[bool],
    *,
    iterations: int = 10_000,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float]:
    if len(baseline) != len(candidate) or not baseline:
        raise ValueError("Paired bootstrap requires non-empty, equally sized inputs")
    rng = random.Random(seed)
    size = len(baseline)
    deltas = []
    for _ in range(iterations):
        total = 0
        for _ in range(size):
            index = rng.randrange(size)
            total += int(candidate[index]) - int(baseline[index])
        deltas.append(total / size)
    return _percentile(deltas, 0.025), _percentile(deltas, 0.975)


def exact_mcnemar_p_value(baseline_only: int, candidate_only: int) -> float:
    discordant = baseline_only + candidate_only
    if discordant == 0:
        return 1.0
    tail = min(baseline_only, candidate_only)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (
        2**discordant
    )
    return min(1.0, 2 * probability)


def compare_results(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    bootstrap_iterations: int = 10_000,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    for key in (
        "task",
        "num_examples",
        "num_fewshot",
        "scoring",
        "chat_template_applied",
    ):
        if baseline["benchmark"][key] != candidate["benchmark"][key]:
            raise ValueError(f"Benchmark mismatch for {key}")
    if baseline["runtime"]["seed"] != candidate["runtime"]["seed"]:
        raise ValueError("Benchmark seed mismatch")

    baseline_items = {item["doc_id"]: item for item in baseline["items"]}
    candidate_items = {item["doc_id"]: item for item in candidate["items"]}
    if baseline_items.keys() != candidate_items.keys():
        raise ValueError("Result files contain different document IDs")

    baseline_correct = []
    candidate_correct = []
    disagreements = []
    for doc_id in sorted(baseline_items):
        old = baseline_items[doc_id]
        new = candidate_items[doc_id]
        if old["gold_choice"] != new["gold_choice"]:
            raise ValueError(f"Gold label mismatch for doc {doc_id}")
        for hash_name in ("doc_hash", "prompt_hash"):
            if old.get(hash_name) != new.get(hash_name):
                raise ValueError(f"{hash_name} mismatch for doc {doc_id}")
        baseline_correct.append(bool(old["correct"]))
        candidate_correct.append(bool(new["correct"]))
        if old["selected_choice"] != new["selected_choice"]:
            disagreements.append(
                {
                    "doc_id": doc_id,
                    "gold_choice": old["gold_choice"],
                    "baseline_choice": old["selected_choice"],
                    "candidate_choice": new["selected_choice"],
                    "baseline_correct": bool(old["correct"]),
                    "candidate_correct": bool(new["correct"]),
                }
            )

    baseline_only = sum(
        old and not new for old, new in zip(baseline_correct, candidate_correct)
    )
    candidate_only = sum(
        new and not old for old, new in zip(baseline_correct, candidate_correct)
    )
    ci_low, ci_high = paired_bootstrap_interval(
        baseline_correct,
        candidate_correct,
        iterations=bootstrap_iterations,
        seed=seed,
    )
    baseline_raw = baseline["benchmark"]["raw_anchor_accuracy"]
    candidate_raw = candidate["benchmark"]["raw_anchor_accuracy"]
    baseline_gpirt = baseline["benchmark"]["official_gp_irt_accuracy"]
    candidate_gpirt = candidate["benchmark"]["official_gp_irt_accuracy"]
    return {
        "schema_version": 1,
        "task": baseline["benchmark"]["task"],
        "baseline_model": baseline["model"]["requested_id"],
        "candidate_model": candidate["model"]["requested_id"],
        "official_gp_irt": {
            "baseline": baseline_gpirt,
            "candidate": candidate_gpirt,
            "delta": candidate_gpirt - baseline_gpirt,
        },
        "raw_anchor_accuracy": {
            "baseline": baseline_raw,
            "candidate": candidate_raw,
            "delta": candidate_raw - baseline_raw,
            "paired_bootstrap_95_percent_ci": [ci_low, ci_high],
        },
        "mcnemar_exact": {
            "baseline_only_correct": baseline_only,
            "candidate_only_correct": candidate_only,
            "p_value": exact_mcnemar_p_value(baseline_only, candidate_only),
        },
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
        "bootstrap_iterations": bootstrap_iterations,
        "seed": seed,
    }


def load_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
