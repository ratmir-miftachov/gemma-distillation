from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
from pathlib import Path
from typing import Any

from monarch_distill.monarch import densify_monarch_linears, is_monarch_linear
from quantize_hf import file_manifest, package_versions, resolve_hf_token


SOURCE_MODEL = "hexoy/gemma-4-e2b-monarch-35mlp"
SOURCE_REVISION = "f897353fca328b1cc5fd2e12d645773ca637f5f0"
DENSE_BASE_MODEL = "google/gemma-4-E2B-it"
DENSE_BASE_REVISION = "9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf"
EXPECTED_MONARCH_LINEAR_COUNT = 105
EXPECTED_DENSE_PARAMETER_COUNT = 5_104_297_504
REFERENCE_PROMPTS = (
    "The capital of France is",
    "Explain model compression in one sentence:",
    "A reliable scientific experiment should",
    "The image shows a red square. The dominant color is",
)
LEGAL_FILES = ("LICENSE", "NOTICE")


def expected_monarch_paths() -> set[str]:
    return {
        f"model.language_model.layers.{layer}.mlp.{projection}"
        for layer in range(35)
        for projection in ("gate_proj", "up_proj", "down_proj")
    }


def monarch_inventory(model: Any) -> list[str]:
    return sorted(name for name, module in model.named_modules() if is_monarch_linear(module))


def factor_tensor_names(model: Any) -> list[str]:
    return sorted(
        name
        for name in model.state_dict()
        if name.endswith(".blk1") or name.endswith(".blk2")
    )


def collect_last_token_logits(model: Any, tokenizer: Any, device: str) -> list[Any]:
    import torch

    results = []
    model.eval()
    with torch.inference_mode():
        for prompt in REFERENCE_PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            logits = model(**inputs, use_cache=False, return_dict=True).logits
            if not torch.isfinite(logits).all():
                raise RuntimeError("reference forward produced non-finite logits")
            results.append(logits[0, -1].float().cpu())
    return results


def compare_logits(reference: list[Any], candidate: list[Any]) -> dict[str, float]:
    import torch
    import torch.nn.functional as functional

    if len(reference) != len(candidate) or not reference:
        raise ValueError("logit comparison requires matching non-empty result lists")
    expected = torch.stack(reference)
    actual = torch.stack(candidate)
    difference = (actual - expected).abs()
    token_kl = functional.kl_div(
        functional.log_softmax(actual, dim=-1),
        functional.softmax(expected, dim=-1),
        reduction="batchmean",
    )
    metrics = {
        "mean_absolute_error": difference.mean().item(),
        "max_absolute_error": difference.max().item(),
        "mean_token_kl": token_kl.item(),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError(f"non-finite logit comparison: {metrics}")
    return metrics


def build_standard_config(source_config: Any) -> Any:
    from transformers import Gemma4Config

    config_dict = source_config.to_dict()
    for key in list(config_dict):
        if key.startswith("monarch_"):
            config_dict.pop(key)
    config_dict.pop("auto_map", None)
    config_dict.pop("model_type", None)
    config_dict["architectures"] = ["Gemma4ForConditionalGeneration"]
    standard = Gemma4Config.from_dict(config_dict)
    if any(key.startswith("monarch_") for key in standard.to_dict()):
        raise RuntimeError("standard configuration retained Monarch metadata")
    if getattr(standard, "auto_map", None):
        raise RuntimeError("standard configuration retained remote-code mappings")
    return standard


def copy_legal_files(output_dir: Path, token: str) -> None:
    from huggingface_hub import hf_hub_download

    for filename in LEGAL_FILES:
        source = Path(
            hf_hub_download(
                repo_id=SOURCE_MODEL,
                filename=filename,
                revision=SOURCE_REVISION,
                token=token,
            )
        )
        shutil.copy2(source, output_dir / filename)


def model_weight_files_size(output_dir: Path) -> int:
    return sum(path.stat().st_size for path in output_dir.glob("*.safetensors"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a custom Monarch Gemma 4 model as standard dense Gemma 4"
    )
    parser.add_argument("--model-id", default=SOURCE_MODEL)
    parser.add_argument("--revision", default=SOURCE_REVISION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--hf-token-file",
        type=Path,
        default=Path.home() / ".config/nebius-gemma/hf_read_token",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    token = resolve_hf_token(args.hf_token_file)

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("the production dense-equivalent export requires CUDA")

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        revision=args.revision,
        token=token,
        trust_remote_code=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        revision=args.revision,
        token=token,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        low_cpu_mem_usage=True,
    )
    model.eval()

    inventory = monarch_inventory(model)
    if len(inventory) != EXPECTED_MONARCH_LINEAR_COUNT or set(inventory) != expected_monarch_paths():
        raise RuntimeError(
            "source model does not contain exactly the expected 105 Monarch MLP projections: "
            f"count={len(inventory)}, difference={sorted(set(inventory) ^ expected_monarch_paths())[:10]}"
        )
    if len(factor_tensor_names(model)) != EXPECTED_MONARCH_LINEAR_COUNT * 2:
        raise RuntimeError("source model does not contain exactly 210 Monarch factors")

    source_logits = collect_last_token_logits(model, processor.tokenizer, args.device)
    replaced = densify_monarch_linears(model, dtype=torch.bfloat16)
    if set(replaced) != expected_monarch_paths():
        raise RuntimeError("densification replaced an unexpected module set")
    if monarch_inventory(model) or factor_tensor_names(model):
        raise RuntimeError("densified model retained Monarch modules or factors")

    dense_logits = collect_last_token_logits(model, processor.tokenizer, args.device)
    source_to_dense = compare_logits(source_logits, dense_logits)
    model.config = build_standard_config(model.config)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_DENSE_PARAMETER_COUNT:
        raise RuntimeError(
            f"unexpected dense parameter count: {parameter_count:,} != "
            f"{EXPECTED_DENSE_PARAMETER_COUNT:,}"
        )

    model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    processor.save_pretrained(args.output_dir)
    copy_legal_files(args.output_dir, token)

    restored = AutoModelForImageTextToText.from_pretrained(
        args.output_dir,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        low_cpu_mem_usage=True,
    )
    restored.eval()
    if monarch_inventory(restored) or factor_tensor_names(restored):
        raise RuntimeError("standard reload unexpectedly contains Monarch factors")
    restored_logits = collect_last_token_logits(
        restored, processor.tokenizer, args.device
    )
    dense_to_reload = compare_logits(dense_logits, restored_logits)
    source_to_reload = compare_logits(source_logits, restored_logits)

    manifest_path = args.output_dir / "dense_equivalent_manifest.json"
    manifest = {
        "schema_version": 1,
        "source_model": args.model_id,
        "source_revision": args.revision,
        "dense_base_model": DENSE_BASE_MODEL,
        "dense_base_revision": DENSE_BASE_REVISION,
        "replaced_monarch_linear_count": len(replaced),
        "replaced_module_names": sorted(replaced),
        "parameter_count": parameter_count,
        "serialized_weight_bytes": model_weight_files_size(args.output_dir),
        "source_to_dense_logits": source_to_dense,
        "dense_to_reloaded_logits": dense_to_reload,
        "source_to_reloaded_logits": source_to_reload,
        "standard_model_type": restored.config.model_type,
        "standard_architectures": restored.config.architectures,
        "trust_remote_code_required": False,
        "packages": package_versions(),
        "files": file_manifest(args.output_dir, exclude={manifest_path.name}),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)

    del model, restored, processor, source_logits, dense_logits, restored_logits
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
