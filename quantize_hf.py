from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path
from typing import Any

from storage_utils import is_torchao_tensor, model_storage_bytes


SOURCE_MODEL = "hexoy/gemma-4-e2b-monarch-35mlp"
SOURCE_REVISION = "f897353fca328b1cc5fd2e12d645773ca637f5f0"
EXPECTED_PARAMETER_COUNT = 3_682_268_704
EXPECTED_MONARCH_FACTOR_COUNT = 210
TORCHAO_INT8_CONFIG_VERSION = 2
REMOTE_CODE_FILES = (
    "configuration_monarch_gemma4.py",
    "modeling_monarch_gemma4.py",
    "monarch.py",
)
LEGAL_FILES = ("LICENSE", "NOTICE")


def resolve_hf_token(token_file: Path) -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and token_file.is_file():
        token = token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is unset and the Hugging Face token file is empty")
    return token


def select_model_loader(config: Any) -> str:
    auto_map = getattr(config, "auto_map", None) or {}
    if "AutoModelForImageTextToText" in auto_map:
        return "AutoModelForImageTextToText"
    if getattr(config, "is_encoder_decoder", False):
        raise ValueError("encoder-decoder models are outside this quantizer's scope")
    return "AutoModelForCausalLM"


def is_monarch_factor_name(name: str) -> bool:
    return name.endswith(".blk1") or name.endswith(".blk2")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(root: Path, *, exclude: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = exclude or set()
    records = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        records.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def package_versions() -> dict[str, str]:
    versions = {}
    for package in ("torch", "torchao", "transformers", "accelerate", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def audit_quantized_model(model: Any, torch_module: Any) -> dict[str, Any]:
    embedding_weight_ids = {
        id(module.weight)
        for module in model.modules()
        if isinstance(module, torch_module.nn.Embedding)
    }
    linear_names = []
    unquantized_linear_names = []
    tied_embedding_linear_names = []
    quantized_weight_numel = 0
    for name, module in model.named_modules():
        if not isinstance(module, torch_module.nn.Linear):
            continue
        linear_names.append(name)
        if is_torchao_tensor(module.weight):
            quantized_weight_numel += module.weight.numel()
        elif id(module.weight) in embedding_weight_ids:
            tied_embedding_linear_names.append(name)
        else:
            unquantized_linear_names.append(name)

    monarch_factors = {
        name: parameter
        for name, parameter in model.named_parameters()
        if is_monarch_factor_name(name)
    }
    bad_monarch_dtypes = {
        name: str(parameter.dtype)
        for name, parameter in monarch_factors.items()
        if parameter.dtype != torch_module.bfloat16
    }
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    dtype_numel: dict[str, int] = {}
    for parameter in model.parameters():
        key = "torchao_quantized" if is_torchao_tensor(parameter) else str(parameter.dtype)
        dtype_numel[key] = dtype_numel.get(key, 0) + parameter.numel()

    if not linear_names:
        raise RuntimeError("the loaded model contains no standard nn.Linear modules")
    if unquantized_linear_names:
        raise RuntimeError(
            "standard linear weights escaped INT8 quantization: "
            + ", ".join(unquantized_linear_names[:20])
        )
    if len(monarch_factors) != EXPECTED_MONARCH_FACTOR_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_MONARCH_FACTOR_COUNT} Monarch factors, "
            f"found {len(monarch_factors)}"
        )
    if bad_monarch_dtypes:
        raise RuntimeError(f"Monarch factors are not BF16: {bad_monarch_dtypes}")
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"parameter count changed: expected {EXPECTED_PARAMETER_COUNT}, got {parameter_count}"
        )

    physical_footprint = model_storage_bytes(model, include_buffers=True)
    logical_footprint = int(model.get_memory_footprint(return_buffers=True))
    return {
        "parameter_count": parameter_count,
        "standard_linear_count": len(linear_names),
        "standard_linear_names": linear_names,
        "tied_embedding_linear_count": len(tied_embedding_linear_names),
        "tied_embedding_linear_names": tied_embedding_linear_names,
        "quantized_linear_weight_numel": quantized_weight_numel,
        "monarch_factor_count": len(monarch_factors),
        "monarch_factor_numel": sum(value.numel() for value in monarch_factors.values()),
        "parameter_numel_by_storage_kind": dtype_numel,
        "loaded_model_footprint_bytes": physical_footprint,
        "logical_model_footprint_bytes": logical_footprint,
    }


def copy_source_support_files(
    *, source_model: str, revision: str, output_dir: Path, token: str
) -> None:
    from huggingface_hub import hf_hub_download

    for filename in (*REMOTE_CODE_FILES, *LEGAL_FILES):
        source = Path(
            hf_hub_download(
                repo_id=source_model,
                filename=filename,
                revision=revision,
                token=token,
            )
        )
        shutil.copy2(source, output_dir / filename)


def write_provisional_model_card(
    output_dir: Path,
    *,
    repo_id: str,
    source_model: str,
    source_revision: str,
    audit: dict[str, Any],
) -> None:
    footprint_gib = audit["loaded_model_footprint_bytes"] / 2**30
    text = f"""---
license: apache-2.0
base_model: {source_model}
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- gemma4
- monarch-matrices
- model-compression
- int8
- torchao
---

# Gemma 4 E2B Monarch 35-MLP INT8

Private experimental INT8 weight-only version of
[`{source_model}`](https://huggingface.co/{source_model}) at immutable revision
`{source_revision}`.

All {audit['standard_linear_count']} remaining standard `nn.Linear` weights use
TorchAO symmetric per-channel INT8 weight-only quantization. The 210 trained
Monarch factors, tied embeddings, normalization parameters, biases, and runtime
activations remain BF16. The parameter count is unchanged at
{audit['parameter_count']:,}; the measured loaded model footprint is
{footprint_gib:.2f} GiB.

## Usage

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "{repo_id}"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype="auto",
    device_map="auto",
)
```

Install the pinned TorchAO dependency before loading this model. This repository
contains custom modeling code, so review it before enabling
`trust_remote_code=True`.

## Limitations

- INT8 applies to weights only; activations and logits remain BF16.
- Peak GPU memory can therefore be dominated by workload-dependent activations.
- Benchmark results will be added after the clean-cache verification run.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def load_quantized_model(args: argparse.Namespace, token: str):
    import torch
    import transformers
    from torchao.quantization import Int8WeightOnlyConfig
    from transformers import TorchAoConfig

    common = {
        "revision": args.revision,
        "token": token,
        "trust_remote_code": True,
    }
    config = transformers.AutoConfig.from_pretrained(args.model_id, **common)
    loader_name = select_model_loader(config)
    loader = getattr(transformers, loader_name)
    quantization_config = TorchAoConfig(
        quant_type=Int8WeightOnlyConfig(
            group_size=None,
            version=TORCHAO_INT8_CONFIG_VERSION,
        )
    )
    model = loader.from_pretrained(
        args.model_id,
        **common,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
    )
    model.eval()
    return model


def verify_local_reload(output_dir: Path, token: str, device: str) -> dict[str, Any]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        output_dir, trust_remote_code=True, token=token
    )
    model = AutoModelForImageTextToText.from_pretrained(
        output_dir,
        trust_remote_code=True,
        token=token,
        dtype=torch.bfloat16,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    model.eval()
    audit = audit_quantized_model(model, torch)

    inputs = processor.tokenizer(
        "Give one short reason to compress a language model.", return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False, return_dict=True)
        generated = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    if not torch.isfinite(outputs.logits).all():
        raise RuntimeError("local INT8 reload produced non-finite logits")
    text = processor.tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"[Verify] INT8 text generation: {text}", flush=True)
    del outputs, generated, inputs, model
    gc.collect()
    torch.cuda.empty_cache()
    return audit


def upload_private(output_dir: Path, repo_id: str, token: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    if not api.model_info(repo_id).private:
        raise RuntimeError(f"refusing to upload because {repo_id} is not private")
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=output_dir,
        commit_message="Add per-channel INT8 weight-only model",
    )
    if not api.model_info(repo_id).private:
        raise RuntimeError(f"repository {repo_id} unexpectedly became public")
    return commit.oid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize the standard linear weights of a Hugging Face model to INT8"
    )
    parser.add_argument("--model-id", default=SOURCE_MODEL)
    parser.add_argument("--revision", default=SOURCE_REVISION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--repo-id", default="hexoy/gemma-4-e2b-monarch-35mlp-int8"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--hf-token-file",
        type=Path,
        default=Path.home() / ".config/nebius-gemma/hf_read_token",
    )
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    token = resolve_hf_token(args.hf_token_file)

    import torch
    from transformers import AutoProcessor

    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("the production INT8 export requires a CUDA device")

    model = load_quantized_model(args, token)
    audit = audit_quantized_model(model, torch)
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        revision=args.revision,
        token=token,
        trust_remote_code=True,
    )
    model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    processor.save_pretrained(args.output_dir)
    copy_source_support_files(
        source_model=args.model_id,
        revision=args.revision,
        output_dir=args.output_dir,
        token=token,
    )
    write_provisional_model_card(
        args.output_dir,
        repo_id=args.repo_id,
        source_model=args.model_id,
        source_revision=args.revision,
        audit=audit,
    )
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()

    reloaded_audit = verify_local_reload(args.output_dir, token, args.device)
    if reloaded_audit != audit:
        raise RuntimeError("the reloaded INT8 model inventory differs from the export")

    manifest_path = args.output_dir / "int8_manifest.json"
    manifest = {
        "schema_version": 1,
        "source_model": args.model_id,
        "source_revision": args.revision,
        "repository": args.repo_id,
        "quantization": {
            "backend": "torchao",
            "method": "Int8WeightOnlyConfig",
            "config_version": TORCHAO_INT8_CONFIG_VERSION,
            "granularity": "symmetric_per_output_channel",
            "activation_dtype": "torch.bfloat16",
            "monarch_factor_dtype": "torch.bfloat16",
        },
        "audit": audit,
        "packages": package_versions(),
        "files": file_manifest(args.output_dir, exclude={manifest_path.name}),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"audit": audit, "manifest": str(manifest_path)}, indent=2))

    if args.upload:
        commit = upload_private(args.output_dir, args.repo_id, token)
        print(f"[Upload] Private model commit: {commit}", flush=True)


if __name__ == "__main__":
    main()
