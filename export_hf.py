import argparse
import gc
import json
import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from monarch_distill.configuration_monarch_gemma4 import MonarchGemma4Config
from monarch_distill.modeling_monarch_gemma4 import MonarchGemma4ForConditionalGeneration


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def checkpoint_tensor_names(layer_indices):
    return {
        f"model.language_model.layers.{layer_index}.mlp.{projection}.{factor}"
        for layer_index in layer_indices
        for projection in PROJECTIONS
        for factor in ("blk1", "blk2")
    }


def replaced_dense_weight_names(layer_indices):
    return {
        f"model.language_model.layers.{layer_index}.mlp.{projection}.weight"
        for layer_index in layer_indices
        for projection in PROJECTIONS
    }


def build_config(base_model: str, layer_indices, blocks: int):
    config_dict = AutoConfig.from_pretrained(base_model).to_dict()
    config_dict.pop("model_type", None)
    config_dict.pop("architectures", None)
    config_dict.pop("auto_map", None)
    return MonarchGemma4Config(
        **config_dict,
        monarch_compressed_layers=layer_indices,
        monarch_blocks_weights=blocks,
        monarch_factor_count=2,
        monarch_format_version=1,
        monarch_base_model=base_model,
        architectures=["MonarchGemma4ForConditionalGeneration"],
    )


def load_export_model(base_model: str, checkpoint: Path, layer_indices, blocks: int):
    config = build_config(base_model, layer_indices, blocks)
    model, loading_info = MonarchGemma4ForConditionalGeneration.from_pretrained(
        base_model,
        config=config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        output_loading_info=True,
    )

    expected_monarch = checkpoint_tensor_names(layer_indices)
    expected_dense = replaced_dense_weight_names(layer_indices)
    missing = set(loading_info["missing_keys"])
    unexpected = set(loading_info["unexpected_keys"])
    if missing != expected_monarch:
        raise RuntimeError(
            "base-model load produced an unexpected Monarch key set: "
            f"missing={sorted(missing ^ expected_monarch)[:10]}"
        )
    if unexpected != expected_dense:
        raise RuntimeError(
            "base-model load produced an unexpected replaced-dense key set: "
            f"unexpected={sorted(unexpected ^ expected_dense)[:10]}"
        )
    if loading_info["mismatched_keys"] or loading_info["error_msgs"]:
        raise RuntimeError(f"base-model load errors: {loading_info}")

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if set(state) != expected_monarch:
        raise RuntimeError(
            "checkpoint does not contain exactly the expected cumulative Monarch tensors: "
            f"difference={sorted(set(state) ^ expected_monarch)[:10]}"
        )

    named_parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in state.items():
            parameter = named_parameters[name]
            if parameter.shape != tensor.shape:
                raise RuntimeError(
                    f"checkpoint shape mismatch for {name}: {tuple(tensor.shape)} != {tuple(parameter.shape)}"
                )
            parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))

    return model


def write_model_card(output_dir: Path, repo_id: str, base_model: str, layer_indices, parameter_count: int):
    layers = ", ".join(str(index) for index in layer_indices)
    text = f"""---
license: apache-2.0
base_model: {base_model}
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- gemma4
- monarch-matrices
- model-compression
---

# Gemma 4 E2B Monarch 8-MLP

Experimental Gemma 4 E2B model in which language-model MLP layers {layers} are
replaced by two-factor rectangular Monarch linear maps. The factors were initialized
with dense-to-Monarch SVD projection and trained with activation alignment followed by
full-model logit distillation.

The exported model contains {parameter_count:,} parameters and preserves the original
Gemma 4 multimodal processor and architecture outside the selected MLPs.

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

This repository contains custom modeling code, so review it before enabling
`trust_remote_code=True`.

## Limitations

- This is an experimental compression artifact, not an official Google model.
- Only eight of the 35 language-model MLPs are compressed.
- Quality and inference speed have not been established on broad downstream benchmarks.
- The model should be evaluated for the intended task before deployment.

## Attribution

Derived from [{base_model}](https://huggingface.co/{base_model}). See `NOTICE` for
the modification summary.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def write_legal_files(output_dir: Path, base_model: str):
    shutil.copy2(Path(__file__).with_name("LICENSE"), output_dir / "LICENSE")
    notice = f"""Gemma 4 E2B Monarch model

This model is derived from {base_model}.

Modifications:
- Replaced language-model MLP layers 34 through 27 with two-factor rectangular
  Monarch linear maps.
- Initialized factors by rank-one SVD projection of each dense Monarch slice.
- Distilled the modified layers against the original model.

The original and modified files are distributed under the Apache License 2.0.
"""
    (output_dir / "NOTICE").write_text(notice, encoding="utf-8")


def verify_local_round_trip(model, processor, output_dir: Path):
    inputs = processor.tokenizer("Say hello in one short sentence.", return_tensors="pt")
    model.eval()
    with torch.no_grad():
        expected_logits = model(**inputs, use_cache=False).logits.float()

    restored = AutoModelForImageTextToText.from_pretrained(
        output_dir,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    restored.eval()
    with torch.no_grad():
        actual_logits = restored(**inputs, use_cache=False).logits.float()
        generated = restored.generate(**inputs, max_new_tokens=8, do_sample=False)

    torch.testing.assert_close(actual_logits, expected_logits, rtol=5e-3, atol=5e-3)
    print("[Verify] Local save/reload logits match within BF16 tolerance")
    print("[Verify] Text generation:", processor.tokenizer.decode(generated[0], skip_special_tokens=True))
    del restored, expected_logits, actual_logits, generated
    gc.collect()


def upload_private(output_dir: Path, repo_id: str, token: str):
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    before = api.model_info(repo_id)
    if not before.private:
        raise RuntimeError(f"refusing to upload because {repo_id} is not private")
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=output_dir)
    after = api.model_info(repo_id)
    if not after.private:
        raise RuntimeError(f"repository {repo_id} unexpectedly became public")


def parse_args():
    parser = argparse.ArgumentParser(description="Export a cumulative Monarch checkpoint as a standalone HF model")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="hexoy/gemma-4-e2b-monarch-8mlp")
    parser.add_argument("--base-model", default="google/gemma-4-E2B-it")
    parser.add_argument("--layers", default="34,33,32,31,30,29,28,27")
    parser.add_argument("--blocks", type=int, default=128)
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    layer_indices = [int(value) for value in args.layers.split(",") if value]
    if len(layer_indices) != 8 or len(set(layer_indices)) != 8:
        raise ValueError("the production export requires exactly eight unique compressed layer indices")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN")
    model = load_export_model(args.base_model, args.checkpoint, layer_indices, args.blocks)
    processor = AutoProcessor.from_pretrained(args.base_model, token=token)

    MonarchGemma4Config.register_for_auto_class("AutoConfig")
    MonarchGemma4ForConditionalGeneration.register_for_auto_class("AutoModelForImageTextToText")
    model.config.architectures = ["MonarchGemma4ForConditionalGeneration"]
    model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    processor.save_pretrained(args.output_dir)
    verify_local_round_trip(model, processor, args.output_dir)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    write_model_card(args.output_dir, args.repo_id, args.base_model, layer_indices, parameter_count)
    write_legal_files(args.output_dir, args.base_model)
    metadata = {
        "base_model": args.base_model,
        "compressed_layers": layer_indices,
        "monarch_blocks_weights": args.blocks,
        "monarch_factor_count": 2,
        "parameter_count": parameter_count,
        "checkpoint": str(args.checkpoint),
    }
    (args.output_dir / "export_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.upload:
        if not token:
            raise RuntimeError("HF_TOKEN is required for upload")
        upload_private(args.output_dir, args.repo_id, token)
        print(f"[Upload] Private model uploaded to https://huggingface.co/{args.repo_id}")
    print(f"[Export] Wrote {parameter_count:,} parameters to {args.output_dir}")


if __name__ == "__main__":
    main()
