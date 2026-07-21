import argparse
import gc
import json
import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi
from transformers import AutoModelForImageTextToText, AutoProcessor

from monarch_distill.config import default_lora_recovery_config
from monarch_distill.configuration_monarch_gemma4 import MonarchGemma4Config
from monarch_distill.lora import load_adapter_file, lora_state_dict
from monarch_distill.modeling_monarch_gemma4 import MonarchGemma4ForConditionalGeneration
from monarch_distill.recovery import load_recovery_student, resolve_hf_token


DEFAULT_REPO_ID = "hexoy/gemma-4-e2b-monarch-35mlp-lora-r8"
EXPECTED_TOTAL_PARAMETERS = 3_691_669_024


def write_model_card(output_dir: Path, repo_id: str, parameter_count: int):
    text = f"""---
license: apache-2.0
base_model: hexoy/gemma-4-e2b-monarch-35mlp
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- gemma4
- monarch-matrices
- lora
- model-compression
---

# Gemma 4 E2B Monarch 35-MLP + LoRA R8

Private experimental recovery model. All 35 language MLPs use the released two-factor
Monarch representation, with a native rank-8 LoRA residual on each gate, up, and down
projection. The frozen Monarch model was jointly distilled against the original dense
Gemma 4 teacher.

The model has {parameter_count:,} parameters. LoRA contributes 9,400,320 trainable
parameters (about 17.9 MiB in BF16) across 105 projections. Gemma 4 uses 6,144-wide
MLPs in layers 0-14 and double-wide 12,288 MLPs in layers 15-34.

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

This repository contains custom modeling code. Review it before enabling
`trust_remote_code=True`. This is an experimental derivative, not an official Google
release, and should be evaluated for its intended use.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def verify_round_trip(model, processor, output_dir: Path):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Explain in one sentence why the sky appears blue.",
                }
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        text=prompt,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    inputs = {name: value.to(device) for name, value in inputs.items()}
    model.eval()
    with torch.no_grad():
        expected_logits = model(**inputs, use_cache=False).logits.float().cpu()

    restored = AutoModelForImageTextToText.from_pretrained(
        output_dir,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    restored.eval()
    restored_inputs = {
        name: value.to(next(restored.parameters()).device) for name, value in inputs.items()
    }
    with torch.no_grad():
        actual_logits = restored(**restored_inputs, use_cache=False).logits.float().cpu()
        generated = restored.generate(
            **restored_inputs,
            max_new_tokens=12,
            do_sample=False,
        )

    torch.testing.assert_close(actual_logits, expected_logits, rtol=5e-3, atol=5e-3)
    expected_state = lora_state_dict(model)
    actual_state = lora_state_dict(restored)
    if set(expected_state) != set(actual_state):
        raise RuntimeError("restored LoRA key set differs from the exported model")
    for name in expected_state:
        torch.testing.assert_close(actual_state[name], expected_state[name], rtol=0, atol=0)
    continuation = processor.tokenizer.decode(
        generated[0][restored_inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    ).strip()
    if not continuation:
        raise RuntimeError("standalone text generation produced no continuation")
    print("[Verify] Standalone save/reload logits and LoRA tensors match")
    print("[Verify] Text generation:", continuation)
    del restored, expected_logits, actual_logits, generated
    gc.collect()


def upload_private(output_dir: Path, repo_id: str, token: str):
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    if not api.model_info(repo_id).private:
        raise RuntimeError(f"refusing to upload because {repo_id} is not private")
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=output_dir)
    info = api.model_info(repo_id)
    if not info.private:
        raise RuntimeError(f"repository {repo_id} unexpectedly became public")
    return info.sha


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export native Monarch LoRA recovery weights as a standalone model"
    )
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.adapter.is_file():
        raise FileNotFoundError(args.adapter)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = default_lora_recovery_config()
    token = resolve_hf_token()
    model, inventory = load_recovery_student(config, token)
    load_adapter_file(model, args.adapter)
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_TOTAL_PARAMETERS:
        raise RuntimeError(
            f"unexpected total parameter count: {parameter_count:,} != "
            f"{EXPECTED_TOTAL_PARAMETERS:,}"
        )

    processor = AutoProcessor.from_pretrained(
        config.student_model_name,
        revision=config.student_revision,
        token=token,
        trust_remote_code=True,
    )
    MonarchGemma4Config.register_for_auto_class("AutoConfig")
    MonarchGemma4ForConditionalGeneration.register_for_auto_class(
        "AutoModelForImageTextToText"
    )
    model.config.architectures = ["MonarchGemma4ForConditionalGeneration"]
    model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    processor.save_pretrained(args.output_dir)
    verify_round_trip(model, processor, args.output_dir)

    write_model_card(args.output_dir, args.repo_id, parameter_count)
    shutil.copy2(Path(__file__).with_name("LICENSE"), args.output_dir / "LICENSE")
    (args.output_dir / "NOTICE").write_text(
        "Derived from google/gemma-4-E2B-it and "
        "hexoy/gemma-4-e2b-monarch-35mlp. Added native rank-8 LoRA residuals "
        "to all 105 compressed language-MLP projections.\n",
        encoding="utf-8",
    )
    manifest = {
        "teacher_model": config.teacher_model_name,
        "teacher_revision": config.teacher_revision,
        "source_model": config.student_model_name,
        "source_revision": config.student_revision,
        "lora_rank": config.lora_rank,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "lora_inventory": inventory,
        "parameter_count": parameter_count,
        "adapter": str(args.adapter),
    }
    (args.output_dir / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.upload:
        if not token:
            raise RuntimeError("HF_TOKEN is required for upload")
        commit = upload_private(args.output_dir, args.repo_id, token)
        print(f"[Upload] Private model commit: {commit}")
    print(f"[Export] Wrote {parameter_count:,} parameters to {args.output_dir}")


if __name__ == "__main__":
    main()
