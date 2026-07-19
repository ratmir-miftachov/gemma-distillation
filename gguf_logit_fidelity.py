from __future__ import annotations

import argparse
import json
import math
import urllib.request
from pathlib import Path
from typing import Any

from quantize_hf import resolve_hf_token
from tinyhellaswag_benchmark import server_tokenize


SOURCE_MODEL = "hexoy/gemma-4-e2b-monarch-35mlp"
SOURCE_REVISION = "f897353fca328b1cc5fd2e12d645773ca637f5f0"
DEFAULT_PROMPT_COUNT = 16


def canonical_prompts(bundle_path: Path, count: int) -> list[dict[str, Any]]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    records = bundle.get("records", [])
    if count <= 0 or len(records) < count:
        raise ValueError(f"canonical bundle has {len(records)} prompts; requested {count}")
    return [
        {
            "doc_id": record["doc_id"],
            "prompt": record["prompt"],
            "prompt_hash": record["prompt_hash"],
        }
        for record in records[:count]
    ]


def create_hf_reference(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    token = resolve_hf_token(args.hf_token_file)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        token=token,
        trust_remote_code=True,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        revision=args.revision,
        token=token,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": args.device},
        low_cpu_mem_usage=True,
    )
    model.eval()
    records = []
    with torch.inference_mode():
        for record in canonical_prompts(args.prompt_bundle, args.prompt_count):
            inputs = tokenizer(record["prompt"], return_tensors="pt").to(args.device)
            logits = model(**inputs, use_cache=False, return_dict=True).logits[0, -1]
            log_probs = torch.log_softmax(logits.float(), dim=-1).cpu()
            records.append(
                {
                    **record,
                    "token_ids": inputs["input_ids"][0].cpu(),
                    "log_probs": log_probs,
                    "top_token": int(log_probs.argmax().item()),
                }
            )
    payload = {
        "schema_version": 1,
        "model": args.model,
        "revision": args.revision,
        "vocab_size": records[0]["log_probs"].numel(),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"[GGUF KL] Wrote {len(records)} HF reference distributions to {args.output}")


def request_full_log_probs(base_url: str, prompt: str, vocab_size: int) -> tuple[list[int], list[float]]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/completion",
        data=json.dumps(
            {
                "prompt": prompt,
                "n_predict": 1,
                "n_probs": vocab_size,
                "temperature": -1.0,
                "cache_prompt": False,
            },
            separators=(",", ":"),
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        payload = json.loads(response.read().decode("utf-8"))
    probabilities = payload.get("completion_probabilities")
    if not isinstance(probabilities, list) or len(probabilities) != 1:
        raise RuntimeError("llama-server did not return one full-vocabulary distribution")
    entries = probabilities[0].get("top_logprobs", [])
    if len(entries) != vocab_size:
        raise RuntimeError(
            f"llama-server returned {len(entries)} probabilities, expected {vocab_size}"
        )
    token_ids = [int(entry["id"]) for entry in entries]
    log_probs = [float(entry["logprob"]) for entry in entries]
    return token_ids, log_probs


def distribution_metrics(reference: Any, token_ids: list[int], log_probs: list[float]) -> dict[str, float | int | bool]:
    import torch

    if len(token_ids) != reference.numel() or len(set(token_ids)) != reference.numel():
        raise ValueError("candidate distribution does not contain every token exactly once")
    candidate = torch.empty_like(reference, dtype=torch.float32)
    candidate[torch.tensor(token_ids, dtype=torch.long)] = torch.tensor(
        log_probs,
        dtype=torch.float32,
    )
    teacher = reference.float()
    teacher_probabilities = teacher.exp()
    kl = (teacher_probabilities * (teacher - candidate)).sum().item()
    return {
        "token_kl": kl,
        "mean_absolute_log_probability_error": (teacher - candidate).abs().mean().item(),
        "max_absolute_log_probability_error": (teacher - candidate).abs().max().item(),
        "top_token_reference": int(teacher.argmax().item()),
        "top_token_candidate": int(candidate.argmax().item()),
        "top_token_agreement": bool(teacher.argmax() == candidate.argmax()),
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def compare_gguf(args: argparse.Namespace) -> None:
    import torch

    reference = torch.load(args.reference, map_location="cpu", weights_only=False)
    records = []
    for expected in reference["records"]:
        server_ids = server_tokenize(
            args.base_url,
            expected["prompt"],
            add_special=True,
        )
        expected_ids = expected["token_ids"].tolist()
        if server_ids != expected_ids:
            raise RuntimeError(
                f"tokenizer mismatch for document {expected['doc_id']}: "
                f"HF={expected_ids[:20]}, GGUF={server_ids[:20]}"
            )
        token_ids, log_probs = request_full_log_probs(
            args.base_url,
            expected["prompt"],
            reference["vocab_size"],
        )
        records.append(
            {
                "doc_id": expected["doc_id"],
                "prompt_hash": expected["prompt_hash"],
                **distribution_metrics(expected["log_probs"], token_ids, log_probs),
            }
        )
    kls = [record["token_kl"] for record in records]
    payload = {
        "schema_version": 1,
        "reference_model": reference["model"],
        "reference_revision": reference["revision"],
        "candidate": args.candidate,
        "candidate_sha256": args.candidate_sha256,
        "prompt_count": len(records),
        "mean_token_kl": sum(kls) / len(kls),
        "p95_token_kl": percentile(kls, 0.95),
        "max_token_kl": max(kls),
        "top_token_agreement": sum(record["top_token_agreement"] for record in records)
        / len(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure full-vocabulary HF/GGUF token KL")
    subparsers = parser.add_subparsers(dest="command", required=True)

    reference = subparsers.add_parser("reference")
    reference.add_argument("--model", default=SOURCE_MODEL)
    reference.add_argument("--revision", default=SOURCE_REVISION)
    reference.add_argument("--prompt-bundle", type=Path, required=True)
    reference.add_argument("--prompt-count", type=int, default=DEFAULT_PROMPT_COUNT)
    reference.add_argument("--output", type=Path, required=True)
    reference.add_argument("--device", default="cuda:0")
    reference.add_argument(
        "--hf-token-file",
        type=Path,
        default=Path.home() / ".config/nebius-gemma/hf_read_token",
    )

    compare = subparsers.add_parser("compare")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--base-url", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--candidate-sha256", required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "reference":
        create_hf_reference(args)
    else:
        compare_gguf(args)


if __name__ == "__main__":
    main()
