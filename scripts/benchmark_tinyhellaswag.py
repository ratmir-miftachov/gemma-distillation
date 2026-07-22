from __future__ import annotations

import argparse
from pathlib import Path

from monarch_distill.benchmarks.tinyhellaswag import (
    DEFAULT_SEED,
    DEFAULT_TOKEN_FILE,
    run_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official TinyHellaSwag benchmark with lm-eval."
    )
    parser.add_argument("--model", required=True, help="HF model ID or local path")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--hf-token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--canonical-prompt-bundle",
        type=Path,
        help="Require exact prompts, labels, hashes, and tokenizer IDs from this bundle",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batch_size = (
        int(args.batch_size) if args.batch_size.isdecimal() else args.batch_size
    )
    result_path = run_benchmark(
        model_id=args.model,
        revision=args.revision,
        dtype_name=args.dtype,
        device=args.device,
        batch_size=batch_size,
        max_batch_size=args.max_batch_size,
        seed=args.seed,
        token_file=args.hf_token_file,
        output_dir=args.output_dir,
        canonical_prompt_bundle=args.canonical_prompt_bundle,
    )
    print(f"TinyHellaSwag result: {result_path}")


if __name__ == "__main__":
    main()
