from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from gguf_recipe import LLAMA_CPP_COMMIT, sha256_file, verify_git_revision


def benchmark_model(binary: Path, model: Path, repetitions: int) -> list[dict]:
    command = [
        str(binary),
        "--model",
        str(model),
        "--n-gpu-layers",
        "99",
        "--batch-size",
        "512",
        "--ubatch-size",
        "512",
        "--n-prompt",
        "512",
        "--n-gen",
        "128",
        "--repetitions",
        str(repetitions),
        "--output",
        "json",
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return json.loads(completed.stdout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled five-repeat llama-bench")
    parser.add_argument("--llama-cpp-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_git_revision(args.llama_cpp_dir, LLAMA_CPP_COMMIT)
    binary = args.llama_cpp_dir / "build/bin/llama-bench"
    if not binary.is_file():
        raise FileNotFoundError(binary)
    results = []
    for model in args.model:
        if not model.is_file():
            raise FileNotFoundError(model)
        results.append(
            {
                "model": str(model),
                "size_bytes": model.stat().st_size,
                "sha256": sha256_file(model),
                "measurements": benchmark_model(binary, model, args.repetitions),
            }
        )
    payload = {
        "schema_version": 1,
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
        "repetitions": args.repetitions,
        "prompt_tokens": 512,
        "generated_tokens": 128,
        "models": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
