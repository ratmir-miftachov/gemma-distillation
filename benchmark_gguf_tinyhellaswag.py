from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import time
from pathlib import Path
from typing import Any

from gguf_recipe import LLAMA_CPP_COMMIT, sha256_file, verify_git_revision
from tinyhellaswag_benchmark import (
    DEFAULT_SEED,
    NUM_EXAMPLES,
    NUM_FEWSHOT,
    TINYBENCHMARKS_CACHE,
    TINYBENCHMARKS_COMMIT,
    TINYBENCHMARKS_DATA_SHA256,
    ensure_tinybenchmarks_data,
    server_tokenize,
    working_directory,
    write_json,
)


PATCH_REPLACEMENTS = (
    (
        '::common_tokenize(ctx, task.question + " " + answer, true)',
        '::common_tokenize(ctx, task.question + answer, false)',
    ),
    (
        "cur_task.log_probs[s] = log_prob / count;",
        """const size_t char_count = std::count_if(
                    cur_task.mc1.answers[s].begin(), cur_task.mc1.answers[s].end(),
                    [](unsigned char c) { return (c & 0xC0) != 0x80; });
                cur_task.log_probs[s] = log_prob / std::max<size_t>(char_count, 1);""",
    ),
    (
        """n_tot_answers += cur_task.log_probs.size();
            if (cur_task.mc1.labels[logprob_max_idx] == 1) {""",
        """int gold_idx = -1;
            for (size_t s = 0; s < cur_task.mc1.labels.size(); ++s) {
                if (cur_task.mc1.labels[s] == 1) gold_idx = (int) s;
            }
            LOG("TINY_RESULT\\t%zu\\t%zu\\t%d", i, logprob_max_idx, gold_idx);
            for (float score : cur_task.log_probs) LOG("\\t%.9g", score);
            LOG("\\n");

            n_tot_answers += cur_task.log_probs.size();
            if (cur_task.mc1.labels[logprob_max_idx] == 1) {""",
    ),
)


def patch_perplexity_source(llama_cpp_dir: Path) -> str:
    source = llama_cpp_dir / "tools/perplexity/perplexity.cpp"
    text = source.read_text(encoding="utf-8")
    changed = False
    for before, after in PATCH_REPLACEMENTS:
        if after in text:
            continue
        if before not in text:
            raise RuntimeError(f"pinned llama.cpp scorer patch context is missing: {before[:80]}")
        text = text.replace(before, after, 1)
        changed = True
    if changed:
        source.write_text(text, encoding="utf-8")
    patch = subprocess.run(
        ["git", "diff", "--", str(source.relative_to(llama_cpp_dir))],
        cwd=llama_cpp_dir,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if not patch:
        raise RuntimeError("llama.cpp scorer patch produced no recorded diff")
    return hashlib.sha256(patch.encode("utf-8")).hexdigest()


def _serialize_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<I", len(encoded)) + encoded


def _serialize_answers(answers: list[str], labels: list[int]) -> bytes:
    if len(answers) != len(labels):
        raise ValueError("answers and labels must have equal length")
    payload = struct.pack("<I", len(answers))
    payload += b"".join(_serialize_string(answer) for answer in answers)
    payload += struct.pack(f"<{len(labels)}i", *labels) if labels else b""
    return payload


def write_multiple_choice_tasks(bundle: dict[str, Any], output: Path) -> None:
    records = bundle.get("records", [])
    if len(records) != NUM_EXAMPLES:
        raise ValueError(f"expected {NUM_EXAMPLES} canonical records, got {len(records)}")
    tasks = []
    for record in records:
        gold = int(record["gold_choice"])
        labels = [int(index == gold) for index in range(4)]
        continuations = list(record["continuations"])
        if len(continuations) != 4:
            raise ValueError(f"document {record['doc_id']} does not have four continuations")
        tasks.append(
            _serialize_string(record["prompt"])
            + _serialize_answers(continuations, labels)
            + _serialize_answers([], [])
        )

    header_size = 4 + 4 * len(tasks)
    positions = []
    position = header_size
    for task in tasks:
        positions.append(position)
        position += len(task)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(
        struct.pack("<I", len(tasks))
        + struct.pack(f"<{len(positions)}I", *positions)
        + b"".join(tasks)
    )


def validate_server_tokenizer(bundle: dict[str, Any], base_url: str) -> None:
    for record in bundle["records"]:
        if server_tokenize(base_url, record["prompt"]) != record["prompt_token_ids"]:
            raise RuntimeError(f"prompt token mismatch for document {record['doc_id']}")
        for continuation, expected in zip(
            record["continuations"], record["request_token_ids"], strict=True
        ):
            actual = server_tokenize(base_url, record["prompt"] + continuation)
            if actual != expected:
                raise RuntimeError(f"request token mismatch for document {record['doc_id']}")


RESULT_PATTERN = re.compile(
    r"^TINY_RESULT\t(?P<index>\d+)\t(?P<selected>\d+)\t(?P<gold>-?\d+)"
    r"(?P<scores>(?:\t[-+0-9.eE]+){4})$"
)


def parse_scorer_output(text: str, bundle: dict[str, Any]) -> list[dict[str, Any]]:
    parsed: dict[int, tuple[int, int, list[float]]] = {}
    for line in text.splitlines():
        match = RESULT_PATTERN.match(line.strip())
        if not match:
            continue
        index = int(match["index"])
        scores = [float(value) for value in match["scores"].split("\t") if value]
        parsed[index] = (int(match["selected"]), int(match["gold"]), scores)
    if sorted(parsed) != list(range(NUM_EXAMPLES)):
        raise RuntimeError(f"scorer returned {len(parsed)} of {NUM_EXAMPLES} item records")

    items = []
    for index, record in enumerate(bundle["records"]):
        selected, gold, scores = parsed[index]
        if gold != int(record["gold_choice"]):
            raise RuntimeError(f"gold label mismatch for document {record['doc_id']}")
        items.append(
            {
                "doc_id": int(record["doc_id"]),
                "doc_hash": record.get("doc_hash"),
                "prompt_hash": record["prompt_hash"],
                "selected_choice": selected,
                "gold_choice": gold,
                "correct": selected == gold,
                "choice_normalized_scores": scores,
            }
        )
    return items


def gpirt_accuracy(correct: list[bool]) -> float:
    import numpy as np
    import tinyBenchmarks as tiny_benchmarks

    ensure_tinybenchmarks_data()
    with working_directory(TINYBENCHMARKS_CACHE):
        return float(
            tiny_benchmarks.evaluate(np.asarray(correct, dtype=float), "hellaswag")[
                "hellaswag"
            ]["gpirt"]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score canonical TinyHellaSwag choices with pinned llama.cpp"
    )
    parser.add_argument("--llama-cpp-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--prompt-bundle", type=Path, required=True)
    parser.add_argument("--server-base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verify_git_revision(args.llama_cpp_dir, LLAMA_CPP_COMMIT)

    bundle = json.loads(args.prompt_bundle.read_text(encoding="utf-8"))
    validate_server_tokenizer(bundle, args.server_base_url)
    patch_sha256 = patch_perplexity_source(args.llama_cpp_dir)
    subprocess.run(
        [
            "cmake",
            "--build",
            str(args.llama_cpp_dir / "build"),
            "--target",
            "llama-perplexity",
            f"-j{args.threads}",
        ],
        check=True,
    )
    scorer = args.llama_cpp_dir / "build/bin/llama-perplexity"
    tasks = args.output_dir / "canonical_tasks.bin"
    write_multiple_choice_tasks(bundle, tasks)
    command = [
        str(scorer),
        "-m",
        str(args.model),
        "-f",
        str(tasks),
        "--multiple-choice",
        "--multiple-choice-tasks",
        "0",
        "-ngl",
        "all",
        "-c",
        "4096",
        "-b",
        "2048",
        "-ub",
        "512",
        "-np",
        "4",
        "-t",
        str(args.threads),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, capture_output=True)
    elapsed = time.perf_counter() - started
    scorer_log = completed.stdout + completed.stderr
    (args.output_dir / "scorer.log").write_text(scorer_log, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(
            f"llama.cpp multiple-choice scorer failed with exit code {completed.returncode}"
        )
    items = parse_scorer_output(scorer_log, bundle)
    correct = [item["correct"] for item in items]
    result = {
        "schema_version": 1,
        "benchmark": {
            "task": "tinyHellaswag",
            "num_examples": NUM_EXAMPLES,
            "num_fewshot": NUM_FEWSHOT,
            "scoring": "character-length-normalized continuation log likelihood",
            "chat_template_applied": False,
            "official_gp_irt_accuracy": gpirt_accuracy(correct),
            "raw_anchor_accuracy": sum(correct) / len(correct),
            "prompt_bundle_sha256": bundle["sha256"],
        },
        "model": {
            "requested_id": args.model_label,
            "path": str(args.model),
            "sha256": sha256_file(args.model),
        },
        "runtime": {
            "backend": "llama.cpp-multiple-choice",
            "elapsed_seconds": elapsed,
            "threads": args.threads,
            "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
            "seed": DEFAULT_SEED,
        },
        "dependencies": {
            "llama_cpp_commit": LLAMA_CPP_COMMIT,
            "llama_cpp_scorer_patch_sha256": patch_sha256,
            "tinyBenchmarks_commit": TINYBENCHMARKS_COMMIT,
            "tinyBenchmarks_data_sha256": TINYBENCHMARKS_DATA_SHA256,
            "seed": DEFAULT_SEED,
        },
        "items": items,
    }
    write_json(args.output_dir / "result.json", result)
    print(json.dumps(result["benchmark"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
