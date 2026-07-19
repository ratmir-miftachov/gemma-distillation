from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


LLAMA_CPP_COMMIT = "571d0d540df04f25298d0e159e520d9fc62ed121"
UNSLOTH_COMMIT = "e8db1cecff48cfda2a3376f5d85b0a10a1170416"
OFFICIAL_REPOSITORY = "unsloth/gemma-4-E2B-it-GGUF"
OFFICIAL_REVISION = "0314792d7f1f7e229411f620751375812bb9faf2"
OFFICIAL_BF16 = "gemma-4-E2B-it-BF16.gguf"
OFFICIAL_DYNAMIC = "gemma-4-E2B-it-UD-Q4_K_XL.gguf"
OFFICIAL_IMATRIX = "imatrix_unsloth.gguf_file"
OFFICIAL_MMPROJ = "mmproj-BF16.gguf"
OFFICIAL_SHA256 = {
    OFFICIAL_BF16: "1eafd61d010ce8ca09db38f370aadd64c6d792db269c365ad0d9ea2709701890",
    OFFICIAL_DYNAMIC: "b52f438017efaec5debf1c0d8be690571e212a07c312f1102bbce927258cfc32",
    OFFICIAL_IMATRIX: "1828d3cc1674eaa32201917412cd0a8ec7a523c721547ae036689a2807ee2af4",
    OFFICIAL_MMPROJ: "a402f10fb5780bf91d03a10cd89061139f522bee2e679b1291bbfdcd71d9547d",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_git_revision(repository: Path, expected: str) -> None:
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if actual != expected:
        raise RuntimeError(f"unexpected revision for {repository}: {actual} != {expected}")


def load_tensor_type_map(gguf_path: Path, llama_cpp_dir: Path) -> dict[str, str]:
    gguf_python = str(llama_cpp_dir / "gguf-py")
    if gguf_python not in sys.path:
        sys.path.insert(0, gguf_python)
    from gguf import GGUFReader

    reader = GGUFReader(str(gguf_path), "r")
    return {tensor.name: tensor.tensor_type.name for tensor in reader.tensors}


def regex_escape_tensor_name(name: str) -> str:
    return "^" + re.escape(name) + "$"


def write_dynamic_recipe(
    *,
    official_dynamic: Path,
    candidate_bf16: Path,
    llama_cpp_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    official_types = load_tensor_type_map(official_dynamic, llama_cpp_dir)
    candidate_types = load_tensor_type_map(candidate_bf16, llama_cpp_dir)
    if official_types.keys() != candidate_types.keys():
        difference = sorted(official_types.keys() ^ candidate_types.keys())
        raise RuntimeError(
            "official and dense-equivalent GGUF tensor names differ: "
            f"{difference[:20]}"
        )

    output_type = official_types.get("output.weight")
    token_embedding_type = official_types.get("token_embd.weight")
    if not output_type or not token_embedding_type:
        raise RuntimeError("official GGUF is missing output or token embedding weights")

    entries = [
        f"{regex_escape_tensor_name(name)}={tensor_type.lower()}"
        for name, tensor_type in sorted(official_types.items())
        if name not in {"output.weight", "token_embd.weight"}
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return {
        "tensor_count": len(official_types),
        "override_count": len(entries),
        "output_tensor_type": output_type.lower(),
        "token_embedding_type": token_embedding_type.lower(),
        "type_counts": {
            tensor_type: sum(value == tensor_type for value in official_types.values())
            for tensor_type in sorted(set(official_types.values()))
        },
        "recipe_sha256": sha256_file(output_path),
    }


def run_command(command: list[str], *, cwd: Path | None = None) -> None:
    print("[GGUF]", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create BF16, Q4_K_M, and transferred Unsloth Dynamic GGUFs"
    )
    parser.add_argument("--dense-model-dir", type=Path, required=True)
    parser.add_argument("--llama-cpp-dir", type=Path, required=True)
    parser.add_argument("--unsloth-dir", type=Path, required=True)
    parser.add_argument("--official-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verify_git_revision(args.llama_cpp_dir, LLAMA_CPP_COMMIT)
    verify_git_revision(args.unsloth_dir, UNSLOTH_COMMIT)

    official_paths = {
        filename: args.official_dir / filename for filename in OFFICIAL_SHA256
    }
    for filename, expected_hash in OFFICIAL_SHA256.items():
        path = official_paths[filename]
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"official pinned artifact failed hash verification: {path}")

    converter = args.llama_cpp_dir / "convert_hf_to_gguf.py"
    quantizer = args.llama_cpp_dir / "build/bin/llama-quantize"
    if not converter.is_file() or not quantizer.is_file():
        raise FileNotFoundError("pinned llama.cpp converter or quantizer is missing")

    bf16 = args.output_dir / "monarch-35mlp-dense-equivalent-BF16.gguf"
    run_command(
        [
            sys.executable,
            str(converter),
            str(args.dense_model_dir),
            "--outfile",
            str(bf16),
            "--outtype",
            "bf16",
        ]
    )

    mmproj_requested = args.output_dir / "mmproj-BF16.gguf"
    run_command(
        [
            sys.executable,
            str(converter),
            str(args.dense_model_dir),
            "--outfile",
            str(mmproj_requested),
            "--outtype",
            "bf16",
            "--mmproj",
        ]
    )
    mmproj_candidates = sorted(args.output_dir.glob("*mmproj*BF16*.gguf"))
    if len(mmproj_candidates) != 1:
        raise RuntimeError(f"expected one BF16 multimodal projector, got {mmproj_candidates}")
    mmproj_bf16 = mmproj_candidates[0]

    q4 = args.output_dir / "monarch-35mlp-dense-equivalent-Q4_K_M.gguf"
    run_command(
        [
            str(quantizer),
            "--imatrix",
            str(official_paths[OFFICIAL_IMATRIX]),
            str(bf16),
            str(q4),
            "Q4_K_M",
            str(args.threads),
        ]
    )

    recipe_file = args.output_dir / "unsloth-dynamic-tensor-types.txt"
    recipe = write_dynamic_recipe(
        official_dynamic=official_paths[OFFICIAL_DYNAMIC],
        candidate_bf16=bf16,
        llama_cpp_dir=args.llama_cpp_dir,
        output_path=recipe_file,
    )
    dynamic = args.output_dir / "monarch-35mlp-dense-equivalent-UD-Q4_K_XL-transfer.gguf"
    run_command(
        [
            str(quantizer),
            "--imatrix",
            str(official_paths[OFFICIAL_IMATRIX]),
            "--output-tensor-type",
            recipe["output_tensor_type"],
            "--token-embedding-type",
            recipe["token_embedding_type"],
            "--tensor-type-file",
            str(recipe_file),
            str(bf16),
            str(dynamic),
            "Q4_K_M",
            str(args.threads),
        ]
    )

    mmproj_q8 = args.output_dir / "mmproj-Q8_0.gguf"
    run_command(
        [
            str(quantizer),
            str(mmproj_bf16),
            str(mmproj_q8),
            "Q8_0",
            str(args.threads),
        ]
    )

    artifacts = [bf16, q4, dynamic, mmproj_bf16, mmproj_q8, recipe_file]
    manifest = {
        "schema_version": 1,
        "description": "Unsloth Dynamic 2.0 recipe transfer; not newly calibrated",
        "llama_cpp_commit": LLAMA_CPP_COMMIT,
        "unsloth_commit": UNSLOTH_COMMIT,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_revision": OFFICIAL_REVISION,
        "official_artifacts": {
            filename: {"sha256": OFFICIAL_SHA256[filename]}
            for filename in sorted(OFFICIAL_SHA256)
        },
        "dynamic_recipe": recipe,
        "artifacts": [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifacts
        ],
    }
    manifest_path = args.output_dir / "gguf_recipe_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
