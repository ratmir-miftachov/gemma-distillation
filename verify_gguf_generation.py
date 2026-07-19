from __future__ import annotations

import argparse
import base64
import json
import urllib.request
from pathlib import Path
from typing import Any


TEXT_PROMPT = "In one sentence, explain why the sky appears blue."
IMAGE_PROMPT = "Describe the main subject of this image in one sentence."


def post_json(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"generation response has no choices: {response}")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"generation response has no text: {response}")
    return content.strip()


def completion(base_url: str, content: str | list[dict[str, Any]]) -> dict[str, Any]:
    return post_json(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 48,
            "temperature": 0.0,
            "seed": 1234,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify bounded text and image-text generation through llama-server"
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_data = base64.b64encode(args.image.read_bytes()).decode("ascii")
    image_media_type = "image/png" if args.image.suffix.lower() == ".png" else "image/jpeg"

    text_response = completion(args.base_url, TEXT_PROMPT)
    image_response = completion(
        args.base_url,
        [
            {"type": "text", "text": IMAGE_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_media_type};base64,{image_data}"},
            },
        ],
    )
    payload = {
        "schema_version": 1,
        "model": args.model_label,
        "text_prompt": TEXT_PROMPT,
        "text_output": response_text(text_response),
        "image_path": str(args.image),
        "image_prompt": IMAGE_PROMPT,
        "image_text_output": response_text(image_response),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
