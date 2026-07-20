import argparse
import tempfile

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

def move_inputs(inputs, device):
    return {name: value.to(device) if hasattr(value, "to") else value for name, value in inputs.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Verify a standalone local or private HF Monarch model")
    parser.add_argument("model", help="Local export directory or Hugging Face model id")
    parser.add_argument("--expected-layers", default="34,33,32,31,30,29,28,27")
    parser.add_argument("--expected-lora-rank", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    expected_layers = [int(value) for value in args.expected_layers.split(",") if value]
    with tempfile.TemporaryDirectory(prefix="monarch-hf-verify-") as cache_dir:
        processor = AutoProcessor.from_pretrained(
            args.model,
            trust_remote_code=True,
            cache_dir=cache_dir,
            force_download=True,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            trust_remote_code=True,
            cache_dir=cache_dir,
            force_download=True,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        model.eval()

        actual_layers = []
        for index, layer in enumerate(model.model.language_model.layers):
            if layer.mlp.gate_proj.__class__.__name__ == "MonarchLinear":
                actual_layers.append(index)
                if not all(
                    projection.__class__.__name__ == "MonarchLinear"
                    for projection in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj)
                ):
                    raise RuntimeError(f"layer {index} is only partially converted to Monarch")
                for projection in (layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj):
                    actual_rank = int(getattr(projection, "lora_rank", 0))
                    if actual_rank != args.expected_lora_rank:
                        raise RuntimeError(
                            f"layer {index} LoRA rank {actual_rank} != "
                            f"{args.expected_lora_rank}"
                        )
        if actual_layers != sorted(expected_layers):
            raise RuntimeError(f"compressed layer mismatch: {actual_layers} != {sorted(expected_layers)}")

        device = next(model.parameters()).device
        text_inputs = processor.tokenizer("Say hello in one sentence.", return_tensors="pt")
        text_inputs = move_inputs(text_inputs, device)
        with torch.no_grad():
            text_output = model.generate(**text_inputs, max_new_tokens=8, do_sample=False)
        print("[Verify] Text generation:", processor.tokenizer.decode(text_output[0], skip_special_tokens=True))

        image = Image.new("RGB", (224, 224), color=(220, 30, 30))
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is the dominant color in this image?"},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        image_inputs = processor(text=prompt, images=image, return_tensors="pt")
        image_inputs = move_inputs(image_inputs, device)
        with torch.no_grad():
            image_output = model.generate(**image_inputs, max_new_tokens=12, do_sample=False)
        print("[Verify] Image-text generation:", processor.tokenizer.decode(image_output[0], skip_special_tokens=True))
        print(f"[Verify] Parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
        print(f"[Verify] Monarch layers: {actual_layers}")


if __name__ == "__main__":
    main()
