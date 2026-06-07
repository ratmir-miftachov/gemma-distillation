import argparse
import gc
import re
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForImageTextToText, AutoTokenizer


MODEL_NAME = "google/gemma-4-E2B-it"
DATASET_NAME = "Rowan/hellaswag"
DEFAULT_CHECKPOINT = (
    "monarch_checkpoints_gemma4_e2b_mlp_4layer_b4_100step_maskfix/"
    "step_003_model_language_model_layers_31_mlp/unfrozen_weights.pt"
)
CHECKPOINT_4LAYER_100 = DEFAULT_CHECKPOINT
CHECKPOINT_4LAYER_200 = (
    "monarch_checkpoints_gemma4_e2b_mlp_4layer_b4_200step_maskfix/"
    "step_003_model_language_model_layers_31_mlp/unfrozen_weights.pt"
)
CHECKPOINT_4LAYER_COMPARISON = (
    ("4-layer 100-step Monarch", CHECKPOINT_4LAYER_100),
    ("4-layer 200-step Monarch", CHECKPOINT_4LAYER_200),
)
DEFAULT_LAYERS = (34, 33, 32, 31)
MONARCH_BLOCKS = 64


class MonarchLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_blocks: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.is_down_proj = out_features < in_features
        if not self.is_down_proj:
            self.n1 = n_blocks
            assert in_features % self.n1 == 0
            self.n2 = in_features // self.n1
            assert out_features % self.n2 == 0
            self.n3 = out_features // self.n2
        else:
            self.n3 = n_blocks
            assert out_features % self.n3 == 0
            self.n2 = out_features // self.n3
            assert in_features % self.n2 == 0
            self.n1 = in_features // self.n2

        self.blk1 = nn.Parameter(torch.empty(self.n1, self.n2, self.n2))
        self.blk2 = nn.Parameter(torch.empty(self.n2, self.n1, self.n3))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.blk1)
        nn.init.zeros_(self.blk2)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x = x.contiguous().view(-1, self.n1, self.n2)
        x = torch.einsum("bij,ijk->bik", x, self.blk1)
        x = x.transpose(1, 2).contiguous()
        x = torch.einsum("bij,ijk->bik", x, self.blk2)
        x = x.transpose(1, 2).contiguous()
        x = x.view(*orig_shape[:-1], self.out_features)
        if self.bias is not None:
            x = x + self.bias
        return x


def parse_layers(raw_layers: str) -> Tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw_layers.split(",") if part.strip())


def replace_linear_children_with_monarch(module: nn.Module, blocks: int):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            device = child.weight.device
            dtype = child.weight.dtype
            replacement = MonarchLinear(
                child.in_features,
                child.out_features,
                blocks,
                bias=child.bias is not None,
            )
            setattr(module, name, replacement.to(device=device, dtype=dtype))
        else:
            replace_linear_children_with_monarch(child, blocks)


def replace_mlp_layers_with_monarch(model: nn.Module, layers: Iterable[int], blocks: int):
    for layer_idx in layers:
        module_path = f"model.language_model.layers.{layer_idx}.mlp"
        mlp = model.get_submodule(module_path)
        replace_linear_children_with_monarch(mlp, blocks)


def load_model(model_name: str, compressed: bool, checkpoint: str, layers: Tuple[int, ...]):
    label = "compressed" if compressed else "baseline"
    print(f"[Load] Loading {label} model: {model_name}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.eval()

    if compressed:
        print(f"[Monarch] Replacing MLPs in layers: {list(layers)}")
        replace_mlp_layers_with_monarch(model, layers, MONARCH_BLOCKS)
        state_dict = torch.load(checkpoint, map_location="cpu")
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")
        print(f"[Checkpoint] Loaded {len(state_dict)} tensors from: {checkpoint}")

    return model


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def preprocess_hellaswag_text(text: str) -> str:
    text = text.strip()
    # Brackets are artifacts in the WikiHow portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def prepare_example(example: Dict, protocol: str) -> Dict:
    if protocol == "raw_ctx":
        context = example["ctx"]
        endings = list(example["endings"])
    elif protocol == "lm_eval_like":
        ctx = example["ctx_a"] + " " + example["ctx_b"].capitalize()
        context = preprocess_hellaswag_text(example["activity_label"] + ": " + ctx)
        endings = [preprocess_hellaswag_text(ending) for ending in example["endings"]]
    else:
        raise ValueError(f"Unknown HellaSwag protocol: {protocol}")

    return {
        "context": context,
        "endings": endings,
        "label": int(example["label"]),
    }


def get_model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def encode_lm_eval_pair(tokenizer, context: str, continuation: str) -> Tuple[List[int], List[int]]:
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]

    whole_enc = tokenizer.encode(context + continuation)
    context_enc = tokenizer.encode(context)
    continuation_enc = whole_enc[len(context_enc):]
    return context_enc, continuation_enc


def continuation_scores(
    model: nn.Module,
    tokenizer,
    context: str,
    endings: List[str],
) -> Dict[str, List[float]]:
    context_encs = []
    continuation_encs = []
    input_tensors = []
    input_lengths = []
    device = get_model_device(model)

    for ending in endings:
        context_enc, continuation_enc = encode_lm_eval_pair(tokenizer, context, " " + ending)
        context_encs.append(context_enc)
        continuation_encs.append(continuation_enc)
        input_ids = torch.tensor(
            (context_enc + continuation_enc)[:-1],
            dtype=torch.long,
            device=device,
        )
        input_tensors.append(input_ids)
        input_lengths.append(input_ids.numel())

    max_len = max(input_lengths)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    input_ids = torch.full(
        (len(input_tensors), max_len),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(input_tensors), max_len),
        dtype=torch.long,
        device=device,
    )
    for row_idx, row in enumerate(input_tensors):
        input_ids[row_idx, : row.numel()] = row
        attention_mask[row_idx, : row.numel()] = 1


    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        log_probs = F.log_softmax(logits.float(), dim=-1)

    raw_scores = []
    norm_scores = []
    token_counts = []
    for row_idx in range(len(endings)):
        continuation_enc = continuation_encs[row_idx]
        continuation_tensor = torch.tensor(continuation_enc, dtype=torch.long, device=device)
        if continuation_tensor.numel() == 0:
            raw_scores.append(float("-inf"))
            norm_scores.append(float("-inf"))
            token_counts.append(0)
        else:
            inplen = input_lengths[row_idx]
            contlen = continuation_tensor.numel()
            continuation_logits = log_probs[row_idx, inplen - contlen : inplen, :]
            token_log_probs = continuation_logits.gather(
                -1,
                continuation_tensor.unsqueeze(-1),
            ).squeeze(-1)
            raw_score = token_log_probs.sum().item()
            raw_scores.append(raw_score)
            norm_scores.append(raw_score / max(1, len(endings[row_idx])))
            token_counts.append(int(contlen))
    return {
        "raw": raw_scores,
        "normalized": norm_scores,
        "tokens": token_counts,
    }


def print_debug_example(example: Dict, scores: Dict[str, List[float]], raw_pred: int, norm_pred: int, name: str, idx: int):
    print(f"\n[Debug:{name}] example={idx}")
    print(f"context: {example['context']}")
    for choice_idx, ending in enumerate(example["endings"]):
        marker = "*" if choice_idx == int(example["label"]) else " "
        print(
            f"{marker} choice {choice_idx}: "
            f"raw={scores['raw'][choice_idx]:.4f} "
            f"norm={scores['normalized'][choice_idx]:.4f} "
            f"tokens={scores['tokens'][choice_idx]} "
            f"ending={ending}"
        )
    print(f"gold={int(example['label'])} raw_pred={raw_pred} norm_pred={norm_pred}")


def evaluate_model(
    model: nn.Module,
    tokenizer,
    examples: List[Dict],
    name: str,
    debug_examples: int = 0,
) -> Dict[str, float]:
    raw_correct = 0
    norm_correct = 0
    total = 0
    for idx, example in enumerate(examples, start=1):
        scores = continuation_scores(
            model,
            tokenizer,
            example["context"],
            list(example["endings"]),
        )
        raw_pred = max(range(len(scores["raw"])), key=lambda i: scores["raw"][i])
        norm_pred = max(range(len(scores["normalized"])), key=lambda i: scores["normalized"][i])
        raw_correct += int(raw_pred == int(example["label"]))
        norm_correct += int(norm_pred == int(example["label"]))
        total += 1
        if idx <= debug_examples:
            print_debug_example(example, scores, raw_pred, norm_pred, name, idx)
        if idx == 1 or idx % 10 == 0 or idx == len(examples):
            print(
                f"[{name}] {idx}/{len(examples)} "
                f"raw_acc={raw_correct / total:.4f} "
                f"acc_norm={norm_correct / total:.4f}"
            )

    return {
        "raw_correct": raw_correct,
        "norm_correct": norm_correct,
        "total": total,
        "raw_accuracy": raw_correct / total,
        "acc_norm": norm_correct / total,
    }


def unload_model(model: nn.Module):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parameter_reduction(layers: Tuple[int, ...]) -> Dict[str, int]:
    original_per_mlp = 56_623_104
    monarch_per_mlp = 2_727_936
    original = original_per_mlp * len(layers)
    monarch = monarch_per_mlp * len(layers)
    return {
        "original": original,
        "monarch": monarch,
        "saved": original - monarch,
    }


def main():
    parser = argparse.ArgumentParser(description="0-shot HellaSwag evaluator for Gemma 4 Monarch checkpoints.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--dataset-name", default=DATASET_NAME)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--layers", default=",".join(str(layer) for layer in DEFAULT_LAYERS))
    parser.add_argument("--protocol", choices=("raw_ctx", "lm_eval_like"), default="raw_ctx")
    parser.add_argument("--debug-examples", type=int, default=0)
    parser.add_argument(
        "--compare-4layer-checkpoints",
        action="store_true",
        help="Compare baseline plus the 4-layer 100-step and 200-step checkpoints.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Evaluate only the baseline model.",
    )
    args = parser.parse_args()

    layers = parse_layers(args.layers)
    tokenizer = load_tokenizer(args.model_name)
    dataset = load_dataset(args.dataset_name, split="validation")
    raw_examples = list(dataset.select(range(args.limit)))
    examples = [prepare_example(example, args.protocol) for example in raw_examples]
    print(f"[Dataset] Loaded {len(examples)} HellaSwag validation examples")
    print(f"[Protocol] {args.protocol}")

    checkpoint_specs = [("Compressed Monarch", args.checkpoint)]
    if args.compare_4layer_checkpoints:
        checkpoint_specs = list(CHECKPOINT_4LAYER_COMPARISON)
    if args.baseline_only:
        checkpoint_specs = []

    baseline_model = load_model(args.model_name, compressed=False, checkpoint=args.checkpoint, layers=layers)
    baseline = evaluate_model(
        baseline_model,
        tokenizer,
        examples,
        "Baseline Gemma 4 E2B",
        debug_examples=args.debug_examples,
    )
    unload_model(baseline_model)

    results = [("Baseline Gemma 4 E2B", baseline)]
    for label, checkpoint in checkpoint_specs:
        compressed_model = load_model(args.model_name, compressed=True, checkpoint=checkpoint, layers=layers)
        compressed = evaluate_model(
            compressed_model,
            tokenizer,
            examples,
            label,
            debug_examples=args.debug_examples,
        )
        unload_model(compressed_model)
        results.append((label, compressed))

    params = parameter_reduction(layers)

    print("\n=== HellaSwag 0-shot Results ===")
    print(f"Examples: {args.limit}")
    print(
        f"| Model | Raw Correct / {args.limit} | Raw Accuracy | Raw Delta | "
        f"Norm Correct / {args.limit} | acc_norm | Norm Delta |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for label, result in results:
        raw_delta = result["raw_accuracy"] - baseline["raw_accuracy"]
        norm_delta = result["acc_norm"] - baseline["acc_norm"]
        print(
            f"| {label} | {result['raw_correct']}/{result['total']} | "
            f"{result['raw_accuracy']:.4f} | {raw_delta:+.4f} | "
            f"{result['norm_correct']}/{result['total']} | "
            f"{result['acc_norm']:.4f} | {norm_delta:+.4f} |"
        )

    print("\n=== Parameter Reduction for Evaluated Checkpoint ===")
    print(f"Compressed MLP layers: {list(layers)}")
    print(f"Original MLP params:   {params['original']:,}")
    print(f"Monarch MLP params:    {params['monarch']:,}")
    print(f"Parameters saved:      {params['saved']:,}")


if __name__ == "__main__":
    main()
