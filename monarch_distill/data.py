from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset

from .config import CompressionConfig


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    split: str
    kind: str
    name: Optional[str] = None


DEFAULT_DATASETS = [
    DatasetSpec("HuggingFaceH4/ultrachat_200k", "train_sft", "messages"),
    DatasetSpec("nvidia/OpenScienceReasoning-2", "train", "input_output"),
    DatasetSpec("nvidia/OpenCodeReasoning", "split_0", "input_output", name="split_0"),
    DatasetSpec("KodCode/KodCode-V1", "train", "question_solution"),
    DatasetSpec("Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b", "train", "input_output", name="stage1"),
    DatasetSpec("Open-Orca/OpenOrca", "train", "system_question_response"),
    DatasetSpec("allenai/sciq", "train", "question_support"),
]


class UnifiedDatasetStreamer(IterableDataset):
    def __init__(self, tokenizer, config: CompressionConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.max_epochs = config.max_epochs
        self.dataset_specs = DEFAULT_DATASETS
        self.streams = []
        self.stream_epochs = [0] * len(self.dataset_specs)

        for spec in self.dataset_specs:
            self.streams.append((spec.kind, self._open_stream(spec), spec))

    def _open_stream(self, spec: DatasetSpec):
        kwargs: Dict[str, Any] = {"streaming": True, "split": spec.split}
        if spec.name is not None:
            kwargs["name"] = spec.name
        return iter(load_dataset(spec.path, **kwargs))

    def format_to_messages(self, data: dict, kind: str) -> List[Dict[str, str]]:
        if kind == "messages":
            return data["messages"]
        if kind == "input_output":
            return [{"role": "user", "content": data["input"]}, {"role": "assistant", "content": data["output"]}]
        if kind == "question_solution":
            return [{"role": "user", "content": data["question"]}, {"role": "assistant", "content": data["solution"]}]
        if kind == "conversations":
            return [{"role": "user", "content": data["conversations"][0]}, {"role": "assistant", "content": data["conversations"][1]}]
        if kind == "system_question_response":
            return [
                {"role": "system", "content": data["system_prompt"]},
                {"role": "user", "content": data["question"]},
                {"role": "assistant", "content": data["response"]},
            ]
        if kind == "question_support":
            return [{"role": "user", "content": data["question"]}, {"role": "assistant", "content": data["support"]}]
        return []

    def __iter__(self):
        active_streams = list(range(len(self.streams)))

        while active_streams:
            idx = torch.randint(0, len(active_streams), (1,)).item()
            stream_idx = active_streams[idx]
            kind, stream, spec = self.streams[stream_idx]

            try:
                data = next(stream)
                messages = self.format_to_messages(data, kind)
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

                encoded = self.tokenizer(
                    text,
                    max_length=self.config.max_seq_len,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_ids = encoded["input_ids"][0]
                loss_weights = torch.ones_like(input_ids.clone(), dtype=torch.float)

                in_think = False
                for i, token_id in enumerate(input_ids):
                    token_str = self.tokenizer.decode([token_id])
                    if "<think>" in token_str:
                        in_think = True

                    loss_weights[i] = self.config.think_token_weight if in_think else 1.0

                    if "</think>" in token_str:
                        in_think = False

                yield {
                    "input_ids": input_ids,
                    "attention_mask": encoded["attention_mask"][0],
                    "loss_weights": loss_weights,
                }
            except StopIteration:
                self.stream_epochs[stream_idx] += 1
                if self.stream_epochs[stream_idx] < self.max_epochs:
                    self.streams[stream_idx] = (kind, self._open_stream(spec), spec)
                else:
                    print("[Data] Stream exhausted and removed")
                    active_streams.remove(stream_idx)


def build_validation_buffer(tokenizer, config: CompressionConfig):
    if not config.validation_enabled:
        return None

    num_examples = int(config.validation_num_examples)
    storage_seq_len = int(config.validation_storage_seq_len)
    validation_batch_size = int(config.validation_batch_size)
    print(
        f"[Validation] Building fixed validation buffer: {num_examples} examples "
        f"at storage_seq_len={storage_seq_len}"
    )
    rng_state = torch.random.get_rng_state()
    torch.manual_seed(config.validation_seed)

    validation_config = CompressionConfig(**config.to_dict())
    validation_config.max_seq_len = storage_seq_len
    validation_dataset = UnifiedDatasetStreamer(tokenizer, validation_config)
    validation_loader = DataLoader(validation_dataset, batch_size=validation_batch_size)
    validation_iter = iter(validation_loader)

    pieces = {"input_ids": [], "attention_mask": [], "loss_weights": []}
    examples_seen = 0
    try:
        while examples_seen < num_examples:
            batch = next(validation_iter)
            take = min(num_examples - examples_seen, batch["input_ids"].shape[0])
            for key in pieces:
                pieces[key].append(batch[key][:take].clone().cpu())
            examples_seen += take
    finally:
        torch.random.set_rng_state(rng_state)

    validation_examples = {key: torch.cat(value, dim=0) for key, value in pieces.items()}
    print(
        f"[Validation] Fixed validation buffer ready with "
        f"{validation_examples['input_ids'].shape[0]} examples at seq_len="
        f"{validation_examples['input_ids'].shape[1]}; training stream is independent"
    )
    return validation_examples


def make_training_loader(tokenizer, config: CompressionConfig) -> DataLoader:
    dataset = UnifiedDatasetStreamer(tokenizer, config)
    return DataLoader(dataset, batch_size=config.batch_size)
