import torch

from .config import CompressionConfig
from .losses import compute_validation_distill_loss_sum


def validation_eval_lengths(validation_examples, config: CompressionConfig):
    storage_len = validation_examples["input_ids"].shape[1]
    lengths = []
    for raw_length in config.validation_eval_lengths:
        length = int(raw_length)
        if length <= 0:
            raise ValueError(f"validation_eval_lengths must be positive, got {length}")
        if length > storage_len:
            raise ValueError(f"validation eval length {length} exceeds storage length {storage_len}")
        if length not in lengths:
            lengths.append(length)
    return lengths


def run_multilength_validation(
    *,
    config: CompressionConfig,
    validation_examples,
    teacher_model,
    student_model,
    writer,
    compressed_layer_count: int,
    module_path: str,
):
    if not config.validation_enabled or validation_examples is None:
        return

    teacher_was_training = teacher_model.training
    student_was_training = student_model.training
    teacher_model.eval()
    student_model.eval()

    num_examples = validation_examples["input_ids"].shape[0]
    batch_size = int(config.validation_batch_size)
    eval_lengths = validation_eval_lengths(validation_examples, config)
    metrics = {eval_len: {"loss_sum": 0.0, "weight_sum": 0.0} for eval_len in eval_lengths}

    try:
        with torch.no_grad():
            for eval_len in eval_lengths:
                for start in range(0, num_examples, batch_size):
                    end = min(start + batch_size, num_examples)
                    input_ids = validation_examples["input_ids"][start:end, :eval_len].to(teacher_model.device)
                    attention_mask = validation_examples["attention_mask"][start:end, :eval_len].to(teacher_model.device)
                    loss_weights = validation_examples["loss_weights"][start:end, :eval_len].to(teacher_model.device) * attention_mask.float()

                    teacher_out = teacher_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                    student_out = student_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

                    loss_sum, weight_sum = compute_validation_distill_loss_sum(
                        student_out.logits,
                        teacher_out.logits,
                        loss_weights,
                        config.kl_chunk_tokens,
                    )
                    metrics[eval_len]["loss_sum"] += loss_sum.item()
                    metrics[eval_len]["weight_sum"] += weight_sum.item()
    finally:
        if teacher_was_training:
            teacher_model.train()
        if student_was_training:
            student_model.train()

    for eval_len in eval_lengths:
        total_loss_sum = metrics[eval_len]["loss_sum"]
        total_weight_sum = metrics[eval_len]["weight_sum"]
        distill_loss = total_loss_sum / max(total_weight_sum, 1.0)
        writer.add_scalar(f"ValidationLoss/eval{eval_len}_distill", distill_loss, compressed_layer_count)
        writer.add_scalar(f"ValidationTokens/eval{eval_len}_active_weight", total_weight_sum, compressed_layer_count)
        print(
            f"[ValidationLoss] eval{eval_len} after {compressed_layer_count} compressed layer(s), "
            f"latest layer {module_path} | {num_examples} examples | "
            f"active_weight: {total_weight_sum:.1f} | distill_loss: {distill_loss:.4f}"
        )
    writer.flush()
