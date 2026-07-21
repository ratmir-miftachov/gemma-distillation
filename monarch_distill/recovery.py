import gc
import json
import os
import random
import shutil
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer

from .config import LoRARecoveryConfig
from .configuration_monarch_gemma4 import MonarchGemma4Config
from .data import build_validation_buffer, make_training_loader
from .io import log_scalars
from .lora import (
    enable_lora_adapters,
    freeze_except_lora,
    load_adapter_file,
    lora_inventory,
    save_adapter_file,
)
from .losses import (
    compute_distillation_metric_sums,
    normalize_distillation_metric_sums,
)
from .modeling_monarch_gemma4 import MonarchGemma4ForConditionalGeneration


EXPECTED_LORA_MODULES = 105
EXPECTED_LORA_PARAMETERS_R8 = 9_400_320


def resolve_hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    token_path = Path.home() / ".config/nebius-gemma/hf_read_token"
    if token_path.is_file():
        return token_path.read_text(encoding="utf-8").strip() or None
    return None


def load_local_monarch_config(model_name: str, revision: str, token: str | None):
    source_config = AutoConfig.from_pretrained(
        model_name,
        revision=revision,
        token=token,
        trust_remote_code=True,
    )
    config_dict = source_config.to_dict()
    config_dict.pop("model_type", None)
    config_dict.pop("architectures", None)
    config_dict.pop("auto_map", None)
    config_dict["monarch_lora_rank"] = 0
    config_dict["monarch_lora_alpha"] = 1.0
    config_dict["monarch_lora_dropout"] = 0.0
    return MonarchGemma4Config(**config_dict)


def load_recovery_student(config: LoRARecoveryConfig, token: str | None):
    local_config = load_local_monarch_config(
        config.student_model_name,
        config.student_revision,
        token,
    )
    student, loading_info = MonarchGemma4ForConditionalGeneration.from_pretrained(
        config.student_model_name,
        revision=config.student_revision,
        token=token,
        config=local_config,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    loading_errors = {
        key: value
        for key, value in loading_info.items()
        if key in {"missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"}
        and value
    }
    if loading_errors:
        raise RuntimeError(f"source Monarch model did not load exactly: {loading_errors}")
    enabled = enable_lora_adapters(
        student,
        rank=config.lora_rank,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
    )
    inventory = lora_inventory(student)
    if len(enabled) != EXPECTED_LORA_MODULES:
        raise RuntimeError(
            f"expected {EXPECTED_LORA_MODULES} LoRA modules, found {len(enabled)}"
        )
    if config.lora_rank == 8 and inventory["parameter_count"] != EXPECTED_LORA_PARAMETERS_R8:
        raise RuntimeError(
            "rank-8 LoRA parameter count mismatch: "
            f"{inventory['parameter_count']:,} != {EXPECTED_LORA_PARAMETERS_R8:,}"
        )
    return student, inventory


def _rng_state():
    return {
        "python": random.getstate(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state):
    random.setstate(state["python"])
    torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


class LoRARecoveryTrainer:
    def __init__(self, config: LoRARecoveryConfig):
        from torch.utils.tensorboard import SummaryWriter

        self.config = config
        self.token = resolve_hf_token()
        self.save_root = Path(config.save_dir)
        self.resume_path = (
            Path(config.resume_from_checkpoint) if config.resume_from_checkpoint else None
        )
        if self.resume_path is None:
            if self.save_root.exists() and any(self.save_root.iterdir()):
                raise FileExistsError(f"recovery save directory is not empty: {self.save_root}")
            tensorboard_path = Path(config.tensorboard_log_dir)
            if tensorboard_path.exists() and any(tensorboard_path.iterdir()):
                raise FileExistsError(f"TensorBoard directory is not empty: {tensorboard_path}")
        self.save_root.mkdir(parents=True, exist_ok=True)

        random.seed(config.recovery_seed)
        torch.manual_seed(config.recovery_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.recovery_seed)

        self.writer = SummaryWriter(config.tensorboard_log_dir)
        print(f"[TensorBoard] Logging to: {config.tensorboard_log_dir}")
        print(
            f"[Load] Teacher {config.teacher_model_name}@{config.teacher_revision}"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.teacher_model_name,
            revision=config.teacher_revision,
            token=self.token,
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.teacher_model = AutoModelForImageTextToText.from_pretrained(
            config.teacher_model_name,
            revision=config.teacher_revision,
            token=self.token,
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.teacher_model.eval()
        for parameter in self.teacher_model.parameters():
            parameter.requires_grad = False

        print(
            f"[Load] Student {config.student_model_name}@{config.student_revision}"
        )
        self.student_model, self.inventory = load_recovery_student(config, self.token)
        self.student_model.eval()
        self.trainable_params = freeze_except_lora(self.student_model)
        trainable_count = sum(parameter.numel() for parameter in self.trainable_params)
        if trainable_count != self.inventory["parameter_count"]:
            raise RuntimeError("trainable parameter inventory does not contain only LoRA")
        print(
            f"[LoRA] {self.inventory['module_count']} modules, "
            f"{trainable_count:,} trainable parameters"
        )

        self.optimizer = torch.optim.AdamW(
            self.trainable_params,
            lr=config.recovery_lr,
            weight_decay=config.recovery_weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda _: 1.0,
        )
        self.validation_examples = build_validation_buffer(self.tokenizer, config)
        self.training_loader = make_training_loader(
            self.tokenizer,
            config,
            selection_seed=config.training_data_seed,
        )
        self.training_iter = iter(self.training_loader)
        self.global_step = 0
        self.batches_seen = 0
        self.best_metrics = None
        if self.resume_path is not None:
            self._load_checkpoint(self.resume_path)

    @property
    def device(self):
        return next(self.student_model.parameters()).device

    def _next_batch(self):
        try:
            batch = next(self.training_iter)
        except StopIteration:
            self.training_iter = iter(self.training_loader)
            batch = next(self.training_iter)
        self.batches_seen += 1
        return batch

    def _replay_training_stream(self, batches_seen: int):
        if batches_seen <= 0:
            return
        print(f"[Resume] Replaying {batches_seen} deterministic training batches")
        self.training_loader = make_training_loader(
            self.tokenizer,
            self.config,
            selection_seed=self.config.training_data_seed,
        )
        self.training_iter = iter(self.training_loader)
        self.batches_seen = 0
        for _ in range(batches_seen):
            self._next_batch()

    def _load_checkpoint(self, checkpoint_dir: Path):
        state_path = checkpoint_dir / "trainer_state.pt"
        adapter_path = checkpoint_dir / "adapter_model.safetensors"
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        load_adapter_file(self.student_model, adapter_path)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.global_step = int(state["global_step"])
        self.best_metrics = state.get("best_metrics")
        self._replay_training_stream(int(state["batches_seen"]))
        _restore_rng_state(state["rng_state"])
        print(
            f"[Resume] Loaded step {self.global_step}, "
            f"batches_seen={self.batches_seen}"
        )

    def _save_checkpoint(self, metrics: dict, *, best: bool = False):
        checkpoint_dir = self.save_root / f"step_{self.global_step:07d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        save_adapter_file(
            self.student_model,
            checkpoint_dir / "adapter_model.safetensors",
        )
        state = {
            "global_step": self.global_step,
            "batches_seen": self.batches_seen,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "rng_state": _rng_state(),
            "best_metrics": self.best_metrics,
            "validation_metrics": metrics,
        }
        torch.save(state, checkpoint_dir / "trainer_state.pt")
        (checkpoint_dir / "recovery_config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        if best:
            best_dir = self.save_root / "best"
            temporary = self.save_root / ".best.tmp"
            if temporary.exists():
                shutil.rmtree(temporary)
            shutil.copytree(checkpoint_dir, temporary)
            if best_dir.exists():
                shutil.rmtree(best_dir)
            os.replace(temporary, best_dir)
        print(f"[Checkpoint] Saved {checkpoint_dir}{' and best' if best else ''}")

    def validate(self, step: int):
        self.teacher_model.eval()
        self.student_model.eval()
        totals = {
            "ce_sum": 0.0,
            "teacher_entropy_sum": 0.0,
            "weight_sum": 0.0,
            "agreement_count": 0.0,
            "active_count": 0.0,
        }
        examples = self.validation_examples
        batch_size = int(self.config.validation_batch_size)
        eval_len = int(self.config.validation_eval_lengths[0])
        with torch.no_grad():
            for start in range(0, examples["input_ids"].shape[0], batch_size):
                end = min(start + batch_size, examples["input_ids"].shape[0])
                input_ids = examples["input_ids"][start:end, :eval_len].to(self.device)
                attention_mask = examples["attention_mask"][start:end, :eval_len].to(
                    self.device
                )
                loss_weights = examples["loss_weights"][start:end, :eval_len].to(
                    self.device
                ) * attention_mask.float()
                teacher_logits = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
                student_logits = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                ).logits
                sums = compute_distillation_metric_sums(
                    student_logits,
                    teacher_logits,
                    loss_weights,
                    self.config.kl_chunk_tokens,
                )
                for key in totals:
                    totals[key] += float(sums[key].detach().cpu())
                del teacher_logits, student_logits, sums

        weight_sum = max(totals["weight_sum"], 1.0)
        active_count = max(totals["active_count"], 1.0)
        cross_entropy = totals["ce_sum"] / weight_sum
        true_kl = max(
            0.0,
            (totals["ce_sum"] - totals["teacher_entropy_sum"]) / weight_sum,
        )
        metrics = {
            "cross_entropy": cross_entropy,
            "true_kl": true_kl,
            "top1_agreement": totals["agreement_count"] / active_count,
            "active_weight": totals["weight_sum"],
            "active_tokens": totals["active_count"],
        }
        log_scalars(
            self.writer,
            {f"Validation/{key}": value for key, value in metrics.items()},
            step,
        )
        self.writer.flush()
        print(
            f"[Validation] step={step} | CE={cross_entropy:.6f} | "
            f"KL={true_kl:.6f} | top1={metrics['top1_agreement']:.4f}"
        )
        return metrics

    def run(self):
        if self.global_step == 0:
            baseline = self.validate(0)
            self.best_metrics = {**baseline, "step": 0}
            self._save_checkpoint(baseline, best=True)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        while self.global_step < self.config.recovery_steps:
            batch = self._next_batch()
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            loss_weights = batch["loss_weights"].to(self.device) * attention_mask.float()

            self.optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = self.teacher_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    use_cache=False,
                ).logits
            student_logits = self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
            ).logits
            sums = compute_distillation_metric_sums(
                student_logits,
                teacher_logits,
                loss_weights,
                self.config.kl_chunk_tokens,
            )
            metrics = normalize_distillation_metric_sums(sums)
            loss = metrics["cross_entropy"]
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.trainable_params,
                self.config.recovery_grad_clip,
            )
            self.optimizer.step()
            self.scheduler.step()
            self.global_step += 1

            if self.global_step == 1 or self.global_step % self.config.tensorboard_log_interval == 0:
                elapsed = max(time.perf_counter() - started, 1e-9)
                log_scalars(
                    self.writer,
                    {
                        "Recovery/cross_entropy": metrics["cross_entropy"],
                        "Recovery/true_kl": metrics["true_kl"],
                        "Recovery/top1_agreement": metrics["top1_agreement"],
                        "Recovery/active_weight": metrics["active_weight"],
                        "Recovery/active_tokens": metrics["active_tokens"],
                        "Recovery/lr": self.optimizer.param_groups[0]["lr"],
                        "Recovery/grad_norm": grad_norm,
                        "Recovery/steps_per_second": self.global_step / elapsed,
                        "Recovery/token_slots_per_second": (
                            self.global_step
                            * self.config.batch_size
                            * self.config.max_seq_len
                            / elapsed
                        ),
                        "Recovery/peak_cuda_gib": (
                            torch.cuda.max_memory_allocated() / 2**30
                            if torch.cuda.is_available()
                            else 0.0
                        ),
                    },
                    self.global_step,
                )
            if self.global_step == 1 or self.global_step % 10 == 0:
                print(
                    f"[Recovery] step={self.global_step}/{self.config.recovery_steps} | "
                    f"CE={loss.item():.6f} | KL={metrics['true_kl'].item():.6f} | "
                    f"top1={metrics['top1_agreement'].item():.4f}"
                )

            del (
                batch,
                input_ids,
                attention_mask,
                loss_weights,
                teacher_logits,
                student_logits,
                sums,
                metrics,
                loss,
            )

            validation_metrics = None
            if (
                self.global_step % self.config.validation_interval == 0
                or self.global_step == self.config.recovery_steps
            ):
                validation_metrics = self.validate(self.global_step)
                if validation_metrics["true_kl"] < self.best_metrics["true_kl"]:
                    self.best_metrics = {**validation_metrics, "step": self.global_step}
                    is_best = True
                else:
                    is_best = False
            else:
                is_best = False

            if (
                self.global_step % self.config.checkpoint_interval == 0
                or self.global_step == self.config.recovery_steps
            ):
                self._save_checkpoint(validation_metrics or {}, best=is_best)

        elapsed = time.perf_counter() - started
        summary = {
            "steps": self.global_step,
            "elapsed_seconds": elapsed,
            "best_metrics": self.best_metrics,
            "inventory": self.inventory,
            "peak_cuda_gib": (
                torch.cuda.max_memory_allocated() / 2**30
                if torch.cuda.is_available()
                else 0.0
            ),
        }
        (self.save_root / "recovery_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[Recovery] Complete: {json.dumps(summary, indent=2)}")
        return summary

    def close(self):
        self.writer.flush()
        self.writer.close()
        self.training_iter = None
        self.training_loader = None
        self.validation_examples = None
        self.teacher_model = None
        self.student_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
