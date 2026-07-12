import gc
import os
import time
from collections import deque
from typing import List

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

from .config import CompressionConfig
from .data import build_validation_buffer, make_training_loader
from .io import log_scalars, profile_activations, save_unfrozen_checkpoint
from .losses import compute_attn_kl_loss, compute_cka_loss, compute_custom_kl_loss
from .monarch import MonarchLinear, replace_with_monarch
from .validation import run_multilength_validation


def select_modules_to_compress(num_layers: int, config: CompressionConfig) -> List[str]:
    modules_to_compress = []
    if config.compress_lm_head:
        modules_to_compress.append("lm_head")
    for i in reversed(range(num_layers)):
        if config.compress_mlp:
            modules_to_compress.append(f"model.language_model.layers.{i}.mlp")
        if config.compress_attention:
            modules_to_compress.append(f"model.language_model.layers.{i}.self_attn")
    if config.compress_embeddings:
        modules_to_compress.append("model.language_model.embed_tokens")
    return modules_to_compress[: config.max_modules]


class MonarchCompressor:
    def __init__(self, config: CompressionConfig):
        from torch.utils.tensorboard import SummaryWriter

        self.config = config
        self.writer = SummaryWriter(config.tensorboard_log_dir)
        print(f"[TensorBoard] Logging to: {config.tensorboard_log_dir}")

        print(f"[Load] Loading tokenizer and Gemma model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.teacher_model = AutoModelForImageTextToText.from_pretrained(
            config.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        self.student_model = AutoModelForImageTextToText.from_pretrained(
            config.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        )

        self.dataloader = make_training_loader(self.tokenizer, config)
        self.data_iter = iter(self.dataloader)
        self.prefetch_queue = deque()
        self.validation_examples = build_validation_buffer(self.tokenizer, config)
        self.teacher_activations = {}
        self.student_activations = {}
        self._in_validation = False

    def _next_raw_batch(self):
        try:
            return next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            return next(self.data_iter)

    def get_batch(self):
        prefetch_batches = max(0, int(self.config.prefetch_batches))
        if prefetch_batches <= 0:
            return self._next_raw_batch()

        while len(self.prefetch_queue) < prefetch_batches:
            self.prefetch_queue.append(self._next_raw_batch())
        return self.prefetch_queue.popleft()

    def should_log_step(self, step: int, total_steps: int) -> bool:
        interval = int(self.config.tensorboard_log_interval)
        return step == 0 or step == total_steps - 1 or (interval > 0 and step % interval == 0)

    def should_flush_step(self, step: int, total_steps: int) -> bool:
        interval = int(self.config.tensorboard_flush_interval)
        return step == 0 or step == total_steps - 1 or (interval > 0 and step > 0 and step % interval == 0)

    def record_phase_throughput(self, module_index: int, mod_path: str, phase_name: str, steps: int, elapsed: float):
        elapsed = max(elapsed, 1e-9)
        examples = steps * int(self.config.batch_size)
        token_slots = examples * int(self.config.max_seq_len)
        metrics = {
            f"Throughput/{mod_path}_{phase_name}_seconds": elapsed,
            f"Throughput/{mod_path}_{phase_name}_steps_per_sec": steps / elapsed,
            f"Throughput/{mod_path}_{phase_name}_examples_per_sec": examples / elapsed,
            f"Throughput/{mod_path}_{phase_name}_token_slots_per_sec": token_slots / elapsed,
        }
        log_scalars(self.writer, metrics, module_index)
        print(
            f"[Throughput] {mod_path} | {phase_name} | {elapsed:.2f}s | "
            f"{steps / elapsed:.2f} steps/s | {examples / elapsed:.2f} examples/s | "
            f"{token_slots / elapsed:.2f} token-slots/s"
        )

    def freeze_all_student(self):
        for param in self.student_model.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def replace_with_monarch(self, module_path: str):
        return replace_with_monarch(
            self.student_model,
            module_path,
            self.config.monarch_blocks_weights,
            self.config.monarch_blocks_head_embed,
            self.config.monarch_init_method,
        )

    def run_validation(self, compressed_layer_count: int, mod_path: str):
        self._in_validation = True
        try:
            run_multilength_validation(
                config=self.config,
                validation_examples=self.validation_examples,
                teacher_model=self.teacher_model,
                student_model=self.student_model,
                writer=self.writer,
                compressed_layer_count=compressed_layer_count,
                module_path=mod_path,
            )
        finally:
            self._in_validation = False

    def clear_training_buffers(self, tracked_paths: List[str]):
        self.teacher_activations = {}
        for path in tracked_paths:
            self.student_activations[path] = []
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_resume_checkpoint(self, checkpoint_path: str, completed_paths: List[str]):
        if not checkpoint_path:
            return
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")

        print(f"[Resume] Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(state_dict, dict):
            raise TypeError(f"resume checkpoint must be a state dict, got {type(state_dict)}")

        named_params = dict(self.student_model.named_parameters())
        expected_prefixes = tuple(path + "." for path in completed_paths)
        unexpected = []
        missing = []
        shape_mismatches = []

        for name, tensor in state_dict.items():
            if expected_prefixes and not name.startswith(expected_prefixes):
                unexpected.append(name)
                continue
            if name not in named_params:
                missing.append(name)
                continue
            param = named_params[name]
            if tuple(param.shape) != tuple(tensor.shape):
                shape_mismatches.append((name, tuple(param.shape), tuple(tensor.shape)))
                continue
            param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))

        if unexpected:
            raise RuntimeError(f"resume checkpoint has unexpected parameter prefixes: {unexpected[:5]}")
        if missing:
            raise RuntimeError(f"resume checkpoint parameters not found in model: {missing[:5]}")
        if shape_mismatches:
            raise RuntimeError(f"resume checkpoint shape mismatches: {shape_mismatches[:5]}")

        print(f"[Resume] Loaded {len(state_dict)} tensors for {len(completed_paths)} completed module(s)")
        self.clear_training_buffers(completed_paths)

    def run_compression(self):
        teacher_layers = self.teacher_model.model.language_model.layers
        num_layers = len(teacher_layers)
        print(f"[Model] Resolved Gemma backbone path: model.language_model.layers ({num_layers} layers)")

        modules_to_compress = select_modules_to_compress(num_layers, self.config)
        print(f"[Target] Selected modules: {modules_to_compress}")
        if len(modules_to_compress) != self.config.max_modules:
            raise RuntimeError(
                f"MLP scaling run expected exactly {self.config.max_modules} target modules, "
                f"got {len(modules_to_compress)}"
            )

        unfrozen_modules = []
        unfrozen_paths = []
        self.teacher_activations = {}
        self.student_activations = {}

        def get_t_hook(name):
            def hook(module, inp, out):
                if getattr(self, "_in_validation", False):
                    return
                self.teacher_activations[name] = out[0] if isinstance(out, tuple) else out
            return hook

        def get_s_hook(name):
            def hook(module, inp, out):
                if getattr(self, "_in_validation", False):
                    return
                if name not in self.student_activations:
                    self.student_activations[name] = []
                self.student_activations[name].append(out[0] if isinstance(out, tuple) else out)
            return hook

        def register_hooks(mod_path):
            t_submod = self.teacher_model.get_submodule(mod_path)
            s_submod = self.student_model.get_submodule(mod_path)
            t_submod.register_forward_hook(get_t_hook(mod_path))
            s_submod.register_forward_hook(get_s_hook(mod_path))

        resume_checkpoint = self.config.resume_from_checkpoint
        resume_start_idx = int(self.config.resume_start_module_index or 0)
        if resume_checkpoint:
            if resume_start_idx <= 0 or resume_start_idx >= len(modules_to_compress):
                raise ValueError("resume_start_module_index must point to the first unfinished module")
            completed_paths = modules_to_compress[:resume_start_idx]
            print(
                f"[Resume] Reconstructing {len(completed_paths)} completed module(s); "
                f"next target will be {modules_to_compress[resume_start_idx]}"
            )
            for completed_path in completed_paths:
                completed_module = self.replace_with_monarch(completed_path)
                unfrozen_modules.append(completed_module)
                unfrozen_paths.append(completed_path)
                register_hooks(completed_path)
            self.load_resume_checkpoint(resume_checkpoint, completed_paths)
            self.freeze_all_student()
            print(f"[Resume] Ready to continue from module index {resume_start_idx}")

        for step_idx, mod_path in enumerate(modules_to_compress):
            if step_idx < resume_start_idx:
                continue

            student_module = self.replace_with_monarch(mod_path)
            self.freeze_all_student()
            self.unfreeze_module(student_module)
            unfrozen_modules.append(student_module)
            unfrozen_paths.append(mod_path)
            register_hooks(mod_path)

            optimizer_p1 = torch.optim.AdamW(student_module.parameters(), lr=self.config.lr_phase1)

            phase1_start = time.perf_counter()
            for step in range(self.config.phase1_steps):
                batch = self.get_batch()
                input_ids = batch["input_ids"].to(self.teacher_model.device)
                attention_mask = batch["attention_mask"].to(self.teacher_model.device)
                loss_weights = batch["loss_weights"].to(self.teacher_model.device) * attention_mask.float()

                # Previous layer hooks remain registered for global adjustment.
                # Clearing their lists prevents retained graphs from accumulating.
                for u_path in unfrozen_paths:
                    self.student_activations[u_path] = []

                with torch.no_grad():
                    teacher_out = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=False,
                        output_attentions=self.config.compress_attention,
                        use_cache=False,
                    )

                student_out = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    output_attentions=self.config.compress_attention,
                    use_cache=False,
                )

                if mod_path == "lm_head":
                    phase1_loss = compute_custom_kl_loss(
                        student_out.logits,
                        teacher_out.logits,
                        loss_weights,
                        self.config.kl_chunk_tokens,
                    )
                    loss_name = "KL Loss"
                else:
                    teacher_act = self.teacher_activations[mod_path]
                    student_acts = self.student_activations[mod_path]
                    cka_loss = sum(compute_cka_loss(teacher_act, student_act) for student_act in student_acts) / len(student_acts)
                    phase1_loss = cka_loss
                    loss_name = "CKA Loss"

                    if "self_attn" in mod_path:
                        layer_idx = int(mod_path.split(".")[3])
                        teacher_attn_map = teacher_out.attentions[layer_idx]
                        student_attn_map = student_out.attentions[layer_idx]
                        attn_kl = compute_attn_kl_loss(student_attn_map, teacher_attn_map)
                        phase1_loss += self.config.attn_kl_weight * attn_kl
                        loss_name = "CKA + Attn KL Loss"

                optimizer_p1.zero_grad(set_to_none=True)
                phase1_loss.backward()
                torch.nn.utils.clip_grad_norm_(student_module.parameters(), max_norm=1.0)
                optimizer_p1.step()

                if self.config.log_training_scalars and self.should_log_step(step, self.config.phase1_steps):
                    log_scalars(self.writer, {f"Phase1/{mod_path}_loss": phase1_loss}, step)
                if self.should_flush_step(step, self.config.phase1_steps):
                    self.writer.flush()
                if step == 0 or step == self.config.phase1_steps - 1 or step % 10 == 0:
                    print(f"[Phase 1] {mod_path} | Step {step}/{self.config.phase1_steps} | {loss_name}: {phase1_loss.item():.4f}")

            self.record_phase_throughput(
                step_idx + 1,
                mod_path,
                "phase1",
                self.config.phase1_steps,
                time.perf_counter() - phase1_start,
            )
            self.clear_training_buffers(unfrozen_paths)

            if mod_path == "lm_head":
                print("[Save] Saving lm_head after Phase 1")
                save_path = save_unfrozen_checkpoint(self.student_model, self.config.save_dir, step_idx, mod_path)
                print(f"[Save] Saved successfully to: {save_path}\n")
                self.writer.flush()
                self.run_validation(step_idx + 1, mod_path)
                continue

            for module in unfrozen_modules:
                self.unfreeze_module(module)

            phase2_params = []
            for module in unfrozen_modules:
                phase2_params.extend(list(module.parameters()))

            optimizer_p2 = torch.optim.AdamW(phase2_params, lr=self.config.lr_phase2)

            phase2_start = time.perf_counter()
            for step in range(self.config.phase2_steps):
                batch = self.get_batch()
                input_ids = batch["input_ids"].to(self.teacher_model.device)
                attention_mask = batch["attention_mask"].to(self.teacher_model.device)
                loss_weights = batch["loss_weights"].to(self.teacher_model.device) * attention_mask.float()

                for u_path in unfrozen_paths:
                    self.student_activations[u_path] = []

                with torch.no_grad():
                    teacher_out = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=False,
                        output_attentions=self.config.compress_attention,
                        use_cache=False,
                    )
                    teacher_logits = teacher_out.logits

                student_out = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    output_attentions=self.config.compress_attention,
                    use_cache=False,
                )
                student_logits = student_out.logits

                kl_loss = compute_custom_kl_loss(
                    student_logits,
                    teacher_logits,
                    loss_weights,
                    self.config.kl_chunk_tokens,
                )

                cka_loss = 0.0
                total_attn_kl = 0.0
                total_comparisons = 0

                for u_path in unfrozen_paths:
                    if u_path in ["lm_head", "model.embed_tokens"]:
                        continue

                    teacher_act = self.teacher_activations[u_path]
                    for student_act in self.student_activations[u_path]:
                        cka_loss += compute_cka_loss(teacher_act, student_act)
                        total_comparisons += 1

                    if "self_attn" in u_path:
                        layer_idx = int(u_path.split(".")[3])
                        teacher_attn_map = teacher_out.attentions[layer_idx]
                        student_attn_map = student_out.attentions[layer_idx]
                        total_attn_kl += compute_attn_kl_loss(student_attn_map, teacher_attn_map)

                if total_comparisons > 0:
                    cka_loss = cka_loss / total_comparisons

                total_loss = kl_loss + cka_loss + (self.config.attn_kl_weight * total_attn_kl)

                optimizer_p2.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(phase2_params, max_norm=1.0)
                optimizer_p2.step()

                if self.config.log_training_scalars and self.should_log_step(step, self.config.phase2_steps):
                    log_scalars(
                        self.writer,
                        {
                            f"Phase2/{mod_path}_total_loss": total_loss,
                            f"Phase2/{mod_path}_kl_loss": kl_loss,
                            f"Phase2/{mod_path}_cka_loss": cka_loss,
                        },
                        step,
                    )
                if self.should_flush_step(step, self.config.phase2_steps):
                    self.writer.flush()
                if step == 0 or step == self.config.phase2_steps - 1 or step % 10 == 0:
                    cka_value = cka_loss if isinstance(cka_loss, float) else cka_loss.item()
                    print(
                        f"[Phase 2] {mod_path} | Step {step}/{self.config.phase2_steps} | "
                        f"Total: {total_loss.item():.4f} (KL: {kl_loss.item():.4f}, CKA: {cka_value:.4f})"
                    )

                if (
                    self.config.log_profile_scalars
                    and self.config.profiling_interval > 0
                    and step > 0
                    and step % self.config.profiling_interval == 0
                ):
                    for sub_module in unfrozen_modules:
                        for name, child in sub_module.named_modules():
                            if isinstance(child, MonarchLinear):
                                grad_norm = 0.0
                                if child.blk1.grad is not None:
                                    grad_norm += child.blk1.grad.norm(2).item()
                                if child.blk2.grad is not None:
                                    grad_norm += child.blk2.grad.norm(2).item()
                                log_scalars(self.writer, {f"Profile/{mod_path}_{name}_grad_norm": grad_norm}, step)

                    if mod_path not in ["lm_head", "model.embed_tokens"]:
                        teacher_act_prof = self.teacher_activations[mod_path]
                        student_act_prof = self.student_activations[mod_path][-1]
                        profile_activations(self.writer, student_act_prof, mod_path, step, teacher_act=teacher_act_prof)

            self.record_phase_throughput(
                step_idx + 1,
                mod_path,
                "phase2",
                self.config.phase2_steps,
                time.perf_counter() - phase2_start,
            )
            self.clear_training_buffers(unfrozen_paths)

            print(f"[Save] Saving trained unfrozen layers after {mod_path}")
            save_path = save_unfrozen_checkpoint(self.student_model, self.config.save_dir, step_idx, mod_path)
            print(f"[Save] Saved successfully to: {save_path}\n")
            self.writer.flush()
            self.run_validation(step_idx + 1, mod_path)
            self.clear_training_buffers(unfrozen_paths)

    def close(self):
        self.writer.flush()
        self.writer.close()
        self.data_iter = None
        self.dataloader = None
        self.prefetch_queue = deque()
        self.validation_examples = None
        self.teacher_model = None
        self.student_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
