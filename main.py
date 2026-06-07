import os
import gc
import math
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForImageTextToText
from datasets import load_dataset
from typing import Dict, List, Any, Optional

# ==========================================
# 1. Configuration
# ==========================================

COMPRESSION_CONFIG = {
    "run_label": "b8-2mlp-400p1-800p2-seq512-klnorm-p2lr3e4",
    "model_name": "google/gemma-4-E2B-it",
    "monarch_blocks_weights": 128,
    "monarch_blocks_head_embed": 64,           
    "max_seq_len": 512,
    "batch_size": 8,
    "phase1_steps": 400,
    "phase2_steps": 800,
    "lr_phase1": 5e-4,
    "lr_phase2": 3e-4,
    "think_token_weight": 0.2,
    "attn_kl_weight": 1.0,  # Gewichtung für die Attention-Map Distillation
    "max_epochs": 500,
    "profiling_interval": 100,
    "tensorboard_log_interval": 10,
    "tensorboard_flush_interval": 100,
    "prefetch_batches": 8,
    "tensorboard_log_dir": "./tensorboard_logs/b8-2mlp-400p1-800p2-seq512-klnorm-p2lr3e4",
    "save_dir": "./monarch_checkpoints_b8_2mlp_400p1_800p2_seq512_klnorm_p2lr3e4",
    "max_modules": 2,
    "resume_from_checkpoint": None,
    "resume_start_module_index": 0,
    "kl_chunk_tokens": 256,
    "force_exit_after_success": True,
    "validation_enabled": True,
    "validation_num_examples": 64,
    "validation_seed": 1234,
    "validation_storage_seq_len": 512,
    "validation_eval_lengths": [64, 128, 256, 512],
    "validation_batch_size": 4,
    "log_training_scalars": True,
    "log_profile_scalars": False,
    "compress_lm_head": False,
    "compress_embeddings": False,
    "compress_attention": False,
    "compress_mlp": True,
}

# ==========================================
# 2. Data Streaming & Processing
# ==========================================

class UnifiedDatasetStreamer(IterableDataset):
    def __init__(self, tokenizer, config: Dict[str, Any]):
        self.tokenizer = tokenizer
        self.config = config
        self.max_epochs = config["max_epochs"]
        
        self.dataset_configs = [
            {"path": "HuggingFaceH4/ultrachat_200k", "split": "train_sft", "type": "messages"},
            {"path": "nvidia/OpenScienceReasoning-2", "split": "train", "type": "input_output"},
            {"path": "nvidia/OpenCodeReasoning", "name": "split_0", "split": "split_0", "type": "input_output"},
            {"path": "KodCode/KodCode-V1", "split": "train", "type": "question_solution"},
            {"path": "Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b", "name": "stage1", "split": "train", "type": "input_output"},
            {"path": "Open-Orca/OpenOrca", "split": "train", "type": "system_question_response"},
            {"path": "allenai/sciq", "split": "train", "type": "question_support"}
        ]
        
        self.streams = []
        self.stream_epochs = [0] * len(self.dataset_configs)
        
        for dconf in self.dataset_configs:
            kwargs = {"streaming": True, "split": dconf["split"]}
            if "name" in dconf:
                kwargs["name"] = dconf["name"]
            stream = iter(load_dataset(dconf["path"], **kwargs))
            self.streams.append((dconf["type"], stream, dconf))

    def format_to_messages(self, data: dict, d_type: str) -> List[Dict[str, str]]:
        if d_type == "messages":
            return data["messages"]
        elif d_type == "input_output":
            return [{"role": "user", "content": data["input"]}, {"role": "assistant", "content": data["output"]}]
        elif d_type == "question_solution":
            return [{"role": "user", "content": data["question"]}, {"role": "assistant", "content": data["solution"]}]
        elif d_type == "conversations":
            return [{"role": "user", "content": data["conversations"][0]}, {"role": "assistant", "content": data["conversations"][1]}]
        elif d_type == "system_question_response":
            return [
                {"role": "system", "content": data["system_prompt"]},
                {"role": "user", "content": data["question"]},
                {"role": "assistant", "content": data["response"]}
            ]
        elif d_type == "question_support":
            return [{"role": "user", "content": data["question"]}, {"role": "assistant", "content": data["support"]}]
        return []

    def __iter__(self):
        active_streams = list(range(len(self.streams)))
        
        while active_streams:
            idx = torch.randint(0, len(active_streams), (1,)).item()
            stream_idx = active_streams[idx]
            d_type, stream, dconf = self.streams[stream_idx]
            
            try:
                data = next(stream)
                messages = self.format_to_messages(data, d_type)
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                
                encoded = self.tokenizer(
                    text, 
                    max_length=self.config["max_seq_len"], 
                    truncation=True, 
                    padding="max_length",
                    return_tensors="pt"
                )
                
                input_ids = encoded["input_ids"][0]
                loss_weights = torch.ones_like(input_ids.clone(), dtype=torch.float)
                
                in_think = False
                for i, token_id in enumerate(input_ids):
                    token_str = self.tokenizer.decode([token_id])
                    if "<think>" in token_str:
                        in_think = True
                    
                    if in_think:
                        loss_weights[i] = self.config["think_token_weight"]
                    else:
                        loss_weights[i] = 1.0 
                    
                    if "</think>" in token_str:
                        in_think = False
                
                yield {
                    "input_ids": input_ids,
                    "attention_mask": encoded["attention_mask"][0],
                    "loss_weights": loss_weights
                }
            except StopIteration:
                self.stream_epochs[stream_idx] += 1
                if self.stream_epochs[stream_idx] < self.max_epochs:
                    kwargs = {"streaming": True, "split": dconf["split"]}
                    if "name" in dconf:
                        kwargs["name"] = dconf["name"]
                    self.streams[stream_idx] = (d_type, iter(load_dataset(dconf["path"], **kwargs)), dconf)
                else:
                    print("DATA-STREAM WURDE ENTFERNT")
                    active_streams.remove(stream_idx)

# ==========================================
# 3. Monarch Architecture
# ==========================================

class MonarchLinear(nn.Module):
    def __init__(self, in_features, out_features, n_blocks, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.is_down_proj = out_features < in_features

        if not self.is_down_proj:
            self.n1 = n_blocks
            assert in_features % self.n1 == 0, f"in_features ({in_features}) must be divisible by n_blocks ({n_blocks})"
            self.n2 = in_features // self.n1
            
            assert out_features % self.n2 == 0, f"out_features ({out_features}) must be divisible by in_block_size ({self.n2})"
            self.n3 = out_features // self.n2
        else:
            self.n3 = n_blocks
            assert out_features % self.n3 == 0, f"out_features ({out_features}) must be divisible by n_blocks ({n_blocks})"
            self.n2 = out_features // self.n3
            
            assert in_features % self.n2 == 0, f"in_features ({in_features}) must be divisible by out_block_size ({self.n2})"
            self.n1 = in_features // self.n2

        self.blk1 = nn.Parameter(torch.empty(self.n1, self.n2, self.n2))
        self.blk2 = nn.Parameter(torch.empty(self.n2, self.n1, self.n3))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.blk1)
        nn.init.zeros_(self.blk2)
        
        for i in range(self.blk1.shape[0]):
            self.blk1.data[i].copy_(torch.eye(self.blk1.shape[1]))
            
        for i in range(self.blk2.shape[0]):
            min_dim = min(self.blk2.shape[1], self.blk2.shape[2])
            self.blk2.data[i, :min_dim, :min_dim].copy_(torch.eye(min_dim))
            
            if self.blk2.shape[1] != self.blk2.shape[2]:
                with torch.no_grad():
                    bound = 1.0 / (min_dim ** 0.5)
                    noise = torch.FloatTensor(self.blk2.shape[1], self.blk2.shape[2]).uniform_(-bound, bound)
                    mask = torch.ones(self.blk2.shape[1], self.blk2.shape[2])
                    mask[:min_dim, :min_dim] -= torch.eye(min_dim)
                    self.blk2.data[i] += (noise * mask).to(self.blk2.device)
            
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        orig_shape = x.shape
        x = x.contiguous().view(-1, self.n1, self.n2)
        x = torch.einsum('bij, ijk -> bik', x, self.blk1)
        x = x.transpose(1, 2).contiguous()
        x = torch.einsum('bij, ijk -> bik', x, self.blk2)
        x = x.transpose(1, 2).contiguous()
        x = x.view(*orig_shape[:-1], self.out_features)
        
        if self.bias is not None:
            x = x + self.bias
            
        return x

class MonarchEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, n_blocks):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.n1 = n_blocks
        assert num_embeddings % self.n1 == 0, "num_embeddings must be divisible by n_blocks"
        self.n2 = num_embeddings // self.n1  
        
        assert embedding_dim % self.n2 == 0, f"embedding_dim ({embedding_dim}) must be divisible by in_block_size ({self.n2})"
        self.n3 = embedding_dim // self.n2
        
        self.blk1 = nn.Parameter(torch.empty(self.n1, self.n2, self.n2))
        self.blk2 = nn.Parameter(torch.empty(self.n2, self.n1, self.n3))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.blk1, mean=0, std=1/math.sqrt(self.n2))
        nn.init.normal_(self.blk2, mean=0, std=1/math.sqrt(self.n1))

    def forward(self, x):
        blk1_flat = self.blk1.view(self.num_embeddings, self.n2)
        v1 = blk1_flat[x]  
        
        block_idx = x // self.n2  
        blk2_transposed = self.blk2.transpose(0, 1)
        w2_selected = blk2_transposed[block_idx]  
        
        out_matrix = v1.unsqueeze(-1) * w2_selected  
        out = out_matrix.transpose(-1, -2).contiguous() 
        out = out.view(*x.shape, self.embedding_dim)
        
        return out

# ==========================================
# 4. Profiling and Metrics
# ==========================================

def linear_cka(gram_x: torch.Tensor, gram_y: torch.Tensor) -> torch.Tensor:
    dot_prod = torch.trace(torch.matmul(gram_x, gram_y))
    norm_x = torch.sqrt(torch.trace(torch.matmul(gram_x, gram_x)))
    norm_y = torch.sqrt(torch.trace(torch.matmul(gram_y, gram_y)))
    return dot_prod / (norm_x * norm_y + 1e-8)

def compute_cka_loss(teacher_activations, student_activations):
    t = teacher_activations.reshape(-1, teacher_activations.size(-1))
    s = student_activations.reshape(-1, student_activations.size(-1))
    gram_t = torch.matmul(t, t.t())
    gram_s = torch.matmul(s, s.t())
    return 1.0 - linear_cka(gram_t, gram_s)

def calculate_shannon_entropy(singular_values):
    norm_sv = singular_values / (singular_values.sum() + 1e-8)
    return -torch.sum(norm_sv * torch.log(norm_sv + 1e-8))

def log_scalars(writer: SummaryWriter, metrics: Dict[str, Any], step: int):
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().item()
        writer.add_scalar(key, float(value), step)

def profile_activations(
    writer: SummaryWriter,
    student_act: torch.Tensor,
    name: str,
    step: int,
    teacher_act: Optional[torch.Tensor] = None,
):
    s_act_2d = student_act.reshape(-1, student_act.size(-1)).float()
    mean = s_act_2d.mean().item()
    var = s_act_2d.var().item()

    _, S, _ = torch.linalg.svd(s_act_2d, full_matrices=False)
    eff_rank = calculate_shannon_entropy(S).item()

    mse_sv = 0.0
    if teacher_act is not None:
        t_act_2d = teacher_act.reshape(-1, teacher_act.size(-1)).float()
        _, S_t, _ = torch.linalg.svd(t_act_2d, full_matrices=False)
        
        S_norm = S / (torch.norm(s_act_2d, p='fro') + 1e-8)
        S_t_norm = S_t / (torch.norm(t_act_2d, p='fro') + 1e-8)
        
        min_dim = min(S_norm.size(0), S_t_norm.size(0))
        mse_sv = F.mse_loss(S_norm[:min_dim], S_t_norm[:min_dim]).item()

    log_scalars(
        writer,
        {
            f"Profile/{name}_act_mean": mean,
            f"Profile/{name}_act_var": var,
            f"Profile/{name}_act_effective_rank": eff_rank,
            f"Profile/{name}_act_sv_mse": mse_sv,
        },
        step,
    )

# ==========================================
# 5. Reverse Sublayer-by-Sublayer Trainer
# ==========================================

class MonarchCompressor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.writer = SummaryWriter(config["tensorboard_log_dir"])
        print(f"[TensorBoard] Logging to: {config['tensorboard_log_dir']}")
        
        print(f"[Load] Loading tokenizer and Gemma model: {config['model_name']}")
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Wir erzwingen "eager" Attention, damit das HF Modell uns bei output_attentions=True die Matrizen zurückgibt
        self.teacher_model = AutoModelForImageTextToText.from_pretrained(
            config["model_name"], device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
            
        self.student_model = AutoModelForImageTextToText.from_pretrained(
            config["model_name"], device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
        
        self.dataset = UnifiedDatasetStreamer(self.tokenizer, config)
        self.dataloader = DataLoader(self.dataset, batch_size=config["batch_size"])
        self.data_iter = iter(self.dataloader)
        self.prefetch_queue = deque()
        self.validation_examples = self.build_validation_buffer()
    
    def _next_raw_batch(self):
        try:
            return next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            return next(self.data_iter)

    def get_batch(self):
        prefetch_batches = max(0, int(self.config.get("prefetch_batches", 0)))
        if prefetch_batches <= 0:
            return self._next_raw_batch()

        while len(self.prefetch_queue) < prefetch_batches:
            self.prefetch_queue.append(self._next_raw_batch())
        return self.prefetch_queue.popleft()

    def build_validation_buffer(self):
        if not self.config.get("validation_enabled", False):
            return None

        num_examples = int(self.config["validation_num_examples"])
        storage_seq_len = int(self.config.get("validation_storage_seq_len", self.config["max_seq_len"]))
        validation_batch_size = int(self.config.get("validation_batch_size", self.config["batch_size"]))
        print(
            f"[Validation] Building fixed validation buffer: {num_examples} examples "
            f"at storage_seq_len={storage_seq_len}"
        )
        rng_state = torch.random.get_rng_state()
        torch.manual_seed(self.config["validation_seed"])

        validation_config = dict(self.config)
        validation_config["max_seq_len"] = storage_seq_len
        validation_dataset = UnifiedDatasetStreamer(self.tokenizer, validation_config)
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

    def should_log_step(self, step: int, total_steps: int) -> bool:
        interval = int(self.config.get("tensorboard_log_interval", 1))
        return step == 0 or step == total_steps - 1 or (interval > 0 and step % interval == 0)

    def should_flush_step(self, step: int, total_steps: int) -> bool:
        interval = int(self.config.get("tensorboard_flush_interval", 0))
        return step == 0 or step == total_steps - 1 or (interval > 0 and step > 0 and step % interval == 0)

    def record_phase_throughput(self, module_index: int, mod_path: str, phase_name: str, steps: int, elapsed: float):
        elapsed = max(elapsed, 1e-9)
        examples = steps * int(self.config["batch_size"])
        token_slots = examples * int(self.config["max_seq_len"])
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
        for p in self.student_model.parameters():
            p.requires_grad = False

    def unfreeze_module(self, module):
        for p in module.parameters():
            p.requires_grad = True
            
    def replace_linear_with_monarch(self, module, blocks):
        """Ersetzt rekursiv alle nn.Linear Instanzen durch MonarchLinear und behält das originale Device bei."""
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                device = child.weight.device
                dtype = child.weight.dtype
                monarch_layer = MonarchLinear(child.in_features, child.out_features, blocks, bias=child.bias is not None)
                # Direkt beim Setzen auf das korrekte Device legen
                setattr(module, name, monarch_layer.to(device=device, dtype=dtype))
            else:
                self.replace_linear_with_monarch(child, blocks)
        return module

    def replace_with_monarch(self, module_path: str):
        blocks_weights = self.config["monarch_blocks_weights"]
        blocks_head = self.config["monarch_blocks_head_embed"]
        
        parent_path = '.'.join(module_path.split('.')[:-1])
        child_name = module_path.split('.')[-1]
        
        parent_module = self.student_model if parent_path == "" else self.student_model.get_submodule(parent_path)
        old_module = getattr(parent_module, child_name)
        
        if module_path == "lm_head":
            device, dtype = old_module.weight.device, old_module.weight.dtype
            new_head = MonarchLinear(old_module.in_features, old_module.out_features, blocks_head, bias=old_module.bias is not None)
            setattr(parent_module, child_name, new_head.to(device=device, dtype=dtype))
            return getattr(parent_module, child_name)
            
        elif "embed_tokens" in module_path:
            device, dtype = old_module.weight.device, old_module.weight.dtype
            new_embed = MonarchEmbedding(old_module.num_embeddings, old_module.embedding_dim, blocks_head)
            setattr(parent_module, child_name, new_embed.to(device=device, dtype=dtype))
            return getattr(parent_module, child_name)
            
        else:
            # Bei mlp und self_attn verändern wir die linearen Schichten intern im Subbaum
            self.replace_linear_with_monarch(old_module, blocks_weights)
            return old_module

    def compute_custom_kl_loss(self, student_logits, teacher_logits, loss_weights):
        vocab_size = student_logits.size(-1)
        s_flat = student_logits.reshape(-1, vocab_size)
        t_flat = teacher_logits.reshape(-1, vocab_size)
        w_flat = loss_weights.reshape(-1).float()
        chunk_tokens = max(1, int(self.config.get("kl_chunk_tokens", s_flat.size(0))))

        total_loss = torch.zeros((), device=student_logits.device, dtype=torch.float32)
        for start in range(0, s_flat.size(0), chunk_tokens):
            end = min(start + chunk_tokens, s_flat.size(0))
            s_chunk = s_flat[start:end].float()
            with torch.no_grad():
                t_probs = F.softmax(t_flat[start:end].float(), dim=-1)
            loss_per_token = F.cross_entropy(s_chunk, t_probs, reduction='none')
            total_loss = total_loss + (loss_per_token * w_flat[start:end]).sum()

        return total_loss / w_flat.sum().clamp_min(1.0)

    def compute_validation_distill_loss_sum(self, student_logits, teacher_logits, loss_weights):
        vocab_size = student_logits.size(-1)
        s_flat = student_logits.reshape(-1, vocab_size)
        t_flat = teacher_logits.reshape(-1, vocab_size)
        w_flat = loss_weights.reshape(-1).float()
        chunk_tokens = max(1, int(self.config.get("kl_chunk_tokens", s_flat.size(0))))

        total_loss = torch.zeros((), device=student_logits.device, dtype=torch.float32)
        for start in range(0, s_flat.size(0), chunk_tokens):
            end = min(start + chunk_tokens, s_flat.size(0))
            s_log_probs = F.log_softmax(s_flat[start:end].float(), dim=-1)
            t_probs = F.softmax(t_flat[start:end].float(), dim=-1)
            per_token_loss = -(t_probs * s_log_probs).sum(dim=-1)
            total_loss = total_loss + (per_token_loss * w_flat[start:end]).sum()

        return total_loss, w_flat.sum()

    def validation_eval_lengths(self):
        storage_len = self.validation_examples["input_ids"].shape[1]
        raw_lengths = self.config.get("validation_eval_lengths", [storage_len])
        lengths = []
        for raw_length in raw_lengths:
            length = int(raw_length)
            if length <= 0:
                raise ValueError(f"validation_eval_lengths must be positive, got {length}")
            if length > storage_len:
                raise ValueError(
                    f"validation eval length {length} exceeds storage length {storage_len}"
                )
            if length not in lengths:
                lengths.append(length)
        return lengths

    def run_validation(self, compressed_layer_count: int, mod_path: str):
        if not self.config.get("validation_enabled", False) or self.validation_examples is None:
            return

        teacher_was_training = self.teacher_model.training
        student_was_training = self.student_model.training
        self.teacher_model.eval()
        self.student_model.eval()
        self._in_validation = True

        num_examples = self.validation_examples["input_ids"].shape[0]
        batch_size = int(self.config.get("validation_batch_size", self.config["batch_size"]))
        eval_lengths = self.validation_eval_lengths()
        metrics = {
            eval_len: {"loss_sum": 0.0, "weight_sum": 0.0}
            for eval_len in eval_lengths
        }

        try:
            with torch.no_grad():
                for eval_len in eval_lengths:
                    for start in range(0, num_examples, batch_size):
                        end = min(start + batch_size, num_examples)
                        input_ids = self.validation_examples["input_ids"][start:end, :eval_len].to(self.teacher_model.device)
                        attention_mask = self.validation_examples["attention_mask"][start:end, :eval_len].to(self.teacher_model.device)
                        loss_weights = self.validation_examples["loss_weights"][start:end, :eval_len].to(self.teacher_model.device) * attention_mask.float()

                        teacher_out = self.teacher_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                        )
                        student_out = self.student_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                        )

                        loss_sum, weight_sum = self.compute_validation_distill_loss_sum(
                            student_out.logits,
                            teacher_out.logits,
                            loss_weights,
                        )
                        metrics[eval_len]["loss_sum"] += loss_sum.item()
                        metrics[eval_len]["weight_sum"] += weight_sum.item()
        finally:
            self._in_validation = False
            if teacher_was_training:
                self.teacher_model.train()
            if student_was_training:
                self.student_model.train()

        for eval_len in eval_lengths:
            total_loss_sum = metrics[eval_len]["loss_sum"]
            total_weight_sum = metrics[eval_len]["weight_sum"]
            distill_loss = total_loss_sum / max(total_weight_sum, 1.0)
            self.writer.add_scalar(
                f"ValidationLoss/eval{eval_len}_distill",
                distill_loss,
                compressed_layer_count,
            )
            self.writer.add_scalar(
                f"ValidationTokens/eval{eval_len}_active_weight",
                total_weight_sum,
                compressed_layer_count,
            )
            print(
                f"[ValidationLoss] eval{eval_len} after {compressed_layer_count} compressed layer(s), "
                f"latest layer {mod_path} | {num_examples} examples | "
                f"active_weight: {total_weight_sum:.1f} | distill_loss: {distill_loss:.4f}"
            )
        self.writer.flush()

    def compute_attn_kl_loss(self, student_attn, teacher_attn):
        """Stabile KL-Divergenz Berechnung für Attention Weights."""
        # s_attn, t_attn Formate: (Batch, Heads, Seq, Seq) - Werte sind Wahrscheinlichkeiten
        s_log = torch.log(student_attn.clamp(min=1e-8))
        t_probs = teacher_attn.clamp(min=1e-8)
        
        # Manuelle KL Divergence, Summe über die letzte Sequenz-Dimension und Mean über alles andere
        kl = t_probs * (torch.log(t_probs) - s_log)
        return kl.sum(dim=-1).mean()

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

        print(
            f"[Resume] Loaded {len(state_dict)} tensors for "
            f"{len(completed_paths)} completed module(s)"
        )
        self.clear_training_buffers(completed_paths)

    def run_compression(self):
        teacher_layers = self.teacher_model.model.language_model.layers
        num_layers = len(teacher_layers)
        print(f"[Model] Resolved Gemma backbone path: model.language_model.layers ({num_layers} layers)")
        
        modules_to_compress = []
        if self.config["compress_lm_head"]:
            modules_to_compress.append('lm_head')
        for i in reversed(range(num_layers)):
            if self.config["compress_mlp"]:
                modules_to_compress.append(f'model.language_model.layers.{i}.mlp')
            if self.config["compress_attention"]:
                modules_to_compress.append(f'model.language_model.layers.{i}.self_attn')
        if self.config["compress_embeddings"]:
            modules_to_compress.append('model.language_model.embed_tokens')

        modules_to_compress = modules_to_compress[: self.config["max_modules"]]
        print(f"[MLP Scaling] Selected modules: {modules_to_compress}")
        expected_modules = self.config["max_modules"]
        if len(modules_to_compress) != expected_modules:
            raise RuntimeError(
                f"MLP scaling run expected exactly {expected_modules} target modules, "
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

        resume_checkpoint = self.config.get("resume_from_checkpoint")
        resume_start_idx = int(self.config.get("resume_start_module_index", 0) or 0)
        if resume_checkpoint:
            if resume_start_idx <= 0 or resume_start_idx >= len(modules_to_compress):
                raise ValueError(
                    "resume_start_module_index must point to the first unfinished module"
                )
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
            
            optimizer_p1 = torch.optim.AdamW(student_module.parameters(), lr=self.config["lr_phase1"])
            
            # Phase 1: Local Distillation auf dem aktuellen Sublayer
            phase1_start = time.perf_counter()
            for step in range(self.config["phase1_steps"]):
                batch = self.get_batch()
                input_ids = batch["input_ids"].to(self.teacher_model.device)
                attention_mask = batch["attention_mask"].to(self.teacher_model.device)
                loss_weights = batch["loss_weights"].to(self.teacher_model.device) * attention_mask.float()
                # Clear every active student hook, not only the current module.
                # Previous layer hooks remain registered for later global adjustment;
                # if their lists are not cleared during Phase 1 they retain graphs
                # across steps and can exhaust VRAM on longer runs.
                for u_path in unfrozen_paths:
                    self.student_activations[u_path] = []

                with torch.no_grad():
                    teacher_out = self.teacher_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=False,
                        output_attentions=self.config["compress_attention"],
                        use_cache=False,
                    )
                
                student_out = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    output_attentions=self.config["compress_attention"],
                    use_cache=False,
                )
                
                if mod_path == "lm_head":
                    phase1_loss = self.compute_custom_kl_loss(student_out.logits, teacher_out.logits, loss_weights)
                    loss_name = "KL Loss"
                else:
                    t_act = self.teacher_activations[mod_path]
                    s_acts = self.student_activations[mod_path]
                    
                    cka_loss = sum([compute_cka_loss(t_act, s_act) for s_act in s_acts]) / len(s_acts)
                    phase1_loss = cka_loss
                    loss_name = "CKA Loss"
                    
                    # Bei self_attn fügen wir die Attention-Map Distillation hinzu
                    if "self_attn" in mod_path:
                        layer_idx = int(mod_path.split('.')[3])
                        t_attn_map = teacher_out.attentions[layer_idx]
                        s_attn_map = student_out.attentions[layer_idx]
                        
                        attn_kl = self.compute_attn_kl_loss(s_attn_map, t_attn_map)
                        phase1_loss += self.config["attn_kl_weight"] * attn_kl
                        loss_name = "CKA + Attn KL Loss"
                
                optimizer_p1.zero_grad(set_to_none=True)
                phase1_loss.backward()
                torch.nn.utils.clip_grad_norm_(student_module.parameters(), max_norm=1.0)
                optimizer_p1.step()
                
                if self.config.get("log_training_scalars", True) and self.should_log_step(step, self.config["phase1_steps"]):
                    log_scalars(self.writer, {f"Phase1/{mod_path}_loss": phase1_loss}, step)
                if self.should_flush_step(step, self.config["phase1_steps"]):
                    self.writer.flush()
                if step == 0 or step == self.config["phase1_steps"] - 1 or step % 10 == 0:
                    print(f"[Phase 1] {mod_path} | Step {step}/{self.config['phase1_steps']} | {loss_name}: {phase1_loss.item():.4f}")

            self.record_phase_throughput(
                step_idx + 1,
                mod_path,
                "phase1",
                self.config["phase1_steps"],
                time.perf_counter() - phase1_start,
            )
            self.clear_training_buffers(unfrozen_paths)

            if mod_path == "lm_head":
                print(f"[Save] Speichere lm_head nach Phase 1 (Sonderfall)...")
                save_dir = os.path.join(self.config.get("save_dir", "./monarch_checkpoints"), f"step_{step_idx:03d}_{mod_path.replace('.', '_')}")
                os.makedirs(save_dir, exist_ok=True)
                
                unfrozen_state_dict = {
                    name: param.detach().cpu() 
                    for name, param in self.student_model.named_parameters() 
                    if param.requires_grad
                }
                save_path = os.path.join(save_dir, "unfrozen_weights.pt")
                torch.save(unfrozen_state_dict, save_path)
                print(f"[Save] Erfolgreich gespeichert unter: {save_path}\n")
                self.writer.flush()
                self.run_validation(step_idx + 1, mod_path)
                continue

            # Phase 2: Global Adjustment
            for m in unfrozen_modules:
                self.unfreeze_module(m)
                
            p2_params = []
            for m in unfrozen_modules:
                p2_params.extend(list(m.parameters()))
                
            optimizer_p2 = torch.optim.AdamW(p2_params, lr=self.config["lr_phase2"])
            
            phase2_start = time.perf_counter()
            for step in range(self.config["phase2_steps"]):
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
                        output_attentions=self.config["compress_attention"],
                        use_cache=False,
                    )
                    teacher_logits = teacher_out.logits
                
                student_out = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=False,
                    output_attentions=self.config["compress_attention"],
                    use_cache=False,
                )
                student_logits = student_out.logits
                
                # 1. Global KL Loss auf den Logits
                kl_loss = self.compute_custom_kl_loss(student_logits, teacher_logits, loss_weights)
                
                # 2. Global CKA Loss & Attn KL Loss
                cka_loss = 0.0
                total_attn_kl = 0.0
                total_comparisons = 0
                
                for u_path in unfrozen_paths:
                    if u_path in ["lm_head", "model.embed_tokens"]:
                        continue
                        
                    t_act = self.teacher_activations[u_path]
                    for s_act in self.student_activations[u_path]:
                        cka_loss += compute_cka_loss(t_act, s_act)
                        total_comparisons += 1
                        
                    if "self_attn" in u_path:
                        layer_idx = int(u_path.split('.')[3])
                        t_attn_map = teacher_out.attentions[layer_idx]
                        s_attn_map = student_out.attentions[layer_idx]
                        total_attn_kl += self.compute_attn_kl_loss(s_attn_map, t_attn_map)

                if total_comparisons > 0:
                    cka_loss = cka_loss / total_comparisons
                
                total_loss = kl_loss + cka_loss + (self.config["attn_kl_weight"] * total_attn_kl)
                
                optimizer_p2.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(p2_params, max_norm=1.0)
                optimizer_p2.step()
                
                if self.config.get("log_training_scalars", True) and self.should_log_step(step, self.config["phase2_steps"]):
                    log_scalars(
                        self.writer,
                        {
                            f"Phase2/{mod_path}_total_loss": total_loss,
                            f"Phase2/{mod_path}_kl_loss": kl_loss,
                            f"Phase2/{mod_path}_cka_loss": cka_loss,
                        },
                        step,
                    )
                if self.should_flush_step(step, self.config["phase2_steps"]):
                    self.writer.flush()
                if step == 0 or step == self.config["phase2_steps"] - 1 or step % 10 == 0:
                    print(f"[Phase 2] {mod_path} | Step {step}/{self.config['phase2_steps']} | Total: {total_loss.item():.4f} (KL: {kl_loss.item():.4f}, CKA: {cka_loss if isinstance(cka_loss, float) else cka_loss.item():.4f})")
                        
                if (
                    self.config.get("log_profile_scalars", True)
                    and self.config["profiling_interval"] > 0
                    and step > 0
                    and step % self.config["profiling_interval"] == 0
                ):
                    for sub_mod in unfrozen_modules:
                        for name, child in sub_mod.named_modules():
                            if isinstance(child, MonarchLinear):
                                gnorm = 0.0
                                if child.blk1.grad is not None:
                                    gnorm += child.blk1.grad.norm(2).item()
                                if child.blk2.grad is not None:
                                    gnorm += child.blk2.grad.norm(2).item()
                                log_scalars(self.writer, {f"Profile/{mod_path}_{name}_grad_norm": gnorm}, step)

                    if mod_path not in ["lm_head", "model.embed_tokens"]:
                        t_act_prof = self.teacher_activations[mod_path]
                        s_act_prof = self.student_activations[mod_path][-1]
                        profile_activations(self.writer, s_act_prof, mod_path, step, teacher_act=t_act_prof)

            self.record_phase_throughput(
                step_idx + 1,
                mod_path,
                "phase2",
                self.config["phase2_steps"],
                time.perf_counter() - phase2_start,
            )
            self.clear_training_buffers(unfrozen_paths)
                        
            print(f"[Save] Speichere bisher trainierte (unfrozen) Layer nach {mod_path}...")
            save_dir = os.path.join(self.config.get("save_dir", "./monarch_checkpoints"), f"step_{step_idx:03d}_{mod_path.replace('.', '_')}")
            os.makedirs(save_dir, exist_ok=True)
            
            unfrozen_state_dict = {
                name: param.detach().cpu() 
                for name, param in self.student_model.named_parameters() 
                if param.requires_grad
            }
            
            save_path = os.path.join(save_dir, "unfrozen_weights.pt")
            torch.save(unfrozen_state_dict, save_path)
            print(f"[Save] Erfolgreich gespeichert unter: {save_path}\n")
            self.writer.flush()
            self.run_validation(step_idx + 1, mod_path)
            self.clear_training_buffers(unfrozen_paths)

    def close(self):
        self.writer.flush()
        self.writer.close()
        self.data_iter = None
        self.dataloader = None
        self.dataset = None
        self.prefetch_queue = deque()
        self.validation_examples = None
        self.teacher_model = None
        self.student_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    compressor = MonarchCompressor(COMPRESSION_CONFIG)
    success = False
    try:
        compressor.run_compression()
        success = True
    finally:
        compressor.close()
    if success and COMPRESSION_CONFIG.get("force_exit_after_success", False):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
