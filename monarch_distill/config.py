from dataclasses import asdict, dataclass, field
from typing import Literal, Optional


@dataclass
class CompressionConfig:
    run_label: str = "b8-all35mlp-400p1-800p2-seq512-projinit-p2lr3e4"
    model_name: str = "google/gemma-4-E2B-it"
    monarch_blocks_weights: int = 128
    monarch_blocks_head_embed: int = 64
    monarch_init_method: Literal["identity_noise", "dense_projection"] = "dense_projection"
    max_seq_len: int = 512
    batch_size: int = 8
    phase1_steps: int = 400
    phase2_steps: int = 800
    lr_phase1: float = 5e-4
    lr_phase2: float = 3e-4
    think_token_weight: float = 0.2
    attn_kl_weight: float = 1.0
    max_epochs: int = 500
    profiling_interval: int = 100
    tensorboard_log_interval: int = 10
    tensorboard_flush_interval: int = 100
    prefetch_batches: int = 8
    tensorboard_log_dir: str = "./tensorboard_logs/b8-all35mlp-400p1-800p2-seq512-projinit-p2lr3e4"
    save_dir: str = "./monarch_checkpoints_b8_all35mlp_400p1_800p2_seq512_projinit_p2lr3e4"
    max_modules: int = 35
    resume_from_checkpoint: Optional[str] = None
    resume_start_module_index: int = 0
    kl_chunk_tokens: int = 256
    force_exit_after_success: bool = True
    validation_enabled: bool = True
    validation_num_examples: int = 64
    validation_seed: int = 1234
    validation_storage_seq_len: int = 512
    validation_eval_lengths: list[int] = field(default_factory=lambda: [512])
    validation_batch_size: int = 4
    log_training_scalars: bool = True
    log_profile_scalars: bool = False
    compress_lm_head: bool = False
    compress_embeddings: bool = False
    compress_attention: bool = False
    compress_mlp: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def default_config() -> CompressionConfig:
    return CompressionConfig()


@dataclass
class LoRARecoveryConfig(CompressionConfig):
    run_label: str = "b8-all35mlp-lora-r8-recovery"
    teacher_model_name: str = "google/gemma-4-E2B-it"
    teacher_revision: str = "9dbdf8a839e4e9e0eb56ed80cc8886661d3817cf"
    student_model_name: str = "hexoy/gemma-4-e2b-monarch-35mlp"
    student_revision: str = "f897353fca328b1cc5fd2e12d645773ca637f5f0"
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    recovery_steps: int = 2000
    recovery_lr: float = 3e-4
    recovery_weight_decay: float = 0.0
    recovery_grad_clip: float = 1.0
    recovery_seed: int = 1234
    training_data_seed: int = 5678
    checkpoint_interval: int = 250
    validation_interval: int = 250
    tensorboard_log_dir: str = "./tensorboard_logs/b8-all35mlp-lora-r8-recovery"
    save_dir: str = "./lora_recovery_checkpoints_b8_all35mlp_r8"
    resume_from_checkpoint: Optional[str] = None
    prefetch_batches: int = 0


def default_lora_recovery_config() -> LoRARecoveryConfig:
    return LoRARecoveryConfig()
