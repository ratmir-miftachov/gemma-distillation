from dataclasses import asdict, dataclass, field
from typing import Literal, Optional


@dataclass
class CompressionConfig:
    run_label: str = "b8-4mlp-400p1-800p2-seq512-projinit-p2lr3e4-h100spot"
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
    tensorboard_log_dir: str = "./tensorboard_logs/b8-4mlp-400p1-800p2-seq512-projinit-p2lr3e4-h100spot"
    save_dir: str = "./monarch_checkpoints_b8_4mlp_400p1_800p2_seq512_projinit_p2lr3e4_h100spot"
    max_modules: int = 4
    resume_from_checkpoint: Optional[str] = None
    resume_start_module_index: int = 0
    kl_chunk_tokens: int = 256
    force_exit_after_success: bool = True
    validation_enabled: bool = True
    validation_num_examples: int = 64
    validation_seed: int = 1234
    validation_storage_seq_len: int = 512
    validation_eval_lengths: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
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
