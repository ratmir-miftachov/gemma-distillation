import argparse

from monarch_distill.config import default_lora_recovery_config
from monarch_distill.recovery import LoRARecoveryTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recover a compressed Monarch model with native LoRA adapters"
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        help="Checkpoint directory containing adapter_model.safetensors and trainer_state.pt",
    )
    parser.add_argument("--steps", type=int, help="Override the total recovery steps")
    return parser.parse_args()


def main():
    args = parse_args()
    config = default_lora_recovery_config()
    config.resume_from_checkpoint = args.resume_from_checkpoint
    if args.steps is not None:
        if args.steps <= 0:
            raise ValueError("--steps must be positive")
        config.recovery_steps = args.steps

    trainer = LoRARecoveryTrainer(config)
    try:
        trainer.run()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
