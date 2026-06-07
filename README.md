# Gemma Monarch Distillation

Minimal code for replacing Gemma MLP linear layers with Monarch-factorized layers and distilling the compressed student against the original teacher.

## What It Does

- Loads `google/gemma-4-E2B-it` as both teacher and student.
- Replaces selected student MLP layers with Monarch-linear modules.
- Phase 1 trains the newly replaced MLP locally with activation CKA loss.
- Phase 2 globally repairs the student with full-model logit KL distillation.
- Evaluates distillation loss on a fixed validation buffer at sequence lengths 64, 128, 256, and 512.
- Writes TensorBoard logs and layer checkpoints locally.

## Setup

Use a GPU machine with enough VRAM for Gemma 4 E2B.

```bash
python3 -m venv mlenv
source mlenv/bin/activate

# Install PyTorch for your CUDA version first:
# https://pytorch.org/get-started/locally/

pip install -r requirements.txt
```

Authenticate with Hugging Face before running:

```bash
export HF_TOKEN="$(cat ~/.config/nebius-gemma/hf_read_token)"
```

The token must have access to `google/gemma-4-E2B-it`.

## Run Compression

Edit `CompressionConfig` defaults in `monarch_distill/config.py`, then run:

```bash
python main.py
```

Current default config:

- `batch_size=8`
- `max_seq_len=512`
- `phase1_steps=400`
- `phase2_steps=800`
- `lr_phase1=5e-4`
- `lr_phase2=3e-4`
- `max_modules=2`
- MLP compression only

Outputs:

- TensorBoard: `tensorboard_logs/...`
- Checkpoints: `monarch_checkpoints.../step_*/unfrozen_weights.pt`

## Code Layout

- `main.py`: thin launcher that preserves the `python main.py` workflow.
- `monarch_distill/config.py`: typed experiment configuration.
- `monarch_distill/data.py`: streamed datasets, formatting, tokenization, validation-buffer construction.
- `monarch_distill/monarch.py`: Monarch layers and model replacement helpers.
- `monarch_distill/losses.py`: CKA, KL, entropy, and attention KL losses.
- `monarch_distill/validation.py`: fixed multi-length validation.
- `monarch_distill/trainer.py`: compression orchestration, resume flow, phase loops.
- `monarch_distill/io.py`: TensorBoard helpers, profiling logs, checkpoint saving.

## TensorBoard

```bash
tensorboard --logdir tensorboard_logs --host 127.0.0.1 --port 6006
```

## Notes

- This repo intentionally excludes checkpoints, TensorBoard logs, caches, datasets, and experiment history.
- Store large artifacts in Hugging Face datasets or external storage, not in Git.
