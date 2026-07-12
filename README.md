# Gemma Distillation

Minimal code for replacing Gemma MLP linear layers with Monarch-factorized layers and distilling the compressed student against the original teacher.

## What It Does

- Loads `google/gemma-4-E2B-it` as both teacher and student.
- Replaces selected student MLP layers with Monarch-linear modules.
- Phase 1 trains the newly replaced MLP locally with activation CKA loss.
- Phase 2 globally repairs the student with full-model logit KL distillation.
- Evaluates distillation loss on a fixed 64-example validation buffer at sequence length 512.
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
- `monarch_init_method="dense_projection"`
- `max_modules=8`
- MLP compression only

`dense_projection` initializes each rectangular Monarch layer with the
minimum-Frobenius-error rank-one projection of the pretrained dense weight.
Set `monarch_init_method="identity_noise"` to use the original initializer.

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

## Export A Standalone Model

After all eight cumulative checkpoints are complete, export the final checkpoint as a
standard sharded Hugging Face model with custom Monarch modeling code:

```bash
python export_hf.py \
  --checkpoint monarch_checkpoints_b8_8mlp_400p1_800p2_seq512_projinit_p2lr3e4_h100spot/step_007_model_language_model_layers_27_mlp/unfrozen_weights.pt \
  --output-dir gemma-4-e2b-monarch-8mlp-export \
  --repo-id hexoy/gemma-4-e2b-monarch-8mlp \
  --upload
```

The exporter refuses non-cumulative or incorrectly shaped checkpoints and refuses to
upload if the destination repository is public. Verify a local directory or private
Hub model from a clean cache with:

```bash
python verify_hf_model.py hexoy/gemma-4-e2b-monarch-8mlp
```

## TensorBoard

```bash
tensorboard --logdir tensorboard_logs --host 127.0.0.1 --port 6006
```

## Notes

- This repo intentionally excludes checkpoints, TensorBoard logs, caches, datasets, and experiment history.
- Store large artifacts in Hugging Face datasets or external storage, not in Git.
