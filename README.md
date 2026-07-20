# Gemma Distillation

Minimal code for replacing Gemma MLP linear layers with Monarch-factorized layers and distilling the compressed student against the original teacher.

## What It Does

- Loads `google/gemma-4-E2B-it` as both teacher and student.
- Replaces selected student MLP layers with Monarch-linear modules.
- Phase 1 trains the newly replaced MLP locally with activation CKA loss.
- Phase 2 globally repairs the student with full-model logit KL distillation.
- Evaluates distillation loss on a fixed 64-example validation buffer at sequence length 512.
- Writes TensorBoard logs and layer checkpoints locally.

## Completed 35-Layer Model

The full all-MLP run is complete and preserved privately:

- Model: [`hexoy/gemma-4-e2b-monarch-35mlp`](https://huggingface.co/hexoy/gemma-4-e2b-monarch-35mlp)
- Compressed language-model MLP layers: `34` through `0`
- Final fixed-buffer eval512 distillation loss: `1.8860`
- Measured optimizer-phase wall time: `5.81 hours`
- Summed source/resume TensorBoard event spans: `5.89 hours`
- Exported parameter count: `3,682,268,704`
- Model weights revision: `f897353fca328b1cc5fd2e12d645773ca637f5f0`
- Model documentation revision: `91e4ba4fe25d481061bb4f515262a61a009cc5fb`
- Full checkpoints, canonical/raw TensorBoard, logs, and hashes: [`hexoy/gemma4-monarch-artifacts@beeee38d`](https://huggingface.co/datasets/hexoy/gemma4-monarch-artifacts/tree/beeee38d493c6bf5696057b12c0844e134b76dfc/runs/b8-all35mlp-400p1-800p2-seq512-projinit-p2lr3e4)

The run resumed from the cumulative four-layer `step_003` layer-31 checkpoint
preserved at artifact revision `ef7f583c3cc55d7473851da69e51cc3466ab3459`.
The eval512 value is a teacher-student distillation loss, not downstream-task accuracy.

### INT8 Dense-Weight Variant

The remaining supported dense linear weights were quantized privately with TorchAO
symmetric per-output-channel INT8 weight-only quantization:

- Model: [`hexoy/gemma-4-e2b-monarch-35mlp-int8`](https://huggingface.co/hexoy/gemma-4-e2b-monarch-35mlp-int8)
- Quantized standard linear modules: `420`
- Quantized linear weights: `779,419,648`
- Preserved BF16 Monarch factors: `210` tensors / `135,106,560` parameters
- Physical loaded weight footprint: `6.13 GiB` versus `6.86 GiB` in BF16
- Serialized repository size: `6.17 GiB` versus `6.89 GiB` in BF16
- TinyHellaSwag: GP-IRT `30.57%`, raw accuracy `21%`, batch `36`
- Immutable INT8 weight revision: `db56825e2e0de59115049d7109632b2f1ce80905`
- INT8 model-card revision: `4e3525b818b0a12af5c181ce1fe983877d0a1160`
- Results, comparisons, logs, and hashes: [`hexoy/gemma4-monarch-artifacts@59b1c450`](https://huggingface.co/datasets/hexoy/gemma4-monarch-artifacts/tree/59b1c450dbd6f339270a2b7942f0ac5c3acf12d3/benchmarks/int8/20260717T110806Z)

Against the same-environment BF16 35-layer control, INT8 changed raw accuracy by
-1 point with a paired bootstrap 95% interval of `[-3, 0]` points and McNemar
`p=1.0`. Prompted MNLI was not rerun for INT8.

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
- `max_modules=35`
- MLP compression only

`dense_projection` initializes each rectangular Monarch layer with the
minimum-Frobenius-error rank-one projection of the pretrained dense weight.
Set `monarch_init_method="identity_noise"` to use the original initializer.

Outputs:

- TensorBoard: `tensorboard_logs/...`
- Checkpoints: `monarch_checkpoints.../step_*/unfrozen_weights.pt`

Resume a preempted or deliberately continued run from the latest cumulative checkpoint:

```bash
python main.py \
  --resume-from-checkpoint monarch_checkpoints.../step_003_model_language_model_layers_31_mlp/unfrozen_weights.pt \
  --resume-start-module-index 4
```

After a resumed run, consolidate all scalar events into one canonical TensorBoard file:

```bash
python consolidate_tensorboard.py tensorboard_raw/RUN_NAME --output-dir tensorboard_logs/RUN_NAME
```

## Code Layout

- `main.py`: thin launcher that preserves the `python main.py` workflow.
- `monarch_distill/config.py`: typed experiment configuration.
- `monarch_distill/data.py`: streamed datasets, formatting, tokenization, validation-buffer construction.
- `monarch_distill/monarch.py`: Monarch layers and model replacement helpers.
- `monarch_distill/losses.py`: CKA, KL, entropy, and attention KL losses.
- `monarch_distill/validation.py`: fixed multi-length validation.
- `monarch_distill/trainer.py`: compression orchestration, resume flow, phase loops.
- `monarch_distill/io.py`: TensorBoard helpers, profiling logs, checkpoint saving.
- `consolidate_tensorboard.py`: canonical scalar-event consolidation for resumed runs.

## Export A Standalone Model

After all 35 cumulative checkpoints are complete, export the final checkpoint as a
standard sharded Hugging Face model with custom Monarch modeling code:

```bash
python export_hf.py \
  --checkpoint monarch_checkpoints_b8_all35mlp_400p1_800p2_seq512_projinit_p2lr3e4/step_034_model_language_model_layers_0_mlp/unfrozen_weights.pt \
  --layers 34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0 \
  --output-dir gemma-4-e2b-monarch-35mlp-export \
  --repo-id hexoy/gemma-4-e2b-monarch-35mlp \
  --upload
```

The exporter refuses non-cumulative or incorrectly shaped checkpoints and refuses to
upload if the destination repository is public. Verify a local directory or private
Hub model from a clean cache with:

```bash
python verify_hf_model.py hexoy/gemma-4-e2b-monarch-35mlp \
  --expected-layers 34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0
```

## Recover Compression Error With LoRA

The experimental recovery command freezes the released 35-layer Monarch model and
trains native rank-8 residual adapters on its 105 MLP projections:

```bash
pip install -r requirements-recovery.txt
python train_lora_recovery.py
```

Recovery checkpoints contain only LoRA tensors plus resumable trainer state. Resume
from a completed checkpoint directory with:

```bash
python train_lora_recovery.py \
  --resume-from-checkpoint lora_recovery_checkpoints_b8_all35mlp_r8/step_0000250
```

Export a selected adapter as a standalone Transformers model with:

```bash
python export_lora_hf.py \
  --adapter lora_recovery_checkpoints_b8_all35mlp_r8/best/adapter_model.safetensors \
  --output-dir gemma-4-e2b-monarch-35mlp-lora-r8-export
```

Rank zero remains the model-format default, so existing Monarch repositories and
checkpoints load without LoRA parameters.

## Quantize Dense Linear Weights

Install the separately pinned TorchAO dependency and convert all remaining standard
`nn.Linear` weights to symmetric per-channel INT8 while retaining Monarch factors and
non-linear parameters in BF16:

```bash
pip install -r requirements-quantization.txt
python quantize_hf.py \
  --model-id hexoy/gemma-4-e2b-monarch-35mlp \
  --revision f897353fca328b1cc5fd2e12d645773ca637f5f0 \
  --output-dir gemma-4-e2b-monarch-35mlp-int8 \
  --repo-id hexoy/gemma-4-e2b-monarch-35mlp-int8 \
  --upload
```

The command refuses to publish if the parameter count changes, any untied standard
linear weight remains unquantized, or any of the 210 trained Monarch factors leaves BF16.
Its manifest records the exact quantized module inventory, loaded footprint, package
versions, file sizes, and SHA-256 hashes.

## TensorBoard

```bash
tensorboard --logdir tensorboard_logs --host 127.0.0.1 --port 6006
```

## TinyHellaSwag Benchmark

Install the separately pinned benchmark environment:

```bash
pip install -r requirements-benchmark.txt
```

Run the official 100-example, 10-shot `tinyHellaswag` task. The evaluator uses
raw continuation prompts for both models and selects the appropriate Hugging
Face Auto class automatically:

```bash
python benchmark_tinyhellaswag.py --model google/gemma-4-E2B-it
python benchmark_tinyhellaswag.py --model hexoy/gemma-4-e2b-monarch-4mlp
```

Each run writes `result.json` and the underlying `lm_eval_results.json` beneath
`benchmark_results/tinyhellaswag/`. Compare two runs with:

```bash
python compare_tinyhellaswag.py \
  benchmark_results/tinyhellaswag/google-gemma-4-E2B-it/<timestamp>/result.json \
  benchmark_results/tinyhellaswag/hexoy-gemma-4-e2b-monarch-4mlp/<timestamp>/result.json \
  --output comparison.json
```

The comparison reports official GP-IRT and raw-accuracy deltas, item-level
disagreements, a paired bootstrap confidence interval, and exact McNemar test.

## Notes

- This repo intentionally excludes checkpoints, TensorBoard logs, caches, datasets, and experiment history.
- Store large artifacts in Hugging Face datasets or external storage, not in Git.
