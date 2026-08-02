# Nebius Experiment Onboarding

Use Nebius only for temporary compute. GitHub stores code; Hugging Face stores
models and experiment artifacts.

## 1. Access

- Use your own Nebius login, SSH key, and Hugging Face token.
- Join only the project-scoped `monarch-experimenters` group.
- Never share private keys, tokens, or billing credentials.
- Create an experiment branch before changing training code.

## 2. Create the Spot VM

In the Nebius console, create:

- Platform: RTX PRO 6000 (`gpu-rtx6000`)
- Preset: `1gpu-24vcpu-218gb`
- Image: Ubuntu 24.04 with CUDA 13
- Priority: preemptible/spot; on preemption: stop
- Public IP: dynamic
- Disk: 80 GiB SSD for trials, 120 GiB for full distillation
- SSH user: `ubuntu`, using your public key

Do not switch GPU families without agreeing on the experiment change.

## 3. Prepare the VM

```bash
ssh -i PATH_TO_YOUR_KEY ubuntu@PUBLIC_IP
git clone https://github.com/ratmir-miftachov/gemma-distillation.git
cd gemma-distillation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/base.txt
```

Store your Hugging Face token in
`~/.config/nebius-gemma/hf_read_token`, set mode `600`, and never commit it.

Before training, run:

```bash
nvidia-smi
python -m py_compile main.py monarch_distill/*.py scripts/*.py
PYTHONPATH=. python -m unittest discover -s tests
```

## 4. Run and Monitor

Commit and push the exact configuration before launch. Use unique run, log,
checkpoint, TensorBoard, and `tmux` names.

```bash
tmux new -s RUN_NAME
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python main.py 2>&1 | tee RUN_NAME.log
```

Monitor with `tail -f RUN_NAME.log`, `nvidia-smi`, checkpoint counts, and
TensorBoard. If the spot VM is preempted, restart the same VM, resolve its new
IP, and resume from the newest complete checkpoint without changing settings.

## 5. Preserve Results

Before cleanup, verify and upload:

- Code and configuration to GitHub
- Standalone model to its Hugging Face model repository
- Checkpoints, logs, TensorBoard, results, environment metadata, and a SHA-256
  manifest to the private artifact dataset

Re-download representative files and verify hashes. Never treat the VM disk as
the only copy.

## 6. Delete Resources

After every preservation check passes:

1. Delete the exact spot VM by name and ID.
2. Confirm its managed boot disk was deleted.
3. Delete the disk explicitly if it remains unattached.
4. Verify both are absent from the instance and disk listings.

Stopping the VM ends GPU charges, but the disk continues to incur storage costs.
Delete both when the experiment is complete, and never touch unrelated resources.
