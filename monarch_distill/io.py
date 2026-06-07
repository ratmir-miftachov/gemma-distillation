import os
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from .losses import calculate_shannon_entropy


def log_scalars(writer: Any, metrics: Dict[str, Any], step: int):
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().item()
        writer.add_scalar(key, float(value), step)


def profile_activations(
    writer: Any,
    student_act: torch.Tensor,
    name: str,
    step: int,
    teacher_act: Optional[torch.Tensor] = None,
):
    s_act_2d = student_act.reshape(-1, student_act.size(-1)).float()
    mean = s_act_2d.mean().item()
    var = s_act_2d.var().item()

    _, s_values, _ = torch.linalg.svd(s_act_2d, full_matrices=False)
    eff_rank = calculate_shannon_entropy(s_values).item()

    mse_sv = 0.0
    if teacher_act is not None:
        t_act_2d = teacher_act.reshape(-1, teacher_act.size(-1)).float()
        _, t_values, _ = torch.linalg.svd(t_act_2d, full_matrices=False)

        s_norm = s_values / (torch.norm(s_act_2d, p="fro") + 1e-8)
        t_norm = t_values / (torch.norm(t_act_2d, p="fro") + 1e-8)

        min_dim = min(s_norm.size(0), t_norm.size(0))
        mse_sv = F.mse_loss(s_norm[:min_dim], t_norm[:min_dim]).item()

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


def save_unfrozen_checkpoint(student_model, save_root: str, step_idx: int, module_path: str):
    save_dir = os.path.join(save_root, f"step_{step_idx:03d}_{module_path.replace('.', '_')}")
    os.makedirs(save_dir, exist_ok=True)

    unfrozen_state_dict = {
        name: param.detach().cpu()
        for name, param in student_model.named_parameters()
        if param.requires_grad
    }

    save_path = os.path.join(save_dir, "unfrozen_weights.pt")
    torch.save(unfrozen_state_dict, save_path)
    return save_path
