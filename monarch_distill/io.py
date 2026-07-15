import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from .losses import calculate_shannon_entropy


@dataclass(frozen=True)
class ScalarEventRecord:
    tag: str
    step: int
    wall_time: float
    value: float
    source_order: int


def discover_tensorboard_event_files(inputs):
    event_files = []
    for raw_path in inputs:
        path = Path(raw_path)
        if path.is_file() and path.name.startswith("events.out.tfevents."):
            event_files.append(path)
        elif path.is_dir():
            event_files.extend(sorted(path.rglob("events.out.tfevents.*")))
        else:
            raise FileNotFoundError(f"TensorBoard input does not exist or is not an event file: {path}")

    unique_files = []
    seen = set()
    for path in event_files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_files.append(path)
    if not unique_files:
        raise FileNotFoundError("no TensorBoard event files were found")
    return unique_files


def read_scalar_events(event_files):
    from tensorboard.backend.event_processing import event_accumulator
    from tensorboard.util import tensor_util

    records = {}
    for source_order, event_file in enumerate(event_files):
        accumulator = event_accumulator.EventAccumulator(
            str(event_file),
            size_guidance={
                event_accumulator.SCALARS: 0,
                event_accumulator.TENSORS: 0,
            },
        )
        accumulator.Reload()
        tags = accumulator.Tags()

        for tag in tags.get("scalars", []):
            for event in accumulator.Scalars(tag):
                record = ScalarEventRecord(tag, int(event.step), float(event.wall_time), float(event.value), source_order)
                key = (record.tag, record.step)
                previous = records.get(key)
                if previous is None or (record.wall_time, record.source_order) >= (
                    previous.wall_time,
                    previous.source_order,
                ):
                    records[key] = record

        for tag in tags.get("tensors", []):
            for event in accumulator.Tensors(tag):
                value = tensor_util.make_ndarray(event.tensor_proto)
                if value.size != 1:
                    continue
                record = ScalarEventRecord(
                    tag,
                    int(event.step),
                    float(event.wall_time),
                    float(value.reshape(-1)[0]),
                    source_order,
                )
                key = (record.tag, record.step)
                previous = records.get(key)
                if previous is None or (record.wall_time, record.source_order) >= (
                    previous.wall_time,
                    previous.source_order,
                ):
                    records[key] = record
    return records


def consolidate_tensorboard_scalars(inputs, output_dir):
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    event_files = discover_tensorboard_event_files(inputs)
    records = read_scalar_events(event_files)
    if not records:
        raise RuntimeError("TensorBoard inputs contain no scalar summaries")

    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"TensorBoard output directory is not empty: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    writer = EventFileWriter(str(output_path), filename_suffix=".canonical")
    try:
        for record in sorted(records.values(), key=lambda item: (item.wall_time, item.tag, item.step)):
            summary = Summary(value=[Summary.Value(tag=record.tag, simple_value=record.value)])
            writer.add_event(Event(wall_time=record.wall_time, step=record.step, summary=summary))
        writer.flush()
    finally:
        writer.close()

    canonical_files = sorted(output_path.glob("events.out.tfevents.*"))
    if len(canonical_files) != 1:
        raise RuntimeError(f"expected one canonical TensorBoard event file, found {len(canonical_files)}")
    return {
        "event_file": str(canonical_files[0]),
        "input_files": [str(path) for path in event_files],
        "scalar_count": len(records),
        "tag_count": len({record.tag for record in records.values()}),
    }


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
