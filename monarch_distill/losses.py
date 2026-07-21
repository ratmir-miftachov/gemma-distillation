import torch
import torch.nn.functional as F


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


def compute_custom_kl_loss(student_logits, teacher_logits, loss_weights, chunk_tokens: int):
    vocab_size = student_logits.size(-1)
    s_flat = student_logits.reshape(-1, vocab_size)
    t_flat = teacher_logits.reshape(-1, vocab_size)
    w_flat = loss_weights.reshape(-1).float()
    chunk_tokens = max(1, int(chunk_tokens))

    total_loss = torch.zeros((), device=student_logits.device, dtype=torch.float32)
    for start in range(0, s_flat.size(0), chunk_tokens):
        end = min(start + chunk_tokens, s_flat.size(0))
        s_chunk = s_flat[start:end].float()
        with torch.no_grad():
            t_probs = F.softmax(t_flat[start:end].float(), dim=-1)
        loss_per_token = F.cross_entropy(s_chunk, t_probs, reduction="none")
        total_loss = total_loss + (loss_per_token * w_flat[start:end]).sum()

    return total_loss / w_flat.sum().clamp_min(1.0)


def compute_distillation_metric_sums(
    student_logits,
    teacher_logits,
    loss_weights,
    chunk_tokens: int,
):
    """Return differentiable CE plus fidelity statistics without full-vocab copies."""
    vocab_size = student_logits.size(-1)
    s_flat = student_logits.reshape(-1, vocab_size)
    t_flat = teacher_logits.reshape(-1, vocab_size)
    w_flat = loss_weights.reshape(-1).float()
    chunk_tokens = max(1, int(chunk_tokens))

    ce_sum = torch.zeros((), device=student_logits.device, dtype=torch.float32)
    teacher_entropy_sum = torch.zeros_like(ce_sum)
    agreement_count = torch.zeros_like(ce_sum)
    active_count = torch.zeros_like(ce_sum)

    for start in range(0, s_flat.size(0), chunk_tokens):
        end = min(start + chunk_tokens, s_flat.size(0))
        s_chunk = s_flat[start:end].float()
        weights = w_flat[start:end]
        with torch.no_grad():
            t_chunk = t_flat[start:end].float()
            t_log_probs = F.log_softmax(t_chunk, dim=-1)
            t_probs = t_log_probs.exp()
            teacher_entropy = -(t_probs * t_log_probs).sum(dim=-1)
            active = weights > 0
            agreement_count = agreement_count + (
                (s_chunk.detach().argmax(dim=-1) == t_chunk.argmax(dim=-1)) & active
            ).sum()
            active_count = active_count + active.sum()

        loss_per_token = F.cross_entropy(s_chunk, t_probs, reduction="none")
        ce_sum = ce_sum + (loss_per_token * weights).sum()
        teacher_entropy_sum = teacher_entropy_sum + (teacher_entropy * weights).sum()

    return {
        "ce_sum": ce_sum,
        "teacher_entropy_sum": teacher_entropy_sum,
        "weight_sum": w_flat.sum(),
        "agreement_count": agreement_count,
        "active_count": active_count,
    }


def normalize_distillation_metric_sums(metric_sums):
    weight_sum = metric_sums["weight_sum"].clamp_min(1.0)
    active_count = metric_sums["active_count"].clamp_min(1.0)
    cross_entropy = metric_sums["ce_sum"] / weight_sum
    teacher_entropy = metric_sums["teacher_entropy_sum"] / weight_sum
    return {
        "cross_entropy": cross_entropy,
        "true_kl": (cross_entropy - teacher_entropy).clamp_min(0.0),
        "top1_agreement": metric_sums["agreement_count"] / active_count,
        "active_weight": metric_sums["weight_sum"],
        "active_tokens": metric_sums["active_count"],
    }


def compute_validation_distill_loss_sum(student_logits, teacher_logits, loss_weights, chunk_tokens: int):
    vocab_size = student_logits.size(-1)
    s_flat = student_logits.reshape(-1, vocab_size)
    t_flat = teacher_logits.reshape(-1, vocab_size)
    w_flat = loss_weights.reshape(-1).float()
    chunk_tokens = max(1, int(chunk_tokens))

    total_loss = torch.zeros((), device=student_logits.device, dtype=torch.float32)
    for start in range(0, s_flat.size(0), chunk_tokens):
        end = min(start + chunk_tokens, s_flat.size(0))
        s_log_probs = F.log_softmax(s_flat[start:end].float(), dim=-1)
        t_probs = F.softmax(t_flat[start:end].float(), dim=-1)
        per_token_loss = -(t_probs * s_log_probs).sum(dim=-1)
        total_loss = total_loss + (per_token_loss * w_flat[start:end]).sum()

    return total_loss, w_flat.sum()


def compute_attn_kl_loss(student_attn, teacher_attn):
    s_log = torch.log(student_attn.clamp(min=1e-8))
    t_probs = teacher_attn.clamp(min=1e-8)
    kl = t_probs * (torch.log(t_probs) - s_log)
    return kl.sum(dim=-1).mean()


def calculate_shannon_entropy(singular_values):
    norm_sv = singular_values / (singular_values.sum() + 1e-8)
    return -torch.sum(norm_sv * torch.log(norm_sv + 1e-8))
