from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_consistency_loss(logits_clean: torch.Tensor, logits_aug: torch.Tensor) -> torch.Tensor:
    """KL(p(clean) || p(augmented)) using detached clean probabilities as target."""
    p_clean = F.softmax(logits_clean.detach(), dim=1)
    log_p_aug = F.log_softmax(logits_aug, dim=1)
    return F.kl_div(log_p_aug, p_clean, reduction="batchmean")


def symmetric_kl_loss(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    p_a = F.softmax(logits_a.detach(), dim=1)
    p_b = F.softmax(logits_b.detach(), dim=1)
    log_a = F.log_softmax(logits_a, dim=1)
    log_b = F.log_softmax(logits_b, dim=1)
    return 0.5 * (F.kl_div(log_b, p_a, reduction="batchmean") + F.kl_div(log_a, p_b, reduction="batchmean"))


def jsd_loss(logits_clean: torch.Tensor, logits_aug1: torch.Tensor, logits_aug2: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence used by AugMix."""
    p_clean = F.softmax(logits_clean, dim=1)
    p_aug1 = F.softmax(logits_aug1, dim=1)
    p_aug2 = F.softmax(logits_aug2, dim=1)
    mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3.0, min=1e-7, max=1.0)
    log_mixture = torch.log(mixture)
    return (
        F.kl_div(log_mixture, p_clean, reduction="batchmean")
        + F.kl_div(log_mixture, p_aug1, reduction="batchmean")
        + F.kl_div(log_mixture, p_aug2, reduction="batchmean")
    ) / 3.0


def linear_warmup(epoch: int, warmup_epochs: int, max_value: float) -> float:
    if warmup_epochs <= 0:
        return max_value
    return max_value * min(1.0, float(epoch + 1) / float(warmup_epochs))
