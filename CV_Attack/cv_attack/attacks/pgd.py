from typing import Callable, Tuple

import torch
import torch.nn.functional as F


def pgd_linf(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    random_start: bool,
    clamp: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    if steps <= 0:
        return images.clone(), torch.zeros_like(images)

    adv = images.detach().clone()
    if random_start:
        adv = adv + torch.empty_like(adv).uniform_(-eps, eps)
        adv = adv.clamp(*clamp)

    for _ in range(steps):
        adv.requires_grad_(True)
        logits = model_fn(adv)
        loss = F.cross_entropy(logits, labels)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.max(torch.min(adv, images + eps), images - eps)
        adv = adv.clamp(*clamp)

    delta = adv - images
    return adv.detach(), delta.detach()
