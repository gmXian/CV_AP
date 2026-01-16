from typing import Callable, Tuple

import torch
import torch.nn.functional as F


def fgsm_linf(
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    images: torch.Tensor,
    labels: torch.Tensor,
    eps: float,
    clamp: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    images = images.detach()
    images.requires_grad_(True)
    logits = model_fn(images)
    loss = F.cross_entropy(logits, labels)
    grad = torch.autograd.grad(loss, images)[0]
    adv = images + eps * grad.sign()
    adv = adv.clamp(*clamp)
    delta = adv - images
    return adv.detach(), delta.detach()
