from dataclasses import dataclass

import torch


@dataclass
class Metrics:
    clean_acc: float
    adv_acc: float
    asr: float


def compute_metrics(clean_correct: int, adv_correct: int, total: int) -> Metrics:
    clean_acc = clean_correct / total if total else 0.0
    adv_acc = adv_correct / total if total else 0.0
    asr = (clean_correct - adv_correct) / clean_correct if clean_correct else 0.0
    return Metrics(clean_acc=clean_acc, adv_acc=adv_acc, asr=asr)


def count_correct(logits: torch.Tensor, labels: torch.Tensor) -> int:
    preds = logits.argmax(dim=1)
    return int((preds == labels).sum().item())
