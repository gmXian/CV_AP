from typing import Callable

from cv_attack.attacks.fgsm import fgsm_linf
from cv_attack.attacks.pgd import pgd_linf


def get_attack(name: str) -> Callable:
    name = name.lower()
    if name == "pgd":
        return pgd_linf
    if name == "fgsm":
        return fgsm_linf
    raise ValueError(f"Unsupported attack: {name}")
