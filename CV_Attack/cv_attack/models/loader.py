from typing import Optional

import torch
from torchvision import models


TORCHVISION_MODELS = {
    name for name in dir(models) if callable(getattr(models, name)) and not name.startswith("_")
}


def load_model(
    model_name: str,
    num_classes: int,
    device: torch.device,
    ckpt: Optional[str] = None,
    pretrained: bool = False,
) -> torch.nn.Module:
    if model_name not in TORCHVISION_MODELS:
        raise ValueError(f"Unsupported model name: {model_name}")

    model_fn = getattr(models, model_name)
    kwargs = {}
    if "num_classes" in model_fn.__code__.co_varnames:
        kwargs["num_classes"] = num_classes
    if pretrained:
        if num_classes != 1000:
            print("Warning: pretrained weights require 1000 classes. Loading without pretrained weights.")
        else:
            try:
                kwargs["weights"] = "DEFAULT"
            except TypeError:
                kwargs["pretrained"] = True

    model = model_fn(**kwargs)

    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)

    model.eval()
    model.to(device)
    return model
