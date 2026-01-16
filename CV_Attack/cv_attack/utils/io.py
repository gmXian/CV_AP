import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import torch
from torchvision.transforms.functional import to_pil_image


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_tensor_image(img_tensor: torch.Tensor, path: Path) -> None:
    img = img_tensor.detach().cpu().clamp(0, 1)
    if img.dim() == 3:
        pil_img = to_pil_image(img)
    else:
        raise ValueError(f"Expected CHW tensor, got shape {img.shape}")
    pil_img.save(path)


def write_manifest_csv(manifest_path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_results_json(path: Path, payload: Dict[str, object]) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def build_save_dirs(save_dir: Path, save_delta: bool) -> Dict[str, Path]:
    clean_dir = save_dir / "clean"
    adv_dir = save_dir / "adv"
    delta_dir = save_dir / "delta" if save_delta else None
    ensure_dir(save_dir)
    ensure_dir(clean_dir)
    ensure_dir(adv_dir)
    if delta_dir is not None:
        ensure_dir(delta_dir)
    return {"clean": clean_dir, "adv": adv_dir, "delta": delta_dir}


def format_image_name(index: int) -> str:
    return f"{index:06d}.png"


def maybe_save_images(
    indices: Sequence[int],
    clean: torch.Tensor,
    adv: torch.Tensor,
    delta: Optional[torch.Tensor],
    save_dirs: Dict[str, Path],
) -> Dict[int, Dict[str, Optional[str]]]:
    saved_paths: Dict[int, Dict[str, Optional[str]]] = {}
    for i, idx in enumerate(indices):
        clean_name = format_image_name(idx)
        adv_name = format_image_name(idx)
        clean_path = save_dirs["clean"] / clean_name
        adv_path = save_dirs["adv"] / adv_name
        save_tensor_image(clean[i], clean_path)
        save_tensor_image(adv[i], adv_path)
        delta_path: Optional[Path] = None
        if delta is not None and save_dirs["delta"] is not None:
            delta_path = save_dirs["delta"] / clean_name
            delta_vis = (delta[i].abs() / delta[i].abs().max().clamp(min=1e-12)).clamp(0, 1)
            save_tensor_image(delta_vis, delta_path)
        saved_paths[idx] = {
            "clean_path": str(clean_path),
            "adv_path": str(adv_path),
            "delta_path": str(delta_path) if delta_path else None,
        }
    return saved_paths
