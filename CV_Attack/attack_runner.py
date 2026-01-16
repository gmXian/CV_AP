import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from tqdm import tqdm

from cv_attack.attacks.registry import get_attack
from cv_attack.datasets.loader import DatasetSpec, load_dataset, normalize
from cv_attack.eval.metrics import compute_metrics, count_correct
from cv_attack.models.loader import load_model
from cv_attack.utils.device import get_device
from cv_attack.utils.io import (
    build_save_dirs,
    ensure_dir,
    maybe_save_images,
    write_manifest_csv,
    write_results_json,
)
from cv_attack.utils.seed import set_seed


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    config_path = Path(__file__).parent / "configs" / "default.yaml"

    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=str, default=str(config_path))
    base_args, _ = base_parser.parse_known_args()
    loaded_config = load_config(Path(base_args.config))

    parser = argparse.ArgumentParser(description="CV_Attack: adversarial attack pipeline")
    parser.add_argument("--config", type=str, default=str(base_args.config), help="Path to config file")
    parser.add_argument("--model", type=str, default=loaded_config["model"])
    parser.add_argument("--dataset", type=str, default=loaded_config["dataset"])
    parser.add_argument("--data_root", type=str, default=loaded_config["data_root"])
    parser.add_argument("--batch_size", type=int, default=loaded_config["batch_size"])
    parser.add_argument("--num_workers", type=int, default=loaded_config["num_workers"])
    parser.add_argument("--attack", type=str, default=loaded_config["attack"])
    parser.add_argument("--eps", type=float, default=loaded_config["eps"])
    parser.add_argument("--alpha", type=float, default=loaded_config["alpha"])
    parser.add_argument("--steps", type=int, default=loaded_config["steps"])
    parser.add_argument("--random_start", action=argparse.BooleanOptionalAction, default=loaded_config["random_start"])
    parser.add_argument("--norm", type=str, default=loaded_config["norm"])
    parser.add_argument("--save_dir", type=str, default=loaded_config["save_dir"])
    parser.add_argument("--save_images", action=argparse.BooleanOptionalAction, default=loaded_config["save_images"])
    parser.add_argument("--save_delta", action=argparse.BooleanOptionalAction, default=loaded_config["save_delta"])
    parser.add_argument("--limit", type=int, default=loaded_config["limit"])
    parser.add_argument("--seed", type=int, default=loaded_config["seed"])
    parser.add_argument("--device", type=str, default=loaded_config["device"])
    parser.add_argument("--ckpt", type=str, default=loaded_config["ckpt"])
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    dataset_spec: DatasetSpec = load_dataset(
        name=args.dataset,
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit,
    )

    model = load_model(
        model_name=args.model,
        num_classes=dataset_spec.num_classes,
        device=device,
        ckpt=args.ckpt,
        pretrained=args.pretrained,
    )

    def model_fn(x: torch.Tensor) -> torch.Tensor:
        return model(normalize(x, dataset_spec.mean, dataset_spec.std))

    attack_fn = get_attack(args.attack)
    save_dir = Path(args.save_dir)
    if args.save_images:
        save_dirs = build_save_dirs(save_dir, save_delta=args.save_delta)
    else:
        ensure_dir(save_dir)
        save_dirs = {}

    total = 0
    clean_correct = 0
    adv_correct = 0
    manifest_rows = []

    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(tqdm(dataset_spec.loader, desc="Attack")):
        batch_size = images.size(0)
        indices = list(range(total, total + batch_size))
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            clean_logits = model_fn(images)
        clean_preds = clean_logits.argmax(dim=1)
        clean_correct += count_correct(clean_logits, labels)

        if args.norm != "linf":
            raise ValueError("Only linf norm is supported in this runner.")

        if args.attack == "pgd":
            adv_images, delta = attack_fn(
                model_fn,
                images,
                labels,
                eps=args.eps,
                alpha=args.alpha,
                steps=args.steps,
                random_start=args.random_start,
            )
        elif args.attack == "fgsm":
            adv_images, delta = attack_fn(
                model_fn,
                images,
                labels,
                eps=args.eps,
            )
        else:
            raise ValueError(f"Unsupported attack: {args.attack}")

        with torch.no_grad():
            adv_logits = model_fn(adv_images)
        adv_preds = adv_logits.argmax(dim=1)
        adv_correct += count_correct(adv_logits, labels)

        saved_paths = {}
        if args.save_images:
            saved_paths = maybe_save_images(indices, images, adv_images, delta if args.save_delta else None, save_dirs)

        for i, idx in enumerate(indices):
            saved = saved_paths.get(idx, {})
            manifest_rows.append(
                {
                    "index": idx,
                    "label": int(labels[i].item()),
                    "clean_pred": int(clean_preds[i].item()),
                    "adv_pred": int(adv_preds[i].item()),
                    "clean_path": saved.get("clean_path"),
                    "adv_path": saved.get("adv_path"),
                    "delta_path": saved.get("delta_path"),
                }
            )

        total += batch_size

    metrics = compute_metrics(clean_correct, adv_correct, total)
    elapsed = time.time() - start_time

    results = {
        "config": vars(args),
        "metrics": {
            "clean_acc": metrics.clean_acc,
            "adv_acc": metrics.adv_acc,
            "asr": metrics.asr,
        },
        "num_samples": total,
        "device": str(device),
        "elapsed_sec": elapsed,
    }

    write_manifest_csv(save_dir / "manifest.csv", manifest_rows)
    write_results_json(save_dir / "results.json", results)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
