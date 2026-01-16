# CV_Attack

CV_Attack is a standalone adversarial attack pipeline extracted from the original workflow. It **only** performs:

1. Model & dataset loading
2. Adversarial attack generation
3. Attack evaluation
4. Saving clean/adv images and logs

No purification/defense logic is included.

## Features

- CLI entrypoint (`attack_runner.py`)
- Supports common datasets (CIFAR-10, ImageNet, custom folder)
- Supports attacks: PGD (default) and FGSM
- Outputs clean/adv images, optional delta visualization, manifest, and results summary

## Install

```bash
cd CV_Attack
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Quick start (CIFAR-10 + ResNet18 + PGD)

```bash
python attack_runner.py \
  --model resnet18 \
  --dataset cifar10 --data_root ./data \
  --attack pgd --eps 0.03137254901960784 --alpha 0.00784313725490196 --steps 10 \
  --batch_size 128 \
  --save_dir ./outputs/cifar10_resnet18_pgd
```

### ImageNet example

```bash
python attack_runner.py \
  --model resnet50 --pretrained \
  --dataset imagenet --data_root /path/to/imagenet \
  --attack pgd --eps 0.00784313725490196 --alpha 0.00196078431372549 --steps 10 \
  --batch_size 64 \
  --save_dir ./outputs/imagenet_resnet50_pgd
```

### Custom folder

Use `custom_folder` to point to an `ImageFolder` directory structure:

```bash
python attack_runner.py \
  --model resnet18 --pretrained \
  --dataset custom_folder --data_root /path/to/images \
  --attack fgsm --eps 0.00784313725490196 \
  --save_dir ./outputs/custom_folder_fgsm
```

## Output Layout

```
outputs/<run>/
  clean/            # clean images
  adv/              # adversarial images
  delta/            # optional perturbation visualization
  manifest.csv      # index, label, preds, paths
  results.json      # config + metrics
```

## Notes

- Attacks are performed in **pixel space** (`[0, 1]`), while models receive normalized inputs internally.
- Default normalization:
  - CIFAR-10: mean `(0.4914, 0.4822, 0.4465)`, std `(0.2471, 0.2435, 0.2616)`
  - ImageNet/custom folder: mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`
- Provide `--ckpt` to load custom weights. For torchvision models, `--pretrained` loads ImageNet weights where available.

## Config

Default parameters live in `configs/default.yaml`. You can override any value via CLI flags.
