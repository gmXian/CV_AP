from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


@dataclass
class DatasetSpec:
    loader: DataLoader
    num_classes: int
    class_names: Optional[List[str]]
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


def _apply_limit(dataset: Dataset, limit: int) -> Dataset:
    if limit and limit > 0:
        limit = min(limit, len(dataset))
        indices = list(range(limit))
        return Subset(dataset, indices)
    return dataset


def load_dataset(
    name: str,
    root: str,
    batch_size: int,
    num_workers: int,
    limit: int = 0,
) -> DatasetSpec:
    name = name.lower()
    if name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
        transform = transforms.ToTensor()
        dataset = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)
        class_names = list(dataset.classes)
        num_classes = 10
    elif name == "imagenet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        dataset = datasets.ImageNet(root=root, split="val", transform=transform)
        class_names = dataset.classes
        num_classes = 1000
    elif name == "custom_folder":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        dataset = datasets.ImageFolder(root=root, transform=transform)
        class_names = dataset.classes
        num_classes = len(class_names)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    dataset = _apply_limit(dataset, limit)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return DatasetSpec(loader=loader, num_classes=num_classes, class_names=class_names, mean=mean, std=std)


def normalize(batch: torch.Tensor, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, device=batch.device).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=batch.device).view(1, 3, 1, 1)
    return (batch - mean_tensor) / std_tensor
