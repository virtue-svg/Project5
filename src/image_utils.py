from __future__ import annotations

from torchvision import transforms


def get_train_transforms(image_size: int = 224, normalize: bool = True):
    ops = [
        transforms.Resize(int(image_size * 1.15)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
    ]
    if normalize:
        ops.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    return transforms.Compose(ops)


def get_eval_transforms(image_size: int = 224, normalize: bool = True):
    ops = [
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ]
    if normalize:
        ops.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    return transforms.Compose(ops)
