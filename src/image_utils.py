# -*- coding: utf-8 -*-
from __future__ import annotations
# 作用: 图像预处理与增强配置。
# 流程: 定义训练/验证阶段的变换管道。
# 输出: torchvision transforms.Compose 对象。

from torchvision import transforms


def get_train_transforms(image_size: int = 224, normalize: bool = True):
    # 训练阶段使用轻量增强（裁剪/翻转/颜色抖动）
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
    # 验证阶段仅做 resize + center crop
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
