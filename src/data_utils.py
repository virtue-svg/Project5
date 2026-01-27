# -*- coding: utf-8 -*-
from __future__ import annotations
# 作用: 数据读取与路径映射工具。
# 流程: 解析标签文件 -> 找到文本/图像路径 -> 生成划分记录。
# 输出: Record 列表及 train/val/test CSV。

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
TEXT_EXTS = [".txt", ".text"]


@dataclass
class Record:
    guid: str
    text_path: Optional[Path]
    image_path: Optional[Path]
    label: Optional[str]


def _split_fields(line: str) -> List[str]:
    # 支持 tab / 逗号 / 空白分隔
    if "\t" in line:
        parts = line.split("\t")
    elif "," in line:
        parts = line.split(",")
    else:
        parts = line.split()
    return [p.strip() for p in parts if p.strip()]


def load_label_file(path: Path, has_label: bool = True) -> List[Tuple[str, Optional[str]]]:
    # 读取标签文件，返回 (guid, label) 列表
    items: List[Tuple[str, Optional[str]]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = _split_fields(line)
            if not parts:
                continue
            guid = parts[0]
            if guid.lower() == "guid":
                continue
            label = parts[1] if has_label and len(parts) > 1 else None
            items.append((guid, label))
    return items


def find_text_path(data_dir: Path, guid: str) -> Optional[Path]:
    # 根据 guid 查找对应文本文件
    for ext in TEXT_EXTS:
        candidate = data_dir / f"{guid}{ext}"
        if candidate.exists():
            return candidate
    return None


def find_image_path(data_dir: Path, guid: str) -> Optional[Path]:
    # 根据 guid 查找对应图像文件
    for ext in IMAGE_EXTS:
        candidate = data_dir / f"{guid}{ext}"
        if candidate.exists():
            return candidate
    return None


def build_records(
    label_path: Path, data_dir: Path, has_label: bool = True
) -> List[Record]:
    # 生成 Record 列表（包含路径与标签）
    items = load_label_file(label_path, has_label=has_label)
    records: List[Record] = []
    for guid, label in items:
        text_path = find_text_path(data_dir, guid)
        image_path = find_image_path(data_dir, guid)
        records.append(
            Record(
                guid=guid,
                text_path=text_path,
                image_path=image_path,
                label=label,
            )
        )
    return records


def ensure_paths(records: Iterable[Record]) -> None:
    # 检查路径是否存在，缺失则报错
    missing = []
    for rec in records:
        if rec.text_path is None or rec.image_path is None:
            missing.append(rec.guid)
    if missing:
        raise FileNotFoundError(
            f"Missing text/image files for {len(missing)} samples. "
            f"Examples: {missing[:5]}"
        )
