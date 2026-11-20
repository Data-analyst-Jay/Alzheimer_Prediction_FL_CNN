"""PyTorch dataset and transforms for MRI images stored in Parquet rows."""

from __future__ import annotations

import io
from collections.abc import Mapping
from typing import Any

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def image_bytes(entry: Any) -> bytes:
    """Extract image bytes from common Hugging Face/Kaggle Parquet encodings."""
    if isinstance(entry, Mapping):
        entry = entry.get("bytes")
    if isinstance(entry, memoryview):
        entry = entry.tobytes()
    if not isinstance(entry, (bytes, bytearray)):
        raise TypeError(f"Expected encoded image bytes or a mapping containing 'bytes'; got {type(entry).__name__}")
    return bytes(entry)


def build_transforms(*, training: bool, image_size: int | None = None):
    ops = []
    if image_size is not None:
        ops.append(transforms.Resize((image_size, image_size)))
    if training:
        ops.extend((transforms.RandomRotation(degrees=10), transforms.RandomHorizontalFlip(p=0.5)))
    ops.extend((transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)))
    return transforms.Compose(ops)


class AlzheimerParquetDataset(Dataset):
    """Decode grayscale/RGB MRI image records and return normalized 3-channel tensors."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform=None,
        *,
        image_column: str = "image",
        label_column: str = "label",
        num_classes: int = 4,
    ) -> None:
        missing = {image_column, label_column} - set(dataframe.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.image_column = image_column
        self.label_column = label_column
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        try:
            with Image.open(io.BytesIO(image_bytes(row[self.image_column]))) as decoded:
                image = decoded.convert("RGB")
        except (TypeError, UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Could not decode image at dataset row {idx}") from exc
        if self.transform is not None:
            image = self.transform(image)
        label = int(row[self.label_column])
        if not 0 <= label < self.num_classes:
            raise ValueError(f"Label {label} at row {idx} is outside [0, {self.num_classes - 1}]")
        return image, torch.tensor(label, dtype=torch.long)
