"""Build reusable PyTorch DataLoaders from the saved Parquet partitions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.parquet_dataset import AlzheimerParquetDataset, build_transforms


def build_partition_loader(
    config: dict,
    partition_name: str,
    *,
    training: bool,
    shuffle: bool | None = None,
    seed: int | None = None,
    persistent_workers: bool | None = None,
) -> DataLoader:
    """Load one saved ``<partition_name>.parquet`` file as a PyTorch DataLoader."""
    data_cfg = config.get("data", config)
    loader_cfg = config.get("loader", {})
    root = Path(data_cfg["processed_dir"])
    path = root / f"{partition_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Partition not found: {path}. Run the ingest and partition stages first."
        )

    frame = pd.read_parquet(path)
    dataset = AlzheimerParquetDataset(
        frame,
        transform=build_transforms(
            training=training,
            image_size=data_cfg.get("image_size"),
        ),
        image_column=data_cfg.get("image_column", "image"),
        label_column=data_cfg.get("label_column", "label"),
        num_classes=int(data_cfg.get("num_classes", 4)),
    )

    if shuffle is None:
        shuffle = training
    worker_count = int(loader_cfg.get("num_workers", 2))
    if persistent_workers is None:
        persistent_workers = worker_count > 0

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=int(loader_cfg.get("batch_size", 32)),
        shuffle=shuffle,
        num_workers=worker_count,
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        persistent_workers=bool(persistent_workers and worker_count > 0),
        generator=generator,
    )


def load_config(path: str | Path = "configs/base.yaml") -> dict:
    """Read a YAML configuration file."""
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)
