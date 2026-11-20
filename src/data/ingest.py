"""Read, validate, and persist source train/test Parquet files."""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image, UnidentifiedImageError

from src.data.parquet_dataset import image_bytes


def validate_frame(df: pd.DataFrame, *, name: str, image_column: str, label_column: str,
                   num_classes: int, verify_images: bool) -> dict:
    missing = {image_column, label_column} - set(df.columns)
    if missing:
        raise ValueError(f"{name} parquet is missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{name} parquet contains no rows")
    if df[label_column].isna().any():
        raise ValueError(f"{name} parquet contains null labels")
    labels = pd.to_numeric(df[label_column], errors="raise").astype(int)
    invalid = sorted(set(labels.unique()) - set(range(num_classes)))
    if invalid:
        raise ValueError(f"{name} parquet has labels outside 0..{num_classes - 1}: {invalid}")
    if verify_images:
        bad_rows = []
        for idx, entry in enumerate(df[image_column]):
            try:
                with Image.open(io.BytesIO(image_bytes(entry))) as image:
                    image.verify()
            except (TypeError, UnidentifiedImageError, OSError, ValueError):
                bad_rows.append(idx)
                if len(bad_rows) == 10:
                    break
        if bad_rows:
            raise ValueError(f"{name} parquet has unreadable images at row indices (first 10): {bad_rows}")
    return {"rows": len(df), "class_counts": labels.value_counts().sort_index().to_dict()}


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    return cfg["data"] if "data" in cfg else cfg


def ingest(config: dict) -> None:
    image_col, label_col = config.get("image_column", "image"), config.get("label_column", "label")
    n_classes = int(config.get("num_classes", 4))
    do_verify = bool(config.get("verify_images", True))
    source_train, source_test = Path(config["train_parquet"]), Path(config["test_parquet"])
    output = Path(config["processed_dir"])
    output.mkdir(parents=True, exist_ok=True)
    for name, source in (("train", source_train), ("test", source_test)):
        if not source.is_file():
            raise FileNotFoundError(f"{name} parquet not found: {source}. Mount Google Drive and check the configured path.")
        frame = pd.read_parquet(source)
        stats = validate_frame(frame, name=name, image_column=image_col, label_column=label_col,
                               num_classes=n_classes, verify_images=do_verify)
        target = output / f"{name}_validated.parquet"
        frame.to_parquet(target, index=False)
        print(f"{name}: {stats['rows']} rows; class counts={stats['class_counts']}; saved={target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    ingest(load_config(args.config))


if __name__ == "__main__":
    main()
