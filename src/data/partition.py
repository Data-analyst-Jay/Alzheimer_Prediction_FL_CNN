"""Create fixed validation, warm-start, four-silo, and untouched test partitions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def _stratified_split(df: pd.DataFrame, *, fraction: float, label_column: str,
                      seed: int, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        left, right = train_test_split(
            df, test_size=fraction, random_state=seed, stratify=df[label_column]
        )
    except ValueError as exc:
        raise ValueError(
            f"Cannot create stratified {name} split. Ensure every class has enough examples; "
            "for four classes, each class needs at least two rows and each requested partition must fit."
        ) from exc
    return left.reset_index(drop=True), right.reset_index(drop=True)


def dirichlet_partition(df: pd.DataFrame, *, label_column: str, num_silos: int,
                        alpha: float, seed: int) -> list[pd.DataFrame]:
    """Allocate each class among clients using a deterministic Dirichlet draw."""
    if num_silos < 2 or alpha <= 0:
        raise ValueError("num_silos must be >= 2 and Dirichlet alpha must be > 0")
    if len(df) < num_silos:
        raise ValueError(f"Need at least {num_silos} training rows to create non-empty silos")
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(num_silos)]
    for label in sorted(df[label_column].unique()):
        indices = np.flatnonzero(df[label_column].to_numpy() == label)
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.full(num_silos, alpha))
        counts = rng.multinomial(len(indices), proportions)
        start = 0
        for silo, count in enumerate(counts):
            buckets[silo].extend(indices[start:start + count].tolist())
            start += count
    # Shuffle rows deterministically within each client.
    for bucket in buckets:
        rng.shuffle(bucket)
    # Very small classes/data can leave a Dirichlet bucket empty. Move one row
    # from the largest bucket to each empty bucket to keep every client usable.
    for bucket in buckets:
        if not bucket:
            largest_idx = max(range(num_silos), key=lambda i: len(buckets[i]))
            bucket.append(buckets[largest_idx].pop())
    return [df.iloc[indices].sample(frac=1, random_state=seed + i).reset_index(drop=True)
            for i, indices in enumerate(buckets)]


def partition(config: dict) -> dict[str, pd.DataFrame]:
    data_dir = Path(config["processed_dir"])
    train_path = data_dir / "train_validated.parquet"
    test_path = data_dir / "test_validated.parquet"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Validated Parquet files not found; run `python -m src.data.ingest` first.")
    label = config.get("label_column", "label")
    seed = int(config.get("seed", 42))
    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)  # Kept whole and unchanged for final testing only.
    if "_source_row_id" in train.columns:
        raise ValueError("Source data already contains reserved column '_source_row_id'")
    train["_source_row_id"] = np.arange(len(train), dtype=np.int64)
    validation_fraction = float(config.get("validation_fraction", 0.15))
    warmstart_fraction = float(config.get("warmstart_fraction", 0.20))
    if not 0 < validation_fraction < 1 or not 0 < warmstart_fraction < 1:
        raise ValueError("validation_fraction and warmstart_fraction must each be between 0 and 1")

    train_pool, validation = _stratified_split(
        train, fraction=validation_fraction, label_column=label, seed=seed, name="validation"
    )
    federated_pool, warmstart = _stratified_split(
        train_pool, fraction=warmstart_fraction, label_column=label, seed=seed + 1, name="warm-start"
    )
    silos = dirichlet_partition(
        federated_pool, label_column=label,
        num_silos=int(config.get("num_silos", 4)),
        alpha=float(config.get("dirichlet_alpha", 0.5)), seed=seed + 2,
    )

    partitions: dict[str, pd.DataFrame] = {
        "validation": validation,
        "warmstart_seed": warmstart,
        "federated_train": federated_pool,
        "test": test,
    }
    partitions.update({f"silo_{i}": frame for i, frame in enumerate(silos)})
    # Save compressed Parquet files (image bytes remain encoded; decoding/transforms happen at batch time).
    for name, frame in partitions.items():
        path = data_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False, compression="snappy")
        print(f"{name}: n={len(frame)}, classes={frame[label].value_counts().sort_index().to_dict()}, file={path}")

    # Assert every source training row appears exactly once, including after silo allocation.
    groups = [validation["_source_row_id"], warmstart["_source_row_id"]]
    groups.extend(silo["_source_row_id"] for silo in silos)
    combined_ids = pd.concat(groups, ignore_index=True)
    if len(combined_ids) != len(train) or combined_ids.duplicated().any() \
            or set(combined_ids) != set(train["_source_row_id"]):
        raise AssertionError("Partition overlap or missing source training records detected")
    return partitions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)["data"]
    partition(config)


if __name__ == "__main__":
    main()
