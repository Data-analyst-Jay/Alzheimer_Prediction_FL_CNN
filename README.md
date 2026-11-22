# Alzheimer MRI Federated Learning: data stage

The repository contains the first pipeline stage: validating the source Parquet files, reserving validation data, creating the warm-start seed and four deterministic non-IID silos, and loading images with the same ImageNet preprocessing for later training code.

## Colab / Google Drive workflow

Mount Drive in a Colab cell, clone the repository, install the project dependencies, then run these commands from the repository root:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!pip install -e .
!python -m src.data.ingest --config configs/base.yaml
!python -m src.data.partition --config configs/base.yaml
```

The defaults in `configs/base.yaml` expect `train.parquet` and `test.parquet` at the Drive paths supplied in the project brief. Edit those paths if needed. Generated partition Parquet files are written under the configured Drive `processed_dir`; they are not written into Git.

## Data allocation

The source test Parquet is copied intact to `test.parquet` and must be used only for final evaluation. From the source training Parquet, the script first reserves 15% for `validation.parquet`. Of the remaining 85%, 20% becomes `warmstart_seed.parquet`; the other 80% becomes the FL pool and is divided across `silo_0.parquet` through `silo_3.parquet` using a Dirichlet class allocation (`alpha=0.5`). `federated_train.parquet` is also saved as the full silo pool for auditing. These fractions and the seed are configurable.

Every training record is assigned to exactly one of validation, warm-start, or one silo. `_source_row_id` makes the partition integrity check possible and is metadata, not a model input. The test set is never passed through the partitioning logic.

## Loading a partition

Later training code can reuse one loader implementation:

```python
from src.data.loaders import build_partition_loader, load_config

cfg = load_config('configs/base.yaml')
train_loader = build_partition_loader(cfg, 'silo_0', training=True)
val_loader = build_partition_loader(cfg, 'validation', training=False)
test_loader = build_partition_loader(cfg, 'test', training=False)
```

Images are decoded on demand, converted to RGB (replicating grayscale MRI across three channels), optionally resized only when `image_size` is set, converted to tensors, then normalized with ImageNet mean and standard deviation. Rotation and horizontal flip are applied only to training loaders. The default preserves native image dimensions, matching the supplied 128×128 example. Set `num_workers: 0` in Colab if multiprocessing workers cause issues.

## Dependencies

Install from `pyproject.toml` with `pip install -e .`. The project expects image labels to be integer class IDs `0..3`, and Parquet columns named `image` and `label` (both configurable).

## Stage 2: warm-start initialization

After Stage 1 has saved `warmstart_seed.parquet` and `validation.parquet` in Drive, run:

```bash
python -m src.warmstart.train_warmstart \
  --config configs/base.yaml \
  --stage-config configs/warmstart.yaml
```

This stage reads only those saved partitions; it does not ingest or repartition the original parquet files. It writes a best checkpoint, an epoch-by-epoch resumable checkpoint, and a CSV training history to the Drive `artifacts_dir`. To continue after a disconnected Colab runtime, remount Drive, clone the repository, and run the same command with `--resume`.
