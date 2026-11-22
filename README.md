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

## Stage 3: cross-silo federated learning

Stage 3 starts from `artifacts/warmstart_densenet121_best.pt` and reads the four existing `silo_0.parquet` through `silo_3.parquet` partitions. It does not rerun Stage 1 or Stage 2. Each Flower SuperNode uses its `partition-id` to select one silo.

The Flower app uses **FedProx + FedBN**. FedProx applies its proximal penalty to shared, non-BatchNorm parameters. FedBN keeps each client's BatchNorm affine parameters and running statistics in that client's Flower `Context.state`; only non-BatchNorm weights are sent to the server and aggregated. For that reason BatchNorm runs in training mode locally and adapts to each silo; its state is not frozen or averaged.

Client evaluations use the held-out `validation.parquet` with each silo's own BN state after every round. The held-out `test.parquet` is evaluated once by the clients in the final round, with only metrics returned. The server does not read the Parquet image data. Since this is a Colab simulation, all four clients are processes on one runtime and can access the mounted Drive; it demonstrates the communication/aggregation logic but does not provide infrastructure-level data isolation.

In Colab, mount Drive, install the updated project dependencies, and run:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!pip install -e .
!flwr run . --stream --federation-config="num-supernodes=4 client-resources-num-cpus=2 client-resources-num-gpus=0.5"
```

The current Flower app configuration uses 20 rounds and one local epoch per round. Simulation GPU allocation is a scheduling hint; Flower documents it as soft resource assignment, so reduce concurrency or the loader batch size if the Colab GPU runs out of memory.

Stage 3 writes `fl_global_best.pt`, `fl_global_final.pt`, `fl_global_latest.pt`, `fl_round_metrics.csv`, and `fl_test_metrics.json` under the Drive `artifacts_dir`. The server checkpoint stores the global shared weights and a warm-start-BN reference model. The participating clients keep their personalized BN values locally in Flower state; a standalone deployment at a client must combine the shared checkpoint with that client's local BN state.
