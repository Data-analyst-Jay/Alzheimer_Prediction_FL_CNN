"""Fine-tune ImageNet DenseNet-121 on the saved warm-start seed partition."""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from src.data.loaders import build_partition_loader
from src.models.backbone import build_densenet121


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Request deterministic kernels where PyTorch provides them. Some CUDA ops
    # may still warn or fail if no deterministic implementation exists.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_configs(base_path: str, stage_path: str) -> dict:
    with open(base_path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    with open(stage_path, encoding="utf-8") as stream:
        stage_config = yaml.safe_load(stream) or {}
    config.update(stage_config)
    return config


def class_weights_from_seed(seed_path: Path, label_column: str, num_classes: int,
                            device: torch.device) -> torch.Tensor:
    labels = pd.read_parquet(seed_path, columns=[label_column])[label_column].astype(int)
    counts = np.bincount(labels.to_numpy(), minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError(f"Warm-start seed must contain every class; class counts are {counts.tolist()}")
    weights = len(labels) / (num_classes * counts.astype(np.float32))
    print(f"Warm-start class counts: {counts.tolist()}")
    print(f"Cross-entropy class weights: {weights.round(4).tolist()}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run(config: dict, *, resume: bool = False) -> None:
    data_cfg = config["data"]
    settings = config["warmstart"]
    seed = int(settings.get("seed", data_cfg.get("seed", 42)))
    seed_everything(seed)

    processed_dir = Path(data_cfg["processed_dir"])
    seed_path = processed_dir / "warmstart_seed.parquet"
    validation_path = processed_dir / "validation.parquet"
    for path in (seed_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required saved partition is missing: {path}. Run the data ingestion and partitioning steps first."
            )

    artifacts_dir = Path(data_cfg["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    best_path = artifacts_dir / settings.get("checkpoint_name", "warmstart_densenet121_best.pt")
    latest_path = artifacts_dir / settings.get("latest_checkpoint_name", "warmstart_densenet121_latest.pt")
    history_path = artifacts_dir / settings.get("history_name", "warmstart_history.csv")

    # A fresh iterator per epoch plus an epoch-specific seed makes data order and
    # augmentations repeatable when resuming from the latest Drive checkpoint.
    loader_seed = seed
    train_loader = build_partition_loader(
        config, "warmstart_seed", training=True, seed=loader_seed, persistent_workers=False
    )
    validation_loader = build_partition_loader(
        config, "validation", training=False, seed=loader_seed + 1, persistent_workers=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(data_cfg.get("num_classes", 4))
    model = build_densenet121(
        num_classes=num_classes,
        pretrained=bool(settings.get("use_imagenet_weights", True)),
        classifier_hidden_units=int(settings.get("classifier_hidden_units", 256)),
        dropout=float(settings.get("dropout", 0.3)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": float(settings.get("backbone_learning_rate", 1e-5))},
            {"params": model.classifier.parameters(), "lr": float(settings.get("classifier_learning_rate", 1e-4))},
        ],
        weight_decay=float(settings.get("weight_decay", 1e-4)),
    )
    max_epochs = int(settings.get("epochs", 15))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights_from_seed(
        seed_path, data_cfg.get("label_column", "label"), num_classes, device
    ))

    start_epoch, best_val_f1, patience_count = 1, -1.0, 0
    history: list[dict] = []
    if resume:
        if not latest_path.is_file():
            raise FileNotFoundError(f"Cannot resume; latest checkpoint does not exist: {latest_path}")
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_f1 = float(checkpoint["best_val_macro_f1"])
        patience_count = int(checkpoint["patience_count"])
        if history_path.exists():
            history = pd.read_csv(history_path).to_dict(orient="records")
        print(f"Resuming at epoch {start_epoch} from {latest_path}")
        if patience_count >= int(settings.get("early_stopping_patience", 5)):
            print("The previous run had already reached early stopping; use a fresh run to restart training.")
            return

    if start_epoch > max_epochs:
        print(f"Warm-start already completed through epoch {max_epochs}; best model: {best_path}")
        return

    print(f"Device: {device}; seed: {seed}; epochs: {max_epochs}; seed rows: {len(train_loader.dataset)}")
    for epoch in range(start_epoch, max_epochs + 1):
        epoch_seed = seed + epoch
        seed_everything(epoch_seed)
        if train_loader.generator is not None:
            train_loader.generator.manual_seed(epoch_seed)
        if validation_loader.generator is not None:
            validation_loader.generator.manual_seed(epoch_seed + 1)

        model.train()
        total_loss, total_seen = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_seen += labels.size(0)
        train_loss = total_loss / max(total_seen, 1)

        model.eval()
        val_targets, val_predictions = [], []
        with torch.inference_mode():
            for images, labels in validation_loader:
                logits = model(images.to(device, non_blocking=True))
                val_predictions.extend(logits.argmax(dim=1).cpu().tolist())
                val_targets.extend(labels.tolist())
        val_accuracy = accuracy_score(val_targets, val_predictions)
        val_macro_f1 = f1_score(val_targets, val_predictions, average="macro", zero_division=0)
        scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_accuracy": val_accuracy,
            "validation_macro_f1": val_macro_f1,
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "classifier_lr": optimizer.param_groups[1]["lr"],
        }
        history.append(epoch_record)
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(
            f"Epoch {epoch:02d}/{max_epochs:02d} | train loss {train_loss:.4f} | "
            f"val accuracy {val_accuracy:.4f} | val macro-F1 {val_macro_f1:.4f}"
        )

        improved = val_macro_f1 > best_val_f1
        if improved:
            best_val_f1 = val_macro_f1
            patience_count = 0
            save_checkpoint(best_path, {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "architecture": "densenet121",
                "num_classes": num_classes,
                "class_names": data_cfg.get("class_names", []),
                "classifier_hidden_units": int(settings.get("classifier_hidden_units", 256)),
                "dropout": float(settings.get("dropout", 0.3)),
                "seed": seed,
                "best_epoch": epoch,
                "validation_macro_f1": val_macro_f1,
                "training_config": settings,
            })
        else:
            patience_count += 1

        # This checkpoint is updated every epoch on Drive, so a disconnected
        # Colab session can resume without repeating the warm-start epochs.
        save_checkpoint(latest_path, {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_macro_f1": best_val_f1,
            "patience_count": patience_count,
            "seed": seed,
        })

        if patience_count >= int(settings.get("early_stopping_patience", 5)):
            print(f"Early stopping: validation macro-F1 did not improve for {patience_count} epochs.")
            break

    print(f"Best warm-start checkpoint: {best_path} (validation macro-F1={best_val_f1:.4f})")
    print(f"Latest resumable checkpoint: {latest_path}")
    print(f"Training history: {history_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--stage-config", default="configs/warmstart.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint on Drive")
    args = parser.parse_args()
    run(load_configs(args.config, args.stage_config), resume=args.resume)


if __name__ == "__main__":
    main()
