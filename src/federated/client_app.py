"""Flower ClientApp: train one saved silo partition with FedProx + FedBN."""

from __future__ import annotations

import copy

import numpy as np
import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from src.data.loaders import build_partition_loader
from src.federated.fed_utils import (
    batchnorm_state_dict,
    batchnorm_state_keys,
    build_model_from_metadata,
    load_app_configs,
    load_warmstart_state,
    shared_state_dict,
)

LOCAL_BN_STATE_KEY = "cvfl-fedbn-local-batchnorm"
app = ClientApp()


def _client_id(context: Context, client_count: int) -> int:
    """Use Flower's partition-id; node-id modulo is a simulation fallback."""
    partition_id = context.node_config.get("partition-id")
    if partition_id is None:
        partition_id = context.node_id % client_count
    partition_id = int(partition_id)
    if not 0 <= partition_id < client_count:
        raise ValueError(
            f"Flower assigned partition-id={partition_id}, but configured clients are 0..{client_count - 1}"
        )
    return partition_id


def _build_client_model(msg: Message, context: Context, base: dict, federated: dict):
    local_bn_record = context.state.get(LOCAL_BN_STATE_KEY)
    if local_bn_record is None:
        # On first contact, load the Stage 2 checkpoint to seed local BN.
        warmstart_state, metadata = load_warmstart_state(base, federated)
        model = build_model_from_metadata(base, metadata)
        state = copy.deepcopy(warmstart_state)
    else:
        # In later rounds, shared values arrive from the server and this node's
        # private BN state arrives from Flower Context; avoid rereading the full checkpoint.
        model = build_model_from_metadata(base, {
            "num_classes": base["data"].get("num_classes", 4),
            "classifier_hidden_units": federated.get("classifier_hidden_units", 256),
            "dropout": federated.get("dropout", 0.3),
        })
        state = model.state_dict()
        state.update(local_bn_record.to_torch_state_dict())
    bn_keys = batchnorm_state_keys(model)
    shared = msg.content["arrays"].to_torch_state_dict()
    state.update(shared)
    model.load_state_dict(state, strict=True)
    return model, bn_keys


def _class_weights(loader, num_classes: int, device: torch.device) -> torch.Tensor:
    labels = loader.dataset.df[loader.dataset.label_column].astype(int).to_numpy()
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    present = counts > 0
    if not present.any():
        raise ValueError("This silo has no labeled examples")
    weights = np.zeros(num_classes, dtype=np.float32)
    weights[present] = len(labels) / (int(present.sum()) * counts[present])
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _evaluate(model, loader, device: torch.device) -> dict[str, float | int]:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.cpu().tolist())
    return {
        "loss": total_loss / max(len(targets), 1),
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "num_examples": len(targets),
    }


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train locally and return only the non-BatchNorm shared state."""
    base, federated = load_app_configs()
    client_id = _client_id(context, int(federated.get("clients", 4)))
    seed = int(federated.get("seed", base["data"].get("seed", 42)))
    round_number = int(msg.content["config"].get("server-round", 1))
    client_seed = seed + client_id * 10_000 + round_number
    torch.manual_seed(client_seed)
    np.random.seed(client_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(client_seed)

    model, bn_keys = _build_client_model(msg, context, base, federated)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_loader = build_partition_loader(
        base,
        f"silo_{client_id}",
        training=True,
        seed=client_seed,
        persistent_workers=False,
    )

    train_cfg = msg.content["config"]
    local_epochs = int(train_cfg.get("local-epochs", federated.get("client_epochs", 1)))
    proximal_mu = float(train_cfg.get("proximal-mu", federated.get("fedprox_mu", 0.0)))
    criterion = nn.CrossEntropyLoss(
        weight=_class_weights(train_loader, int(base["data"].get("num_classes", 4)), device)
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": float(federated.get("backbone_learning_rate", 1e-5))},
            {"params": model.classifier.parameters(), "lr": float(federated.get("classifier_learning_rate", 1e-4))},
        ],
        weight_decay=float(federated.get("weight_decay", 1e-4)),
    )

    # FedProx regularizes only the shared parameters. BN parameters and buffers
    # remain silo-local and are deliberately excluded from this term.
    global_named = {
        name: value.detach().clone().to(device)
        for name, value in model.named_parameters()
        if name not in bn_keys
    }
    model.train()  # BatchNorm statistics adapt locally; they are never aggregated.
    total_loss, examples_seen = 0.0, 0
    for _ in range(local_epochs):
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if proximal_mu > 0:
                proximal_term = torch.zeros((), device=device)
                for name, parameter in model.named_parameters():
                    if name not in bn_keys:
                        proximal_term = proximal_term + torch.sum(
                            (parameter - global_named[name]) ** 2
                        )
                loss = loss + (proximal_mu / 2.0) * proximal_term
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            examples_seen += labels.size(0)

    # Retain this node's BatchNorm affine parameters and running buffers in its
    # private Flower Context so they survive the next ClientApp invocation.
    local_bn = batchnorm_state_dict(model.state_dict(), bn_keys)
    context.state[LOCAL_BN_STATE_KEY] = ArrayRecord(local_bn)

    shared_after_training = shared_state_dict(model.state_dict(), bn_keys)
    metrics = MetricRecord({
        "train_loss": total_loss / max(examples_seen, 1),
        "num-examples": len(train_loader.dataset),
    })
    content = RecordDict({
        "arrays": ArrayRecord(shared_after_training),
        "metrics": metrics,
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the personalized FedBN model on validation, and test only at the final round."""
    base, federated = load_app_configs()
    client_id = _client_id(context, int(federated.get("clients", 4)))
    model, _bn_keys = _build_client_model(msg, context, base, federated)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    metrics: dict[str, float | int] = {}
    validation_loader = build_partition_loader(
        base, "validation", training=False, seed=int(federated.get("seed", 42)),
        persistent_workers=False,
    )
    validation = _evaluate(model, validation_loader, device)
    metrics.update({
        "eval_loss": validation["loss"],
        "eval_accuracy": validation["accuracy"],
        "eval_macro_f1": validation["macro_f1"],
        "num-examples": validation["num_examples"],
    })

    config = msg.content["config"]
    if bool(config.get("evaluate-test", False)):
        test_loader = build_partition_loader(
            base, "test", training=False, seed=int(federated.get("seed", 42)) + 1,
            persistent_workers=False,
        )
        test = _evaluate(model, test_loader, device)
        metrics.update({
            "test_loss": test["loss"],
            "test_accuracy": test["accuracy"],
            "test_macro_f1": test["macro_f1"],
            "test-num-examples": test["num_examples"],
        })

    return Message(
        content=RecordDict({"metrics": MetricRecord(metrics)}),
        reply_to=msg,
    )
