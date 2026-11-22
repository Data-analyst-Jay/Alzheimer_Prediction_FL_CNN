"""Flower ServerApp: FedProx aggregation over FedBN shared parameters."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedProx

from src.federated.fed_utils import (
    assemble_reference_state,
    batchnorm_state_keys,
    build_model_from_metadata,
    load_app_configs,
    load_warmstart_state,
    save_torch_atomic,
    shared_state_dict,
)

app = ServerApp()


class FedProxFedBN(FedProx):
    """FedProx strategy that checkpoints shared weights selected by client FedBN validation."""

    def __init__(self, *, client_count: int, rounds: int, proximal_mu: float,
                 artifacts_dir: Path, warmstart_state: dict, bn_keys: set[str],
                 metadata: dict, num_classes: int) -> None:
        super().__init__(
            fraction_train=1.0,
            fraction_evaluate=1.0,
            min_train_nodes=client_count,
            min_evaluate_nodes=client_count,
            min_available_nodes=client_count,
            weighted_by_key="num-examples",
            proximal_mu=proximal_mu,
        )
        self.client_count = client_count
        self.rounds = rounds
        self.artifacts_dir = artifacts_dir
        self.warmstart_state = warmstart_state
        self.bn_keys = bn_keys
        self.metadata = metadata
        self.num_classes = num_classes
        self.round_shared: dict[int, dict] = {}
        self.best_validation_macro_f1 = float("-inf")
        self.history: list[dict] = []

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        # Cache only shared model state; BatchNorm values remain on each client.
        self.round_shared[server_round] = arrays.to_torch_state_dict()
        # The held-out test set is evaluated only after the last training round.
        config["evaluate-test"] = server_round == self.rounds
        return super().configure_evaluate(server_round, arrays, config, grid)

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]) -> MetricRecord | None:
        metrics = super().aggregate_evaluate(server_round, replies)
        if metrics is None:
            return None

        row = {"round": server_round}
        for key in (
            "eval_loss", "eval_accuracy", "eval_macro_f1",
            "test_loss", "test_accuracy", "test_macro_f1",
        ):
            if key in metrics:
                row[key] = float(metrics[key])
        self.history.append(row)
        self._write_history()

        shared = self.round_shared.get(server_round)
        if shared is not None:
            payload = self._checkpoint_payload(shared, server_round, row)
            save_torch_atomic(self.artifacts_dir / "fl_global_latest.pt", payload)
            val_f1 = row.get("eval_macro_f1")
            if val_f1 is not None and val_f1 > self.best_validation_macro_f1:
                self.best_validation_macro_f1 = val_f1
                payload["best_validation_macro_f1"] = val_f1
                save_torch_atomic(self.artifacts_dir / "fl_global_best.pt", payload)

        # These test metrics are produced by clients with their silo-local BN.
        if server_round == self.rounds and "test_accuracy" in row:
            test_result = {
                "evaluation": "mean of four personalized client models using silo-local BatchNorm",
                "round": server_round,
                "test_loss": row.get("test_loss"),
                "test_accuracy": row.get("test_accuracy"),
                "test_macro_f1": row.get("test_macro_f1"),
                "test_examples_per_client": int(metrics.get("test-num-examples", 0)),
            }
            self._write_json(self.artifacts_dir / "fl_test_metrics.json", test_result)
        return metrics

    def _checkpoint_payload(self, shared: dict, server_round: int, metrics: dict) -> dict:
        reference_state = assemble_reference_state(self.warmstart_state, shared, self.bn_keys)
        return {
            # `model_state_dict` is a portable reference model using the warm-start
            # BN state. Actual FedBN inference at a participating silo uses its
            # private local BN state, which is deliberately absent here.
            "model_state_dict": reference_state,
            "shared_state_dict": shared_state_dict(shared, self.bn_keys),
            "architecture": "densenet121",
            "num_classes": self.num_classes,
            "class_names": self.metadata.get("class_names", []),
            "classifier_hidden_units": int(self.metadata.get("classifier_hidden_units", 256)),
            "dropout": float(self.metadata.get("dropout", 0.3)),
            "batchnorm_policy": "FedBN: BatchNorm parameters and buffers are silo-local and never aggregated",
            "reference_batchnorm": "warm-start checkpoint; use each client's local BN state for personalized inference",
            "server_round": server_round,
            "validation_metrics": metrics,
        }

    def _write_history(self) -> None:
        path = self.artifacts_dir / "fl_round_metrics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "round", "eval_loss", "eval_accuracy", "eval_macro_f1",
            "test_loss", "test_accuracy", "test_macro_f1",
        ]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(self.history)

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(path)


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Start the configured four-client cross-silo run from the Stage 2 checkpoint."""
    base, federated = load_app_configs()
    warmstart_state, metadata = load_warmstart_state(base, federated)
    num_classes = int(base["data"].get("num_classes", 4))
    model = build_model_from_metadata(base, metadata)
    model.load_state_dict(warmstart_state, strict=True)
    bn_keys = batchnorm_state_keys(model)

    artifacts_dir = Path(base["data"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    clients = int(federated.get("clients", 4))
    rounds = int(context.run_config.get("num-server-rounds", federated.get("rounds", 20)))
    local_epochs = int(context.run_config.get("local-epochs", federated.get("client_epochs", 1)))
    proximal_mu = float(federated.get("fedprox_mu", 0.01))

    strategy = FedProxFedBN(
        client_count=clients,
        rounds=rounds,
        proximal_mu=proximal_mu,
        artifacts_dir=artifacts_dir,
        warmstart_state=warmstart_state,
        bn_keys=bn_keys,
        metadata=metadata,
        num_classes=num_classes,
    )
    initial_shared = shared_state_dict(warmstart_state, bn_keys)
    print(
        f"Starting FedProx + FedBN: {clients} silos, {rounds} rounds, "
        f"{local_epochs} local epoch(s), mu={proximal_mu:g}"
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(initial_shared),
        train_config=ConfigRecord({"local-epochs": local_epochs}),
        evaluate_config=ConfigRecord({}),
        num_rounds=rounds,
    )

    if result.arrays is not None:
        final_shared = result.arrays.to_torch_state_dict()
        final_state = assemble_reference_state(warmstart_state, final_shared, bn_keys)
        save_torch_atomic(artifacts_dir / "fl_global_final.pt", {
            "model_state_dict": final_state,
            "shared_state_dict": shared_state_dict(final_shared, bn_keys),
            "architecture": "densenet121",
            "num_classes": num_classes,
            "class_names": metadata.get("class_names", []),
            "batchnorm_policy": "FedBN: BatchNorm remains silo-local",
            "reference_batchnorm": "warm-start BN; use a client's local BN state for personalized inference",
            "server_round": rounds,
        })
    print(f"Best FedBN shared-weight checkpoint: {artifacts_dir / 'fl_global_best.pt'}")
    print(f"Final-round checkpoint: {artifacts_dir / 'fl_global_final.pt'}")
