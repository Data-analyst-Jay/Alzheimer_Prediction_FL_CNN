"""Shared helpers for the Flower FedProx + FedBN applications."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import yaml
from torch import nn

from src.models.backbone import build_densenet121

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_app_configs() -> tuple[dict, dict]:
    """Load the same base and FL YAML settings in server and client processes."""
    with (PROJECT_ROOT / "configs" / "base.yaml").open(encoding="utf-8") as stream:
        base = yaml.safe_load(stream)
    with (PROJECT_ROOT / "configs" / "federated.yaml").open(encoding="utf-8") as stream:
        federated = yaml.safe_load(stream)["federated"]
    return base, federated


def load_warmstart_state(base: dict, federated: dict, map_location="cpu") -> tuple[dict, dict]:
    """Load the completed Stage 2 checkpoint and return its state and metadata."""
    artifact_dir = Path(base["data"]["artifacts_dir"])
    checkpoint_name = federated.get("warmstart_checkpoint", "warmstart_densenet121_best.pt")
    checkpoint_path = artifact_dir / checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Warm-start checkpoint not found: {checkpoint_path}. Complete Stage 2 first."
        )
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        metadata = payload
    else:
        state_dict = payload
        metadata = {}
    return copy.deepcopy(state_dict), metadata


def build_model_from_metadata(base: dict, metadata: dict) -> nn.Module:
    """Construct the exact Stage 2 DenseNet-121 architecture without downloading weights."""
    return build_densenet121(
        num_classes=int(metadata.get("num_classes", base["data"].get("num_classes", 4))),
        pretrained=False,
        classifier_hidden_units=int(metadata.get("classifier_hidden_units", 256)),
        dropout=float(metadata.get("dropout", 0.3)),
    )


def batchnorm_state_keys(model: nn.Module) -> set[str]:
    """Return every BatchNorm parameter and buffer key, including running stats."""
    keys: set[str] = set()
    for module_name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            prefix = f"{module_name}." if module_name else ""
            keys.update(key for key in model.state_dict() if key.startswith(prefix))
    return keys


def shared_state_dict(state_dict: dict, bn_keys: set[str]) -> dict:
    """Keep only state entries that the server is allowed to aggregate."""
    return {key: value.detach().cpu().clone() for key, value in state_dict.items() if key not in bn_keys}


def batchnorm_state_dict(state_dict: dict, bn_keys: set[str]) -> dict:
    """Keep only silo-local BatchNorm parameters and buffers."""
    return {key: value.detach().cpu().clone() for key, value in state_dict.items() if key in bn_keys}


def assemble_reference_state(warmstart_state: dict, shared_state: dict, bn_keys: set[str]) -> dict:
    """Combine global shared weights with warm-start BN as a portable reference model."""
    result = copy.deepcopy(warmstart_state)
    result.update({key: value.detach().cpu().clone() for key, value in shared_state.items()})
    # A server checkpoint cannot contain each silo's personalized BN without
    # receiving those local values. Preserve the warm-start BN as a clearly
    # identified reference; each client keeps its personalized BN locally.
    for key in bn_keys:
        if key not in result:
            raise KeyError(f"Warm-start checkpoint is missing BatchNorm key: {key}")
    return result


def save_torch_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
