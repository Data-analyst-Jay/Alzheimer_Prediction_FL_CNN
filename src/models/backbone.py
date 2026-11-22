"""Shared DenseNet-121 model definition for warm-start and federated learning."""

from __future__ import annotations

from torch import nn
from torchvision import models
from torchvision.models import DenseNet121_Weights


def build_densenet121(
    num_classes: int = 4,
    *,
    pretrained: bool = True,
    classifier_hidden_units: int = 256,
    dropout: float = 0.3,
) -> nn.Module:
    """Build DenseNet-121 with the project's shared four-class prediction head."""
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, classifier_hidden_units),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(classifier_hidden_units, num_classes),
    )
    return model
