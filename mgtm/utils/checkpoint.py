"""Checkpoint serialization for MGTM runs."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path, model, classifier, *, epoch, metrics=None, config=None):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "epoch": int(epoch),
        "state_dict": model.state_dict(),
        "classifier_state_dict": classifier.state_dict() if classifier is not None else None,
        "metrics": dict(metrics or {}),
        "config": dict(config or {}),
    }
    torch.save(payload, destination)
    return destination
