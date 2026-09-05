"""Portable repository-relative paths used by MGTM commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepositoryPaths:
    repository_root: Path
    multichat_root: Path
    dstc10_root: Path
    pretrained_root: Path
    output_root: Path


@dataclass(frozen=True)
class TrainingOutputPaths:
    logs: Path
    results: Path
    checkpoints: Path


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_paths(root: Path | None = None) -> RepositoryPaths:
    base = Path(root).resolve() if root is not None else repository_root()
    return RepositoryPaths(
        repository_root=base,
        multichat_root=base / "data" / "MultiChat",
        dstc10_root=base / "data" / "DSTC10-MOD",
        pretrained_root=base / "pretrained_weights",
        output_root=base / "outputs",
    )


def build_training_output_paths(
    output_root: Path,
    *parts: str,
) -> TrainingOutputPaths:
    base = Path(output_root).joinpath(*parts)
    return TrainingOutputPaths(
        logs=base / "logs",
        results=base / "results",
        checkpoints=base / "checkpoints",
    )
