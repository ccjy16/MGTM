"""Command-line configuration for the two MGTM workflows."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from mgtm.utils.paths import default_paths


def _intent_class_count(multichat_root: Path, mode: str) -> int:
    mapping = multichat_root / mode / "intent.txt"
    if not mapping.is_file():
        raise FileNotFoundError(f"MultiChat intent mapping not found: {mapping}")
    identifiers = []
    with mapping.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split("\t", 1)
            if len(fields) == 2:
                identifiers.append(int(fields[0]))
    if not identifiers:
        raise ValueError(f"MultiChat intent mapping is empty: {mapping}")
    return max(identifiers) + 1


def _common_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--valid-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--valid-num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-3)
    parser.add_argument("--precision", choices=("amp", "fp32"), default="amp")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--mask-ratio", type=float, default=0.0)
    parser.add_argument("--max-memory-items", type=int, default=256)
    parser.add_argument("--astra-semantic-weight", type=float, default=100.0)
    parser.add_argument("--astra-time-weight", type=float, default=0.35)


def _validate_astra_score_weights(args) -> None:
    for name in ("astra_semantic_weight", "astra_time_weight"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0.0:
            option = name.replace("_", "-")
            raise ValueError(f"--{option} must be finite and positive")


def build_multichat_parser() -> argparse.ArgumentParser:
    paths = default_paths()
    parser = argparse.ArgumentParser(description="Train MGTM on MultiChat")
    parser.add_argument("--data-mode", choices=("kuaile", "yongyuan"), default="kuaile")
    parser.add_argument("--data-root", type=Path, default=paths.multichat_root)
    parser.add_argument("--text-model", type=Path, default=paths.pretrained_root / "Taiyi-CLIP")
    parser.add_argument("--pvt-weights", type=Path, default=paths.pretrained_root / "PVT_v2_b2.pth")
    parser.add_argument("--max-steps", type=int, default=None)
    _common_training_arguments(parser)
    return parser


def build_dstc10_parser() -> argparse.ArgumentParser:
    paths = default_paths()
    parser = argparse.ArgumentParser(description="Train MGTM on DSTC10-MOD")
    parser.add_argument("--data-root", type=Path, default=paths.dstc10_root)
    parser.add_argument("--taiyi-text-model", type=Path, default=paths.pretrained_root / "Taiyi-CLIP")
    parser.add_argument("--pvt-weights", type=Path, default=paths.pretrained_root / "PVT_v2_b2.pth")
    parser.add_argument("--num-emotions", type=int, default=52)
    _common_training_arguments(parser)
    return parser


def parse_multichat_args(argv=None):
    return _finalize_multichat_args(build_multichat_parser().parse_args(argv))


def _finalize_multichat_args(args):
    paths = default_paths()
    args.data_root = args.data_root.resolve()
    lmdb_root = args.data_root / f"lmdb_{args.data_mode}_intent_style_attribute"
    args.train_data = str(lmdb_root / "train")
    args.val_data = str(lmdb_root / "val")
    args.test_data = str(lmdb_root / "test")
    args.output_root = (args.output_root or paths.output_root).resolve()
    args.taiyi_text_model_path = str(args.text_model.resolve())
    args.pvt_weights = args.pvt_weights.resolve()
    args.num_class = _intent_class_count(args.data_root, args.data_mode)
    args.text_feature_mode = "taiyi_CLIP"
    args.text_lr = 5e-6
    args.vision_lr = 1e-5
    args.warmup = 100
    args.log_interval = 10
    args.report_training_batch_acc = True
    args.use_augment = False
    args.online_memory_encode_batch_size = 64
    if args.max_steps is None:
        args.max_steps = 2**63 - 1
    _validate_astra_score_weights(args)
    return args


def parse_dstc10_args(argv=None):
    return _finalize_dstc10_args(build_dstc10_parser().parse_args(argv))


def _finalize_dstc10_args(args):
    paths = default_paths()
    args.data_root = args.data_root.resolve()
    args.output_root = (args.output_root or paths.output_root).resolve()
    args.taiyi_text_model = args.taiyi_text_model.resolve()
    args.pvt_weights = args.pvt_weights.resolve()
    _validate_astra_score_weights(args)
    return args
