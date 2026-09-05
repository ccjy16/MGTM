"""MGTM training on MultiChat."""

from __future__ import annotations

import json
import logging
import random

import numpy as np
import torch
from torch import optim

from mgtm.data.datasets import SessionSummaryPool, get_data
from mgtm.models.mgtm import Classifier, MGTMModel, load_pvtv2_weights
from mgtm.training.config import parse_multichat_args
from mgtm.training.scheduler import cosine_lr
from mgtm.training.trainer import evaluate, train
from mgtm.utils.checkpoint import save_checkpoint
from mgtm.utils.paths import build_training_output_paths


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model(args, device):
    model = MGTMModel(
        embed_dim=512,
        image_resolution=args.resolution,
        taiyi_text_model_path=str(args.taiyi_text_model_path),
        context_length=args.context_length,
        init_time_scale=64.0,
        astra_semantic_weight=args.astra_semantic_weight,
        astra_time_weight=args.astra_time_weight,
    ).to(device)
    load_pvtv2_weights(model.visual, args.pvt_weights)
    classifier = Classifier(512, args.num_class).to(device)
    return model, classifier


def _build_memory_pool(data):
    pool = SessionSummaryPool()
    for split in data.values():
        pool.get_unique_sessions_from_dataset(split.dataset)
    return pool


def _refresh_memory(pool, model, device, batch_size):
    pool.refresh_online_cache(
        model,
        device,
        encode_batch_size=batch_size,
    )


def _checkpoint_config(args):
    return {
        "dataset": "MultiChat",
        "data_mode": args.data_mode,
        "context_length": args.context_length,
        "resolution": args.resolution,
        "astra_semantic_weight": args.astra_semantic_weight,
        "astra_time_weight": args.astra_time_weight,
    }


def train_main(args):
    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classifier = _load_model(args, device)
    data = get_data(args, epoch_id=0, max_txt_length=args.context_length)
    memory_pool = _build_memory_pool(data)
    train_steps = data["train"].dataloader.num_batches * args.max_epochs
    args.max_steps = min(args.max_steps, train_steps)
    optimizer = optim.AdamW(
        [
            {
                "params": model.taiyi_text_tower.parameters(),
                "lr": args.text_lr,
                "lr_scale": args.text_lr / args.lr,
            },
            {
                "params": model.visual.parameters(),
                "lr": args.vision_lr,
                "lr_scale": args.vision_lr / args.lr,
            },
            {
                "params": [
                    parameter
                    for name, parameter in model.named_parameters()
                    if not name.startswith(("taiyi_text_tower.", "visual."))
                ] + list(classifier.parameters()),
                "lr": args.lr,
                "lr_scale": 1.0,
            },
        ],
        lr=args.lr,
        weight_decay=args.wd,
    )
    scheduler = cosine_lr(optimizer, args.lr, args.warmup, max(args.max_steps, 1))
    scaler = (
        torch.amp.GradScaler("cuda")
        if args.precision == "amp" and device.type == "cuda"
        else None
    )
    output = build_training_output_paths(
        args.output_root, "multichat", args.data_mode
    )
    last_checkpoint = None
    trained_steps = 0
    for epoch in range(args.max_epochs):
        _refresh_memory(
            memory_pool,
            model,
            device,
            args.online_memory_encode_batch_size,
        )
        epoch_steps, train_loss = train(
            model,
            classifier,
            data,
            epoch,
            optimizer,
            scaler,
            scheduler,
            args,
            trained_steps,
            memory_pool=memory_pool,
        )
        trained_steps += epoch_steps
        _refresh_memory(
            memory_pool,
            model,
            device,
            args.online_memory_encode_batch_size,
        )
        validation = evaluate(
            model,
            data,
            epoch,
            args,
            classifier,
            is_test=0,
            memory_pool=memory_pool,
        )
        test_metrics = evaluate(
            model,
            data,
            epoch,
            args,
            classifier,
            is_test=1,
            memory_pool=memory_pool,
        )
        epoch_metrics = {
            "train_loss": train_loss,
            "val": validation,
            "test": test_metrics,
        }
        last_checkpoint = output.checkpoints / f"mgtm_epoch_{epoch + 1:03d}.pt"
        save_checkpoint(
            last_checkpoint,
            model,
            classifier,
            epoch=epoch + 1,
            metrics=epoch_metrics,
            config=_checkpoint_config(args),
        )
        logging.info(
            "Epoch %d metrics: %s",
            epoch + 1,
            json.dumps(epoch_metrics),
        )
    return last_checkpoint


def main(argv=None):
    logging.basicConfig(level=logging.INFO)
    args = parse_multichat_args(argv)
    return train_main(args)
