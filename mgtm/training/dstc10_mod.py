"""Direct MGTM training on DSTC10-MOD."""

from __future__ import annotations

import json
import logging
import random

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

from mgtm.data.dstc10 import (
    DSTC10MODDataset,
    DSTCSessionMemoryPool,
    MemoryLengthSampler,
    collate_dstc10,
)
from mgtm.models.mgtm import Classifier, MGTMModel, load_pvtv2_weights
from mgtm.training.config import parse_dstc10_args
from mgtm.training.dstc10_trainer import evaluate, load_gallery_inputs, train_epoch
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
        taiyi_text_model_path=str(args.taiyi_text_model),
        context_length=args.context_length,
        init_time_scale=64.0,
        astra_semantic_weight=args.astra_semantic_weight,
        astra_time_weight=args.astra_time_weight,
    ).to(device)
    load_pvtv2_weights(model.visual, args.pvt_weights)
    return model


def _build_memory_embeddings(model, pool, device, batch_size=64):
    keys = sorted(pool.texts)
    if not keys:
        return
    was_training = model.training
    model.eval()
    with torch.inference_mode():
        embeddings = []
        for start in range(0, len(keys), batch_size):
            texts = [pool.texts[key] for key in keys[start:start + batch_size]]
            _, features = model.encode_text(texts, return_pooled=True)
            embeddings.append(features.detach().cpu().float())
    pool.set_embedding_tensor(keys, torch.cat(embeddings, dim=0))
    model.train(was_training)


def _loader(dataset, batch_size, workers, shuffle, max_memory_items=256):
    sampler = (
        MemoryLengthSampler(dataset, max_memory_items=max_memory_items)
        if not shuffle
        else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=workers,
        persistent_workers=workers > 0,
        collate_fn=collate_dstc10,
    )


def _load_sticker_emotions(data_root):
    path = data_root / "labels" / "sticker_emotion_top2_train.json"
    if not path.is_file():
        raise FileNotFoundError(f"DSTC10-MOD sticker emotion labels not found: {path}")
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _build_datasets(args, splits):
    return {
        split: DSTC10MODDataset(
            args.data_root,
            split,
            resolution=args.resolution,
            load_images=split == "train",
        )
        for split in splits
    }


def _checkpoint_config(args):
    return {
        "dataset": "DSTC10-MOD",
        "context_length": args.context_length,
        "resolution": args.resolution,
        "astra_semantic_weight": args.astra_semantic_weight,
        "astra_time_weight": args.astra_time_weight,
        "num_emotions": args.num_emotions,
        "initialization": "base_taiyi_text_and_pvtv2",
    }


def train_main(args):
    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(args, device)
    classifier = Classifier(512, args.num_emotions).to(device)
    datasets = _build_datasets(args, ("train", "val", "test"))
    memory_pool = DSTCSessionMemoryPool.from_datasets(datasets.values())
    loaders = {
        "train": _loader(
            datasets["train"], args.batch_size, args.num_workers, True,
            max_memory_items=args.max_memory_items,
        ),
        "val": _loader(
            datasets["val"], args.valid_batch_size, args.valid_num_workers, False,
            max_memory_items=args.max_memory_items,
        ),
        "test": _loader(
            datasets["test"], args.valid_batch_size, args.valid_num_workers, False,
            max_memory_items=args.max_memory_items,
        ),
    }
    gallery_inputs = load_gallery_inputs(args.data_root, args.resolution)
    gallery_ids = gallery_inputs[0]
    sticker_emotions = _load_sticker_emotions(args.data_root)
    optimizer = optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()),
        lr=args.lr,
        weight_decay=args.wd,
    )
    scaler = (
        torch.amp.GradScaler("cuda")
        if args.precision == "amp" and device.type == "cuda"
        else None
    )
    args.use_intent_loss = True
    args.log_every_steps = 100
    output = build_training_output_paths(args.output_root, "dstc10_mod")
    last_checkpoint = None
    for epoch in range(args.max_epochs):
        _build_memory_embeddings(model, memory_pool, device)
        train_loss = train_epoch(
            model,
            classifier,
            loaders["train"],
            optimizer,
            scaler=scaler,
            device=device,
            memory_pool=memory_pool,
            gallery_ids=gallery_ids,
            args=args,
            epoch=epoch,
        )
        _build_memory_embeddings(model, memory_pool, device)
        validation, gallery_cache = evaluate(
            model,
            loaders["val"],
            args.data_root,
            sticker_emotions,
            device,
            memory_pool,
            resolution=args.resolution,
            gallery_inputs=gallery_inputs,
            max_memory_items=args.max_memory_items,
        )
        test_metrics, _ = evaluate(
            model,
            loaders["test"],
            args.data_root,
            sticker_emotions,
            device,
            memory_pool,
            resolution=args.resolution,
            gallery_cache=gallery_cache,
            gallery_inputs=gallery_inputs,
            max_memory_items=args.max_memory_items,
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
    args = parse_dstc10_args(argv)
    return train_main(args)
