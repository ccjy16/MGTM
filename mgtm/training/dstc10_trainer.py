"""DSTC10-MOD training, ranking metrics, and evaluation."""

from __future__ import annotations

import logging
from pathlib import Path
from time import monotonic

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

def _encode_string_ids(row_ids, column_ids, device):
    id_codes = {
        value: index for index, value in enumerate(sorted({
            *(str(value) for value in row_ids),
            *(str(value) for value in column_ids),
        }))
    }
    rows = torch.tensor(
        [id_codes[str(value)] for value in row_ids], dtype=torch.long, device=device
    )
    columns = torch.tensor(
        [id_codes[str(value)] for value in column_ids], dtype=torch.long, device=device
    )
    return rows, columns


def _build_multi_positive_targets(target_ids, gallery_ids, device, dtype):
    row_codes, column_codes = _encode_string_ids(target_ids, gallery_ids, device)
    return row_codes.unsqueeze(1).eq(column_codes.unsqueeze(0)).to(dtype=dtype)


def compute_sticker_metrics(scores, target_ids, gallery_ids, ranked_indices=None):
    ranked_indices = (
        torch.argsort(scores, dim=1, descending=True)
        if ranked_indices is None else ranked_indices
    )
    relevant = _build_multi_positive_targets(
        target_ids, gallery_ids, scores.device, torch.bool
    )
    ranked_relevant = relevant.gather(1, ranked_indices)
    count = max(len(target_ids), 1)
    return {
        f"top{k}": float(
            ranked_relevant[:, :min(k, ranked_relevant.shape[1])]
            .any(dim=1).sum().item()
        ) / count
        for k in (1, 3, 5)
    }


def compute_emotion_metrics(scores, target_emotions, sticker_emotion_top2,
                            gallery_ids, valid_mask, ranked_indices=None):
    target_emotions = torch.as_tensor(
        target_emotions, dtype=torch.long, device=scores.device
    )
    valid_mask = torch.as_tensor(valid_mask, dtype=torch.bool, device=scores.device)
    emotion_values = sorted({
        int(emotion)
        for labels in sticker_emotion_top2.values()
        for emotion in labels
    } | {
        int(emotion) for emotion in target_emotions.tolist() if int(emotion) >= 0
    })
    emotion_to_row = {emotion: row for row, emotion in enumerate(emotion_values)}
    relevance_by_emotion = torch.zeros(
        len(emotion_values), len(gallery_ids), dtype=torch.bool, device=scores.device
    )
    for column, sticker_id in enumerate(gallery_ids):
        for emotion in sticker_emotion_top2.get(str(sticker_id), []):
            row = emotion_to_row.get(int(emotion))
            if row is not None:
                relevance_by_emotion[row, column] = True
    target_rows = torch.tensor(
        [emotion_to_row.get(int(emotion), -1) for emotion in target_emotions.tolist()],
        dtype=torch.long,
        device=scores.device,
    )
    safe_rows = target_rows.clamp_min(0)
    sample_relevance = relevance_by_emotion.index_select(0, safe_rows)
    sample_relevance &= target_rows.unsqueeze(1).ge(0)
    relevant_counts = sample_relevance.sum(dim=1)
    included = valid_mask & target_emotions.ge(0) & relevant_counts.gt(0)
    count = int(included.sum().item())
    if count == 0:
        return {
            "emotion_mAP": 0.0,
            "emotion_top1": 0.0,
            "emotion_top3": 0.0,
            "emotion_top5": 0.0,
            "emotion_count": 0,
        }
    ranked_indices = (
        torch.argsort(scores, dim=1, descending=True)
        if ranked_indices is None else ranked_indices
    )
    ranked_relevant = sample_relevance.gather(1, ranked_indices)
    cumulative_hits = ranked_relevant.cumsum(dim=1)
    ranks = torch.arange(
        1, scores.shape[1] + 1, dtype=torch.float64, device=scores.device
    )
    precision = cumulative_hits.to(torch.float64) / ranks.unsqueeze(0)
    average_precision = (
        (precision * ranked_relevant).sum(dim=1)
        / relevant_counts.clamp_min(1).to(torch.float64)
    )
    result = {
        "emotion_mAP": float(average_precision[included].mean().item()),
        "emotion_count": count,
    }
    for k in (1, 3, 5):
        hits = ranked_relevant[:, :min(k, scores.shape[1])].any(dim=1)
        result[f"emotion_top{k}"] = float(hits[included].float().mean().item())
    return result


def _emotion_multi_positive_bce(similarity, emotion_ids, valid_mask):
    emotion_ids = torch.as_tensor(
        emotion_ids, dtype=torch.long, device=similarity.device
    )
    valid_mask = torch.as_tensor(
        valid_mask, dtype=torch.bool, device=similarity.device
    ) & emotion_ids.ge(0)
    valid_indices = valid_mask.nonzero(as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return similarity.sum() * 0.0

    valid_similarity = similarity.index_select(0, valid_indices).index_select(
        1, valid_indices
    )
    valid_emotions = emotion_ids.index_select(0, valid_indices)
    targets = valid_emotions.unsqueeze(1).eq(valid_emotions.unsqueeze(0)).to(
        dtype=valid_similarity.dtype
    )
    return F.binary_cross_entropy_with_logits(valid_similarity, targets)


def select_training_loss(retrieval_loss, emotion_loss, use_intent_loss):
    if not use_intent_loss:
        return retrieval_loss
    if emotion_loss is None:
        raise ValueError("emotion_loss is required when intent loss is enabled")
    return (retrieval_loss + emotion_loss) / 2.0


def encode_text_batch(model, batch, device, memory_pool, max_memory_items=256):
    texts = batch["texts"]
    if isinstance(texts, torch.Tensor):
        texts = texts.to(device, non_blocking=True)
    turn_ids = batch["turn_ids"].to(device, non_blocking=True)
    pool = memory_pool.query(
        batch["dialogue_ids"], batch["turn_ids"].tolist(), device,
        max_pool_size=max_memory_items,
    ) if memory_pool is not None else (None, None, None)
    intent_features, flag, text_features = model(
        1, None, text=texts, current_session_id=turn_ids,
        pool_embeddings=pool[0], pool_session_ids=pool[1],
        pool_valid_mask=pool[2], mask_ratio=0.0,
    )
    final_text_features = model.fuse_retrieval_query(
        text_features, intent_features
    )
    return final_text_features, intent_features, flag


def _encode_batch(model, batch, device, memory_pool, mask_ratio=0.0,
                  max_memory_items=256):
    images = batch["images"].to(device, non_blocking=True)
    texts = batch["texts"]
    if isinstance(texts, torch.Tensor):
        texts = texts.to(device, non_blocking=True)
    pool = memory_pool.query(
        batch["dialogue_ids"], batch["turn_ids"].tolist(), device,
        max_pool_size=max_memory_items,
    ) if memory_pool is not None else (None, None, None)
    intent_features, flag, text_features = model(
        1, images, text=texts,
        current_session_id=batch["turn_ids"].to(device, non_blocking=True),
        pool_embeddings=pool[0], pool_session_ids=pool[1],
        pool_valid_mask=pool[2], mask_ratio=mask_ratio,
    )
    image_features, final_text_features, logit_scale = model(
        0, images, text=texts, intent_features=intent_features,
        precomputed_text_features=text_features, mask_ratio=mask_ratio,
    )
    return image_features, final_text_features, logit_scale, intent_features, flag


def train_epoch(model, classifier_model, dataloader, optimizer, scaler, device,
                memory_pool, gallery_ids, args, epoch=None):
    model.train()
    use_intent_loss = getattr(args, "use_intent_loss", True)
    classifier_model.train(use_intent_loss)
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    started = monotonic()
    total_steps = len(dataloader)
    log_every_steps = max(0, int(getattr(args, "log_every_steps", 0)))
    for step, batch in enumerate(dataloader, start=1):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=scaler is not None):
            image_features, text_features, logit_scale, intent_features, flag = _encode_batch(
                model, batch, device, memory_pool, getattr(args, "mask_ratio", 0.0),
                getattr(args, "max_memory_items", 256),
            )
            similarity = logit_scale.mean() * image_features @ text_features.t()
            retrieval_loss = _emotion_multi_positive_bce(
                similarity, batch["emotion_ids"], batch["emotion_valid"]
            )
            emotion_loss = None
            if use_intent_loss:
                emotion_logits, _ = classifier_model(intent_features, flag)
                emotion_ids = batch["emotion_ids"].to(device, non_blocking=True)
                valid_cpu = batch["emotion_valid"]
                valid = valid_cpu.to(device, non_blocking=True)
                emotion_loss = (
                    F.cross_entropy(emotion_logits[valid], emotion_ids[valid])
                    if bool(valid_cpu.any()) else image_features.sum() * 0.0
                )
            loss = select_training_loss(
                retrieval_loss, emotion_loss, use_intent_loss
            )
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.detach().to(torch.float64)
        if log_every_steps and (
            step == 1 or step % log_every_steps == 0 or step == total_steps
        ):
            elapsed = max(monotonic() - started, 1e-9)
            logging.info(
                "Epoch %s train progress: %d/%d (%.2f%%), "
                "average_loss=%.6f, steps_per_second=%.2f",
                "?" if epoch is None else epoch + 1,
                step,
                total_steps,
                100.0 * step / max(total_steps, 1),
                float((total_loss / step).item()),
                step / elapsed,
            )
    return float((total_loss / max(len(dataloader), 1)).item())


def load_gallery_inputs(root, resolution):
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        ),
    ])
    image_dir = Path(root) / "images"
    gallery_ids = sorted(path.stem for path in image_dir.glob("*.png"))
    images = []
    for sticker_id in gallery_ids:
        with Image.open(image_dir / f"{sticker_id}.png") as image:
            images.append(transform(image.convert("RGB")))
    return gallery_ids, torch.stack(images)


def _should_log_evaluation_progress(step, total_steps, interval=10):
    return step == 1 or step % interval == 0 or step == total_steps


@torch.no_grad()
def evaluate(model, dataloader, dataset_root, sticker_emotions, device,
             memory_pool, resolution=224, gallery_cache=None,
             gallery_inputs=None, max_memory_items=256):
    model.eval()
    if gallery_inputs is not None:
        gallery_ids, gallery_images = gallery_inputs
    elif gallery_cache is None:
        gallery_ids, gallery_images = load_gallery_inputs(dataset_root, resolution)
    else:
        gallery_ids = sorted(
            path.stem for path in (Path(dataset_root) / "images").glob("*.png")
        )
        gallery_images = None
    if gallery_cache is None:
        gallery_chunks = []
        for start in range(0, len(gallery_images), 32):
            features = model.encode_image(
                gallery_images[start:start + 32].to(device)
            )
            gallery_chunks.append(F.normalize(features, dim=-1))
        gallery_cache = torch.cat(gallery_chunks, dim=0)
    rows = []
    target_ids = []
    target_emotions = []
    valid_emotions = []
    total_steps = len(dataloader)
    for step, batch in enumerate(dataloader, start=1):
        if _should_log_evaluation_progress(step, total_steps):
            logging.info("================================")
            logging.info("Eval batch: %d/%d", step, total_steps)
        text_features, _, _ = encode_text_batch(
            model, batch, device, memory_pool,
            max_memory_items=max_memory_items,
        )
        rows.append(F.normalize(text_features, dim=-1) @ gallery_cache.T)
        target_ids.extend(batch["target_sticker_ids"])
        target_emotions.extend(batch["emotion_ids"].tolist())
        valid_emotions.extend(batch["emotion_valid"].tolist())
    scores = torch.cat(rows, dim=0) if rows else torch.empty(0, len(gallery_ids))
    scores = scores.cpu()
    ranked_indices = torch.argsort(scores, dim=1, descending=True)
    metrics = compute_sticker_metrics(
        scores, target_ids, gallery_ids, ranked_indices=ranked_indices
    )
    metrics.update(compute_emotion_metrics(
        scores, target_emotions, sticker_emotions, gallery_ids, valid_emotions,
        ranked_indices=ranked_indices,
    ))
    metrics["emotion_top_avg"] = (
        metrics["emotion_top1"] + metrics["emotion_top3"] + metrics["emotion_top5"]
    ) / 3.0
    metrics["map_emotion_score"] = (
        metrics["emotion_mAP"] + metrics["emotion_top_avg"]
    ) / 2.0
    return metrics, gallery_cache
