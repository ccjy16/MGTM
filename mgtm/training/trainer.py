"""MultiChat optimization and retrieval evaluation for MGTM."""

from __future__ import annotations

import logging
from time import monotonic

import torch
from torch import nn
import torch.nn.functional as F

from mgtm.training.retrieval_targets import (
    build_image_intent_index,
    build_intent_positive_targets,
)


def amp_enabled(scaler):
    return scaler is not None


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    return value


def query_memory_pool(memory_pool, session_ids, device, args):
    if memory_pool is None:
        return None, None, None
    return memory_pool.query(
        session_ids,
        device,
        max_pool_size=args.max_memory_items,
    )


def get_loss(
    model,
    classifier_model,
    images,
    session_ids,
    image_ids,
    raw_texts,
    intent_ids,
    args,
    *,
    memory_pool=None,
    image_intents=None,
):
    device = next(model.parameters()).device
    images = to_device(images, device)
    raw_texts = to_device(raw_texts, device)
    session_ids = session_ids.to(device, non_blocking=True)
    intent_ids = torch.as_tensor(intent_ids, dtype=torch.long, device=device)
    pool_embeddings, pool_positions, pool_valid_mask = query_memory_pool(
        memory_pool, session_ids, device, args
    )
    auxiliary_features, has_memory, query_features = model(
        1,
        images,
        text=raw_texts,
        current_session_id=session_ids,
        pool_embeddings=pool_embeddings,
        pool_session_ids=pool_positions,
        pool_valid_mask=pool_valid_mask,
        mask_ratio=args.mask_ratio,
    )
    intent_logits, predicted_labels = classifier_model(
        auxiliary_features, has_memory
    )
    intent_loss = F.cross_entropy(intent_logits, intent_ids)
    image_features, retrieval_query, logit_scale = model(
        0,
        images,
        text=raw_texts,
        intent_features=auxiliary_features,
        precomputed_text_features=query_features,
        mask_ratio=args.mask_ratio,
    )
    similarity = logit_scale.mean() * image_features @ retrieval_query.T
    targets = build_intent_positive_targets(
        image_ids,
        intent_ids,
        image_intents,
        device=device,
        dtype=similarity.dtype,
    )
    retrieval_loss = F.binary_cross_entropy_with_logits(similarity, targets)
    loss = (intent_loss + retrieval_loss) / 2.0
    metrics = None
    if args.report_training_batch_acc:
        predicted_text = similarity.argmax(dim=1, keepdim=True)
        predicted_image = similarity.argmax(dim=0)
        query_indices = torch.arange(similarity.shape[0], device=device)
        metrics = {
            "i2t": targets.gather(1, predicted_text).float().mean(),
            "t2i": targets[predicted_image, query_indices].float().mean(),
            "intent_acc": float(
                predicted_labels.eq(intent_ids).float().mean().item()
            ),
        }
    return loss, metrics


def train(
    model,
    classifier_model,
    data,
    epoch,
    optimizer,
    scaler,
    scheduler,
    args,
    global_trained_steps,
    *,
    memory_pool=None,
):
    model.train()
    classifier_model.train()
    dataloader = data["train"].dataloader
    dataset = dataloader.dataset
    image_intents = build_image_intent_index(
        [dataset.img_ids[sample_id] for sample_id in dataset.sample_ids],
        [dataset.fined_intents[sample_id] for sample_id in dataset.sample_ids],
    )
    device = next(model.parameters()).device
    losses = []
    started = monotonic()
    trained_steps = 0
    for batch_index, batch in enumerate(dataloader):
        step = global_trained_steps + trained_steps
        if step >= args.max_steps:
            break
        scheduler(step)
        session_ids, image_ids, images, raw_texts, intent_ids, _ = batch
        images = images.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled(scaler)):
            loss, batch_metrics = get_loss(
                model,
                classifier_model,
                images,
                session_ids,
                image_ids,
                raw_texts,
                intent_ids,
                args,
                memory_pool=memory_pool,
                image_intents=image_intents,
            )
        if not torch.isfinite(loss):
            logging.warning("Skipping non-finite loss at global step %d", step)
            continue
        parameters = list(model.parameters()) + list(classifier_model.parameters())
        if scaler is None:
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        losses.append(float(loss.detach().item()))
        trained_steps += 1
        if (step + 1) % args.log_interval == 0:
            elapsed = max(monotonic() - started, 1e-9)
            metric_text = ""
            if batch_metrics is not None:
                metric_text = f" intent_acc={batch_metrics['intent_acc']:.4f}"
            logging.info(
                "Epoch %d step %d/%d loss=%.6f steps_per_second=%.2f%s",
                epoch + 1,
                batch_index + 1,
                dataloader.num_batches,
                losses[-1],
                trained_steps / elapsed,
                metric_text,
            )
    mean_loss = sum(losses) / max(len(losses), 1)
    return trained_steps, mean_loss


def compute_retrieval_metrics(
    similarity,
    target_image_ids,
    target_intent_ids,
    candidate_ids,
    candidate_intents,
    candidate_intent_matrix=None,
    candidate_index=None,
):
    device = similarity.device
    batch_size, candidate_count = similarity.shape
    ranked_indices = torch.argsort(similarity, dim=1, descending=True)
    target_intent_ids = target_intent_ids.to(device=device, dtype=torch.long)
    if candidate_intent_matrix is None:
        candidate_intent_matrix = build_candidate_intent_matrix(
            candidate_ids, candidate_intents, device
        )
    relevance = torch.zeros(
        (batch_size, candidate_count), dtype=torch.bool, device=device
    )
    valid_targets = (
        (target_intent_ids >= 0)
        & (target_intent_ids < candidate_intent_matrix.shape[1])
    )
    if valid_targets.any():
        relevance[valid_targets] = candidate_intent_matrix[
            :, target_intent_ids[valid_targets]
        ].T
    ranked_relevance = relevance.gather(1, ranked_indices)
    positions = torch.arange(
        1, candidate_count + 1, device=device, dtype=torch.float32
    ).unsqueeze(0)
    precision = ranked_relevance.cumsum(dim=1).float() / positions
    relevant_counts = relevance.sum(dim=1)
    average_precision = (
        (precision * ranked_relevance).sum(dim=1)
        / relevant_counts.clamp_min(1).float()
    )
    average_precision = torch.where(
        relevant_counts > 0,
        average_precision,
        torch.zeros_like(average_precision),
    )
    if candidate_index is None:
        candidate_index = {
            image_id: index for index, image_id in enumerate(candidate_ids)
        }
    target_indices = torch.tensor(
        [candidate_index.get(image_id, -1) for image_id in target_image_ids],
        device=device,
    )
    result = {"ap_sum": float(average_precision.sum().item())}
    for top_k in (1, 3, 5):
        width = min(top_k, candidate_count)
        top_indices = ranked_indices[:, :width]
        result[f"top{top_k}_count"] = int(
            (top_indices == target_indices.unsqueeze(1)).any(dim=1).sum().item()
        )
        result[f"intent_top{top_k}_count"] = int(
            ranked_relevance[:, :width].any(dim=1).sum().item()
        )
    return result


def build_candidate_intent_matrix(candidate_ids, candidate_intents, device):
    normalized = []
    max_intent = -1
    for candidate_id in candidate_ids:
        intents = candidate_intents.get(candidate_id, [])
        if not isinstance(intents, list):
            intents = list(intents) if isinstance(intents, tuple) else [intents]
        values = [int(value) for value in intents]
        normalized.append(values)
        if values:
            max_intent = max(max_intent, max(values))
    matrix = torch.zeros(
        (len(candidate_ids), max_intent + 1), dtype=torch.bool, device=device
    )
    for row, values in enumerate(normalized):
        if values:
            matrix[row, torch.tensor(values, device=device)] = True
    return matrix


@torch.no_grad()
def evaluate(
    model,
    data,
    epoch,
    args,
    classifier_model,
    is_test=0,
    memory_pool=None,
    collect_memory_diagnostics=False,
):
    split = "test" if is_test else "val"
    logging.info("Evaluating %s split after epoch %d", split, epoch + 1)
    model.eval()
    classifier_model.eval()
    dataloader = data[split].dataloader
    dataset = data[split].dataset
    device = next(model.parameters()).device
    candidate_ids = list(dataset.all_image_features)
    candidate_index = {
        image_id: index for index, image_id in enumerate(candidate_ids)
    }
    candidate_intents = dataset.intent2id
    candidate_intent_matrix = build_candidate_intent_matrix(
        candidate_ids, candidate_intents, device
    )
    image_tensors = []
    for image_id in candidate_ids:
        tensor = dataset.all_image_features[image_id]
        image_tensors.append(tensor.unsqueeze(0) if tensor.ndim == 3 else tensor)
    all_images = torch.cat(image_tensors, dim=0).to(device)
    image_features = []
    image_batch_size = max(args.valid_batch_size, 16)
    for start in range(0, len(all_images), image_batch_size):
        features = model.encode_image(all_images[start:start + image_batch_size])
        image_features.append(F.normalize(features, dim=-1))
    gallery = torch.cat(image_features, dim=0)
    totals = {
        "ap_sum": 0.0,
        "top1_count": 0,
        "top3_count": 0,
        "top5_count": 0,
        "intent_top1_count": 0,
        "intent_top3_count": 0,
        "intent_top5_count": 0,
    }
    query_count = 0
    diagnostic_valid = 0.0
    diagnostic_selected = 0.0
    diagnostic_queries = 0
    diagnostic_time_scale = None
    diagnostic_decay_rates = None
    for batch in dataloader:
        session_ids, image_ids, images, raw_texts, intent_ids, _ = batch
        session_ids = session_ids.to(device, non_blocking=True)
        pool = query_memory_pool(memory_pool, session_ids, device, args)
        output = model(
            1,
            images.to(device, non_blocking=True),
            text=raw_texts,
            current_session_id=session_ids,
            pool_embeddings=pool[0],
            pool_session_ids=pool[1],
            pool_valid_mask=pool[2],
            return_memory_diagnostics=collect_memory_diagnostics,
        )
        if collect_memory_diagnostics:
            auxiliary, _, query_features, diagnostics = output
            if diagnostics is not None:
                diagnostic_valid += diagnostics["valid_count"].sum().item()
                diagnostic_selected += diagnostics["selected_count"].sum().item()
                diagnostic_queries += diagnostics["valid_count"].shape[0]
                diagnostic_time_scale = float(diagnostics["time_scale"].item())
                diagnostic_decay_rates = diagnostics["decay_rates"].cpu().tolist()
        else:
            auxiliary, _, query_features = output
        retrieval_query = model.fuse_retrieval_query(query_features, auxiliary)
        similarity = retrieval_query @ gallery.T
        batch_metrics = compute_retrieval_metrics(
            similarity,
            image_ids,
            intent_ids,
            candidate_ids,
            candidate_intents,
            candidate_intent_matrix,
            candidate_index,
        )
        for key in totals:
            totals[key] += batch_metrics[key]
        query_count += len(image_ids)
    denominator = max(query_count, 1)
    metrics = {
        "mAP": totals["ap_sum"] / denominator,
        **{
            key.removesuffix("_count"): value / denominator
            for key, value in totals.items()
            if key != "ap_sum"
        },
    }
    if collect_memory_diagnostics:
        metrics["memory_diagnostics"] = {
            "mean_valid_count": diagnostic_valid / max(diagnostic_queries, 1),
            "mean_selected_count": diagnostic_selected / max(diagnostic_queries, 1),
            "time_scale": diagnostic_time_scale,
            "decay_rates": diagnostic_decay_rates,
        }
    return metrics
