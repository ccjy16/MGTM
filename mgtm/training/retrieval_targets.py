import torch


def build_image_intent_index(image_ids, intent_ids):
    if isinstance(intent_ids, torch.Tensor):
        intent_ids = intent_ids.detach().cpu().tolist()
    if len(image_ids) != len(intent_ids):
        raise ValueError("image_ids and intent_ids must have the same length")

    image_intents = {}
    for image_id, intent_id in zip(image_ids, intent_ids):
        image_intents.setdefault(image_id, set()).add(int(intent_id))
    return {
        image_id: tuple(sorted(intents))
        for image_id, intents in image_intents.items()
    }


def build_intent_positive_targets(
    image_ids,
    query_intent_ids,
    image_intents,
    device,
    dtype,
):
    if isinstance(query_intent_ids, torch.Tensor):
        query_intent_ids = query_intent_ids.detach().cpu().tolist()
    query_intent_ids = [int(intent_id) for intent_id in query_intent_ids]

    missing = [image_id for image_id in image_ids if image_id not in image_intents]
    if missing:
        raise KeyError(f"Missing training intent labels for image IDs: {missing[:5]}")

    targets = [
        [intent_id in image_intents[image_id] for intent_id in query_intent_ids]
        for image_id in image_ids
    ]
    return torch.tensor(targets, device=device, dtype=dtype)
