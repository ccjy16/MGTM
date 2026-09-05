"""DSTC10-MOD dataset and dialogue-local memory utilities."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms


def format_dstc_turn(turn: dict) -> str:
    return f"{turn['speaker_id']}: {turn.get('txt', '')}".rstrip()


def build_dstc_query(turns: list[dict], target_turn_id: int) -> str:
    return "[SEP]".join(
        format_dstc_turn(turn) for turn in turns[: target_turn_id + 1]
    )


class DSTC10MODDataset(Dataset):
    def __init__(self, root, split="train", max_txt_length=512,
                 use_augment=False, resolution=224, load_images=True,
                 cache_images=False, image_cache=None):
        del max_txt_length
        self.root = Path(root)
        self.split = split
        self.load_images = bool(load_images)
        self.cache_images = bool(cache_images and load_images and not use_augment)
        self.dialogues = self._load_dialogues()
        self.records = self._load_split()
        self.number_samples = len(self.records)
        self.dataset_len = self.number_samples
        self.sample_ids = [record["sample_id"] for record in self.records]
        self.raw_text_strings = {
            record["sample_id"]: record["query_text"] for record in self.records
        }
        self._transform = self._build_transform(resolution, use_augment)
        self._image_cache = (
            self._build_image_cache(image_cache) if self.cache_images else {}
        )
        self.time_deltas = [
            int(record["turn_id"]) - int(memory_turn_id)
            for record in self.records
            for memory_turn_id in record["memory_turn_ids"]
        ]

    def _load_dialogues(self):
        path = self.root / "dialogues.jsonl"
        result = {}
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                result[str(record["dialogue_id"])] = list(record["turns"])
        return result

    def _load_split(self):
        path = self.root / "splits" / f"{self.split}.jsonl"
        records = []
        with path.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if not line.strip():
                    continue
                source = json.loads(line)
                dialogue_id = str(source["dialogue_id"])
                turn_id = int(source["turn_id"])
                turns = self.dialogues[dialogue_id]
                memory_turn_ids = list(range(turn_id))
                records.append({
                    **source,
                    "sample_id": str(source["sample_id"]),
                    "dialogue_id": dialogue_id,
                    "turn_id": turn_id,
                    "target_sticker_id": str(source["target_sticker_id"]),
                    "emotion_id": int(source.get("emotion_id", -1)),
                    "emotion_valid": bool(source.get("emotion_valid", False)),
                    "memory_turn_ids": memory_turn_ids,
                    "query_text": build_dstc_query(turns, turn_id),
                })
        return records

    @staticmethod
    def _build_transform(resolution, use_augment):
        operations = [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ]
        if use_augment:
            operations.insert(0, transforms.RandomHorizontalFlip())
        return transforms.Compose(operations)

    def _load_image(self, relative_path):
        with Image.open(self.root / relative_path) as image:
            return self._transform(image.convert("RGB"))

    @staticmethod
    def _image_key(relative_path):
        return Path(relative_path).as_posix()

    def _build_image_cache(self, image_cache=None):
        cache = {
            self._image_key(relative_path): tensor
            for relative_path, tensor in (image_cache or {}).items()
        }
        for relative_path in sorted({
            record["target_image_path"] for record in self.records
        }):
            key = self._image_key(relative_path)
            if key not in cache:
                cache[key] = self._load_image(relative_path)
        return cache

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        record = self.records[index % self.number_samples]
        image_tensor = None
        if self.load_images:
            image_tensor = self._image_cache.get(
                self._image_key(record["target_image_path"])
            )
            if image_tensor is None:
                image_tensor = self._load_image(record["target_image_path"])
        return {
            "dialogue_id": record["dialogue_id"],
            "turn_id": record["turn_id"],
            "sample_id": record["sample_id"],
            "target_sticker_id": record["target_sticker_id"],
            "image": image_tensor,
            "text": record["query_text"],
            "query_text": record["query_text"],
            "emotion_id": record["emotion_id"],
            "emotion_valid": record["emotion_valid"],
        }


def collate_dstc10(batch):
    first = batch[0]
    texts = [item["text"] for item in batch]
    if isinstance(first["text"], torch.Tensor):
        texts = torch.stack(texts)
    images = None
    if first["image"] is not None:
        images = torch.stack([item["image"] for item in batch])
    return {
        "dialogue_ids": [item["dialogue_id"] for item in batch],
        "turn_ids": torch.tensor([item["turn_id"] for item in batch], dtype=torch.long),
        "sample_ids": [item["sample_id"] for item in batch],
        "target_sticker_ids": [item["target_sticker_id"] for item in batch],
        "images": images,
        "texts": texts,
        "emotion_ids": torch.tensor([item["emotion_id"] for item in batch], dtype=torch.long),
        "emotion_valid": torch.tensor([item["emotion_valid"] for item in batch], dtype=torch.bool),
    }


class MemoryLengthSampler(Sampler):
    def __init__(self, dataset, max_memory_items=256):
        self.dataset = dataset
        self.max_memory_items = int(max_memory_items)
        self._indices = sorted(
            range(len(dataset)),
            key=lambda index: min(
                int(dataset.records[index % len(dataset.records)]["turn_id"]),
                self.max_memory_items,
            ),
        )

    def __iter__(self):
        return iter(self._indices)

    def __len__(self):
        return len(self._indices)


class DSTCSessionMemoryPool:
    def __init__(self):
        self.texts = {}
        self.embeddings = {}
        self._dialogue_index = {}
        self._embedding_dim = None

    @classmethod
    def from_datasets(cls, datasets: Iterable[DSTC10MODDataset]):
        pool = cls()
        for dataset in datasets:
            dialogue_ids = {record["dialogue_id"] for record in dataset.records}
            for dialogue_id in dialogue_ids:
                turns = dataset.dialogues[dialogue_id]
                for turn_id, turn in enumerate(turns):
                    pool.texts.setdefault(
                        (str(dialogue_id), int(turn_id)),
                        format_dstc_turn(turn),
                    )
        return pool

    def set_embeddings(self, embeddings):
        normalized = {
            (str(dialogue_id), int(turn_id)): embedding.detach().cpu().float()
            for (dialogue_id, turn_id), embedding in embeddings.items()
        }
        self.embeddings = normalized
        keys = list(normalized)
        if keys:
            tensor = torch.stack([normalized[key] for key in keys])
        else:
            tensor = torch.empty((0, 0), dtype=torch.float32)
        self.set_embedding_tensor(keys, tensor, keep_legacy_embeddings=True)

    def set_embedding_tensor(self, keys, embeddings, keep_legacy_embeddings=False):
        normalized_keys = [
            (str(dialogue_id), int(turn_id)) for dialogue_id, turn_id in keys
        ]
        embeddings = embeddings.detach().cpu().float()
        if embeddings.ndim != 2 or embeddings.shape[0] != len(normalized_keys):
            raise ValueError("Memory feature tensor has an invalid shape")
        if len(normalized_keys) != len(set(normalized_keys)):
            raise ValueError("Memory feature keys contain duplicates")
        if not keep_legacy_embeddings:
            self.embeddings = {}
        self._embedding_dim = int(embeddings.shape[1]) if embeddings.ndim == 2 else None
        rows_by_dialogue = defaultdict(list)
        for row, (dialogue_id, turn_id) in enumerate(normalized_keys):
            rows_by_dialogue[dialogue_id].append((turn_id, row))
        self._dialogue_index = {}
        for dialogue_id, turn_rows in rows_by_dialogue.items():
            turn_rows.sort(key=lambda item: item[0])
            turn_ids = torch.tensor(
                [turn_id for turn_id, _ in turn_rows], dtype=torch.long
            )
            rows = [row for _, row in turn_rows]
            if rows == list(range(rows[0], rows[0] + len(rows))):
                dialogue_embeddings = embeddings[rows[0]:rows[0] + len(rows)]
            else:
                dialogue_embeddings = embeddings.index_select(
                    0, torch.tensor(rows, dtype=torch.long)
                )
            self._dialogue_index[dialogue_id] = (turn_ids, dialogue_embeddings)

    def query(self, dialogue_ids, turn_ids, device, max_pool_size=256):
        if not self._dialogue_index:
            return None, None, None
        candidates = []
        for dialogue_id, current_turn_id in zip(dialogue_ids, turn_ids):
            current_turn_id = int(current_turn_id)
            indexed = self._dialogue_index.get(str(dialogue_id))
            if indexed is None:
                candidates.append(None)
                continue
            local_turn_ids, local_embeddings = indexed
            end = int(torch.searchsorted(
                local_turn_ids, torch.tensor(current_turn_id), right=False
            ))
            start = max(0, end - max_pool_size)
            candidates.append((
                local_turn_ids[start:end],
                local_embeddings[start:end],
            ))
        max_count = max(
            (len(items[0]) for items in candidates if items is not None),
            default=0,
        )
        if max_count == 0:
            return None, None, None
        pin_memory = device.type == "cuda" and torch.cuda.is_available()
        pool_embeddings = torch.zeros(
            len(candidates), max_count, self._embedding_dim,
            dtype=torch.float32, pin_memory=pin_memory,
        )
        pool_turn_ids = torch.zeros(
            len(candidates), max_count, dtype=torch.long, pin_memory=pin_memory,
        )
        pool_valid_mask = torch.zeros(
            len(candidates), max_count, dtype=torch.bool, pin_memory=pin_memory,
        )
        for row, items in enumerate(candidates):
            if items is None:
                continue
            candidate_turn_ids, candidate_embeddings = items
            count = len(candidate_turn_ids)
            pool_embeddings[row, :count].copy_(candidate_embeddings)
            pool_turn_ids[row, :count].copy_(candidate_turn_ids)
            pool_valid_mask[row, :count] = True
        return (
            pool_embeddings.to(device, non_blocking=pin_memory),
            pool_turn_ids.to(device, non_blocking=pin_memory),
            pool_valid_mask.to(device, non_blocking=pin_memory),
        )
