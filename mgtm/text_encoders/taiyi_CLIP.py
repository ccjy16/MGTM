"""Trainable Taiyi text tower for MGTM training."""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer



def tokenize_texts(texts, tokenizer, context_length=512):
    return tokenizer(
        [str(text) for text in texts],
        padding="max_length",
        truncation=True,
        max_length=int(context_length),
        return_tensors="pt",
    )


class TrainableTaiyiTextTower(nn.Module):
    def __init__(self, encoder, tokenizer, context_length=512):
        super().__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.context_length = int(context_length)
        self._token_cache = {}

    @classmethod
    def from_pretrained(cls, model_path, context_length=512):
        model_path = Path(model_path).resolve()
        if not model_path.is_dir():
            raise FileNotFoundError(f"Taiyi text model directory not found: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            clean_up_tokenization_spaces=False,
        )
        encoder = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
        )
        return cls(encoder, tokenizer, context_length=context_length)

    def set_grad_checkpointing(self, enable=True):
        if enable:
            self.encoder.gradient_checkpointing_enable()
        else:
            self.encoder.gradient_checkpointing_disable()

    def tokenize(self, texts):
        keys = [str(text) for text in texts]
        missing = list(dict.fromkeys(
            key for key in keys if key not in self._token_cache
        ))
        if missing:
            encoded = tokenize_texts(
                missing,
                self.tokenizer,
                context_length=self.context_length,
            )
            for index, key in enumerate(missing):
                self._token_cache[key] = {
                    "input_ids": encoded["input_ids"][index].clone(),
                    "attention_mask": encoded["attention_mask"][index].clone(),
                }
        return {
            "input_ids": torch.stack([
                self._token_cache[key]["input_ids"] for key in keys
            ]),
            "attention_mask": torch.stack([
                self._token_cache[key]["attention_mask"] for key in keys
            ]),
        }

    def encode_tokens(self, input_ids, attention_mask):
        effective_length = int(attention_mask.sum(dim=1).max().item())
        input_ids = input_ids[:, :effective_length]
        attention_mask = attention_mask[:, :effective_length]
        device = next(self.encoder.parameters()).device
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        features = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).logits
        if features.ndim != 2 or features.shape[1] != 512:
            raise RuntimeError(
                f"Taiyi text output must be [batch, 512], got {tuple(features.shape)}"
            )
        return F.normalize(features.float(), dim=-1)

    def forward(self, texts):
        if isinstance(texts, dict):
            tokens = texts
        else:
            tokens = self.tokenize(texts)
        return self.encode_tokens(tokens["input_ids"], tokens["attention_mask"])
