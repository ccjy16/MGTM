"""MGTM model with one Taiyi, PVT, and ASTRA execution path."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from mgtm.models.astra import ASTRA
from mgtm.models.pvt import PVTMultiGranularityVisualEncoder
from mgtm.text_encoders.taiyi_CLIP import TrainableTaiyiTextTower


class Classifier(nn.Module):
    """Auxiliary intent or emotion classifier over query-memory features."""

    def __init__(self, input_size, num_classes):
        super().__init__()
        self.projection = nn.Linear(input_size * 2, num_classes)

    def forward(self, features, has_memory=None):
        del has_memory
        logits = self.projection(features)
        return logits, logits.argmax(dim=1)


class MGTMModel(nn.Module):
    """Multi-Granularity Temporal Memory Model."""

    def __init__(
        self,
        embed_dim=512,
        image_resolution=224,
        taiyi_text_model_path=None,
        context_length=512,
        init_time_scale=64.0,
        astra_semantic_weight=100.0,
        astra_time_weight=0.35,
        local_k=8,
        astra_hidden_size=512,
        text_tower=None,
        visual_encoder=None,
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.taiyi_text_tower = text_tower
        if self.taiyi_text_tower is None:
            if taiyi_text_model_path is None:
                raise ValueError("taiyi_text_model_path is required")
            self.taiyi_text_tower = TrainableTaiyiTextTower.from_pretrained(
                taiyi_text_model_path,
                context_length=context_length,
            )
        self.visual = visual_encoder or PVTMultiGranularityVisualEncoder(
            image_size=image_resolution,
            output_dim=self.embed_dim,
            local_k=local_k,
        )
        self.astra = ASTRA(
            hidden_size=astra_hidden_size,
            embed_dim=self.embed_dim,
            init_time_scale=init_time_scale,
            astra_semantic_weight=astra_semantic_weight,
            astra_time_weight=astra_time_weight,
        )
        self.intent_proj_with_summary = nn.Linear(self.embed_dim * 3, self.embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    @property
    def dtype(self):
        parameter = next(self.visual.parameters(), None)
        return parameter.dtype if parameter is not None else torch.float32

    def set_grad_checkpointing(self, enable=True):
        self.taiyi_text_tower.set_grad_checkpointing(enable)

    def set_time_scale_bounds(self, init_time_scale, max_time_scale=None):
        return self.astra.set_time_scale_bounds(init_time_scale, max_time_scale)

    def encode_image(self, images, mask_ratio=0.0):
        return self.visual(images.to(dtype=self.dtype), mask_ratio=mask_ratio)

    def encode_text(self, texts, sent_text=None, return_pooled=False):
        del sent_text
        features = self.taiyi_text_tower(texts)
        if features.ndim != 2 or features.shape[-1] != self.embed_dim:
            raise RuntimeError(
                f"Taiyi text output must be [batch, {self.embed_dim}], "
                f"got {tuple(features.shape)}"
            )
        features = F.normalize(features.float(), dim=-1)
        return (features, features) if return_pooled else features

    def fuse_retrieval_query(self, query_features, auxiliary_features):
        fused = self.intent_proj_with_summary(
            torch.cat([query_features, auxiliary_features], dim=-1)
        )
        return F.normalize(fused, dim=-1)

    def _read_memory(
        self,
        query_embedding,
        current_positions,
        memory_embeddings,
        memory_positions,
        valid_mask,
        return_diagnostics,
    ):
        batch = query_embedding.shape[0]
        if memory_embeddings is None:
            memory = query_embedding.new_zeros(batch, self.embed_dim)
            return (memory, None) if return_diagnostics else memory
        if current_positions is None or memory_positions is None or valid_mask is None:
            raise ValueError("memory reads require positions and a valid mask")
        result = self.astra(
            query_embedding,
            memory_embeddings,
            current_positions,
            memory_positions,
            valid_mask,
            return_diagnostics=return_diagnostics,
        )
        if return_diagnostics:
            memory, _, diagnostics = result
            return memory, diagnostics
        memory, _ = result
        return memory

    def forward(
        self,
        Flag,
        image,
        text,
        history=None,
        intent_features=None,
        mask_ratio=0.0,
        precomputed_text_features=None,
        current_session_id=None,
        history_session_ids=None,
        pool_embeddings=None,
        pool_session_ids=None,
        pool_valid_mask=None,
        return_memory_diagnostics=False,
    ):
        del history, history_session_ids
        if Flag == 1:
            if precomputed_text_features is None:
                query_features, query_embedding = self.encode_text(
                    text, return_pooled=True
                )
            else:
                query_features = F.normalize(precomputed_text_features.float(), dim=-1)
                query_embedding = query_features
            memory_result = self._read_memory(
                query_embedding,
                current_session_id,
                pool_embeddings,
                pool_session_ids,
                pool_valid_mask,
                return_memory_diagnostics,
            )
            if return_memory_diagnostics:
                memory_features, diagnostics = memory_result
            else:
                memory_features = memory_result
            auxiliary_features = torch.cat([query_features, memory_features], dim=-1)
            if return_memory_diagnostics:
                return auxiliary_features, 1, query_features, diagnostics
            return auxiliary_features, 1, query_features

        if Flag == 0:
            image_features = F.normalize(
                self.encode_image(image, mask_ratio=mask_ratio), dim=-1
            )
            query_features = (
                self.encode_text(text)
                if precomputed_text_features is None
                else F.normalize(precomputed_text_features.float(), dim=-1)
            )
            if intent_features is None:
                raise ValueError("intent_features are required for retrieval")
            retrieval_query = self.fuse_retrieval_query(
                query_features, intent_features
            )
            return image_features, retrieval_query, self.logit_scale.exp()

        raise ValueError("Flag must be 0 for retrieval or 1 for memory encoding")

    def get_similarity(self, images, texts):
        image_features = F.normalize(self.encode_image(images), dim=-1)
        text_features = F.normalize(self.encode_text(texts), dim=-1)
        logits = self.logit_scale.exp() * image_features @ text_features.T
        return logits, logits.T


__all__ = ["Classifier", "MGTMModel"]
