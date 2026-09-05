"""Adaptive Sparse Temporal Retrieval and Aggregation (ASTRA)."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class AdaptiveMultiScaleTemporalDecay(nn.Module):
    """Query-dependent mixture of learnable temporal decay scales (AMTD)."""

    def __init__(self, hidden_size=768, num_scales=4, init_time_scale=64.0):
        super().__init__()
        if num_scales <= 0:
            raise ValueError("num_scales must be positive")

        self.num_scales = int(num_scales)
        minimum_scale = 1.0
        maximum_scale = max(float(init_time_scale), 2.0)
        self.register_buffer(
            "time_scale_min", torch.tensor(minimum_scale, dtype=torch.float32)
        )
        self.register_buffer(
            "time_scale_max", torch.tensor(maximum_scale, dtype=torch.float32)
        )

        ratio = (float(init_time_scale) - minimum_scale) / (
            maximum_scale - minimum_scale
        )
        ratio = float(np.clip(ratio, 0.01, 0.99))
        self.time_scale_logit = nn.Parameter(
            torch.tensor(math.log(ratio / (1.0 - ratio)), dtype=torch.float32)
        )
        self.initial_time_scale = float(init_time_scale)
        default_rates = torch.tensor([0.5, 2.0, 5.0, 10.0])
        if self.num_scales != len(default_rates):
            default_rates = torch.linspace(0.5, 10.0, self.num_scales)
        self.decay_rates = nn.Parameter(default_rates)

        self.semantic_projection = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        self.time_projection = nn.Sequential(
            nn.Linear(1, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
        )
        fusion_size = hidden_size + hidden_size // 2
        self.scale_assignment = nn.Sequential(
            nn.Linear(fusion_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, self.num_scales),
            nn.Softmax(dim=-1),
        )

    @property
    def time_scale(self):
        ratio = torch.sigmoid(self.time_scale_logit)
        return self.time_scale_min + ratio * (
            self.time_scale_max - self.time_scale_min
        )

    def set_time_scale_bounds(self, init_time_scale, max_time_scale=None):
        minimum = float(self.time_scale_min.item())
        initial = max(float(init_time_scale), minimum)
        maximum = max(
            float(max_time_scale if max_time_scale is not None else initial),
            initial,
            minimum + 1.0,
        )
        ratio = (initial - minimum) / max(maximum - minimum, 1e-8)
        ratio = float(np.clip(ratio, 0.01, 0.99))
        logit = math.log(ratio / (1.0 - ratio))
        with torch.no_grad():
            self.time_scale_max.fill_(maximum)
            self.time_scale_logit.fill_(logit)
        self.initial_time_scale = initial
        return {
            "time_scale_min": minimum,
            "time_scale_max": maximum,
            "init_time_scale": initial,
            "time_scale_logit": logit,
        }

    def forward(self, interaction_features, time_deltas):
        time_scale = self.time_scale
        time_deltas = F.relu(time_deltas)
        range_gate = torch.sigmoid(
            (time_scale - time_deltas) / (0.1 * time_scale + 1e-8)
        )
        proximity = torch.exp(-time_deltas / (time_scale + 1e-8))
        semantic_features = self.semantic_projection(interaction_features)
        temporal_features = self.time_projection(proximity.unsqueeze(-1))
        scale_probabilities = self.scale_assignment(
            torch.cat([semantic_features, temporal_features], dim=-1)
        )

        normalized_age = time_deltas / (time_scale + 1e-8)
        positive_rates = F.softplus(self.decay_rates)
        per_scale = torch.exp(
            -normalized_age.unsqueeze(-1) * positive_rates.view(1, 1, -1)
        )
        decay = (
            range_gate.unsqueeze(-1) * per_scale * scale_probabilities
        ).sum(dim=-1)
        return torch.clamp(decay, min=1e-6, max=1.0), scale_probabilities


class DynamicSparseMemoryAggregation(nn.Module):
    """Query-thresholded sparse gating and rank-aware aggregation (DSMA)."""

    def __init__(self, gate_temperature=1.0, rank_temperature=1.0):
        super().__init__()
        if gate_temperature <= 0 or rank_temperature <= 0:
            raise ValueError("DSMA temperatures must be positive")
        self.gate_temperature = float(gate_temperature)
        self.rank_temperature = float(rank_temperature)

    def forward(self, scores, valid_mask, thresholds):
        if scores.ndim != 2 or valid_mask.shape != scores.shape:
            raise ValueError("scores and valid_mask must have shape [batch, memories]")
        if thresholds.shape != (scores.shape[0], 1):
            raise ValueError("thresholds must have shape [batch, 1]")
        if scores.shape[1] == 0:
            return scores, scores

        safe_scores = scores.masked_fill(~valid_mask, 0.0)
        soft_gates = torch.sigmoid(
            (safe_scores - thresholds) / self.gate_temperature
        )
        sparse_gates = torch.clamp(2.0 * soft_gates - 1.0, min=0.0)
        gates = soft_gates + (sparse_gates - soft_gates).detach()
        gates = gates.masked_fill(~valid_mask, 0.0)

        masked_scores = safe_scores.masked_fill(~valid_mask, float("-inf"))
        has_valid = valid_mask.any(dim=-1, keepdim=True)
        maximum = masked_scores.max(dim=-1, keepdim=True).values
        maximum = torch.where(has_valid, maximum, torch.zeros_like(maximum))
        rank_logits = (safe_scores - maximum) / self.rank_temperature
        rank_logits = rank_logits.masked_fill(~valid_mask, float("-inf"))
        contributions = gates * torch.exp(rank_logits)
        mass = contributions.sum(dim=-1, keepdim=True)
        weights = contributions / mass.clamp_min(1.0)
        return weights, gates


class ASTRA(nn.Module):
    """Full adaptive sparse temporal memory read used by MGTM."""

    @staticmethod
    def _inverse_softplus(value):
        value = float(value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("score weights must be finite and positive")
        return math.log(math.expm1(value)) if value < 20.0 else value

    def __init__(
        self,
        hidden_size=768,
        embed_dim=512,
        init_time_scale=64.0,
        gate_temperature=1.0,
        rank_temperature=1.0,
        astra_semantic_weight=100.0,
        astra_time_weight=0.35,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.embed_dim = int(embed_dim)
        self.astra_semantic_weight_raw = nn.Parameter(
            torch.tensor(
                self._inverse_softplus(astra_semantic_weight), dtype=torch.float32
            )
        )
        self.astra_time_weight_raw = nn.Parameter(
            torch.tensor(
                self._inverse_softplus(astra_time_weight), dtype=torch.float32
            )
        )
        self.amtd = AdaptiveMultiScaleTemporalDecay(
            hidden_size=hidden_size,
            num_scales=4,
            init_time_scale=init_time_scale,
        )
        self.dsma = DynamicSparseMemoryAggregation(
            gate_temperature=gate_temperature,
            rank_temperature=rank_temperature,
        )
        self.query_projection = nn.Linear(hidden_size, hidden_size)
        self.key_projection = nn.Linear(hidden_size, hidden_size)
        self.value_projection = nn.Linear(hidden_size, hidden_size)
        self.value_output = nn.Linear(hidden_size, embed_dim)
        threshold_hidden = max(hidden_size // 4, 32)
        self.threshold_base = nn.Parameter(torch.zeros(()))
        self.threshold_norm = nn.LayerNorm(hidden_size)
        self.threshold_network = nn.Sequential(
            nn.Linear(hidden_size, threshold_hidden),
            nn.GELU(),
            nn.Linear(threshold_hidden, 1),
        )
        nn.init.zeros_(self.threshold_network[-1].weight)
        nn.init.zeros_(self.threshold_network[-1].bias)

    @property
    def astra_semantic_weight(self):
        return F.softplus(self.astra_semantic_weight_raw)

    @property
    def astra_time_weight(self):
        return F.softplus(self.astra_time_weight_raw)

    @property
    def gamma_sem(self):
        return self.astra_semantic_weight

    @property
    def lambda_time(self):
        return self.astra_time_weight

    def set_time_scale_bounds(self, init_time_scale, max_time_scale=None):
        return self.amtd.set_time_scale_bounds(init_time_scale, max_time_scale)

    def _empty_result(self, batch_size, memory_count, reference):
        memory = reference.new_zeros(batch_size, self.embed_dim)
        diagnostics = {
            "valid_count": torch.zeros(
                batch_size, device=reference.device, dtype=torch.long
            ),
            "selected_count": torch.zeros(
                batch_size, device=reference.device, dtype=torch.long
            ),
            "aggregation_weights": reference.new_zeros(batch_size, memory_count),
            "sparse_gates": reference.new_zeros(batch_size, memory_count),
            "time_scale": self.amtd.time_scale,
            "decay_rates": F.softplus(self.amtd.decay_rates),
        }
        return memory, None, diagnostics

    def forward(
        self,
        query_embedding,
        memory_embeddings,
        current_positions,
        memory_positions,
        valid_mask,
        return_diagnostics=False,
    ):
        batch_size, memory_count, hidden_size = memory_embeddings.shape
        if hidden_size != self.hidden_size:
            raise ValueError("memory hidden size does not match ASTRA")

        causal_mask = memory_positions < current_positions.unsqueeze(1)
        valid_mask = valid_mask.bool() & causal_mask
        if memory_count == 0 or not valid_mask.any():
            memory, probabilities, diagnostics = self._empty_result(
                batch_size, memory_count, query_embedding
            )
            if return_diagnostics:
                return memory, probabilities, diagnostics
            return memory, probabilities

        query = self.query_projection(query_embedding)
        safe_memories = memory_embeddings.masked_fill(
            ~valid_mask.unsqueeze(-1), 0.0
        )
        keys = self.key_projection(safe_memories)
        values = self.value_projection(safe_memories)
        time_deltas = F.relu(
            current_positions.unsqueeze(1).float() - memory_positions.float()
        )
        expanded_query = query.unsqueeze(1).expand(-1, memory_count, -1)
        interactions = torch.cat(
            [expanded_query * keys, torch.abs(expanded_query - keys)], dim=-1
        )
        semantic_scores = (
            query.unsqueeze(1) * keys
        ).sum(dim=-1) / math.sqrt(self.hidden_size)
        decay, scale_probabilities = self.amtd(interactions, time_deltas)
        scores = (
            self.gamma_sem * semantic_scores
            + self.lambda_time * torch.log(decay + 1e-6)
        )
        thresholds = self.threshold_base + self.threshold_network(
            self.threshold_norm(query)
        )
        weights, gates = self.dsma(scores, valid_mask, thresholds)
        candidate_values = F.normalize(
            self.value_output(values), p=2, dim=-1, eps=1e-6
        )
        memory = (weights.unsqueeze(-1) * candidate_values).sum(dim=1)

        if not return_diagnostics:
            return memory, scale_probabilities
        diagnostics = {
            "valid_count": valid_mask.sum(dim=-1),
            "selected_count": (gates > 0).sum(dim=-1),
            "aggregation_weights": weights,
            "sparse_gates": gates,
            "time_scale": self.amtd.time_scale,
            "decay_rates": F.softplus(self.amtd.decay_rates),
        }
        return memory, scale_probabilities, diagnostics
