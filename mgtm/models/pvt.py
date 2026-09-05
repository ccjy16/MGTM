"""Four-stage PVTv2 visual encoder used by MGTM."""

from __future__ import annotations

import math
import re
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class DepthwiseConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, height, width):
        batch, _, channels = x.shape
        x = x.transpose(1, 2).view(batch, channels, height, width)
        return self.dwconv(x).flatten(2).transpose(1, 2)


class PVTMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.dwconv = DepthwiseConv(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x, height, width):
        x = self.fc1(x)
        x = self.dwconv(x, height, width)
        return self.fc2(self.act(x))


class PVTPatchEmbed(nn.Module):
    def __init__(self, image_size, kernel_size, stride, in_channels, embed_dim):
        super().__init__()
        self.padding = (kernel_size - stride + 1) // 2
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.padding,
        )
        self.norm = nn.LayerNorm(embed_dim)
        height = (image_size + 2 * self.padding - kernel_size) // stride + 1
        self.num_patches = height * height

    def forward(self, x):
        x = self.proj(x)
        _, _, height, width = x.shape
        return self.norm(x.flatten(2).transpose(1, 2)), height, width


class SpatialReductionAttention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, height, width):
        batch, tokens, channels = x.shape
        q = self.q(x).reshape(
            batch, tokens, self.num_heads, channels // self.num_heads
        ).permute(0, 2, 1, 3)
        source = x
        if self.sr_ratio > 1:
            source = x.permute(0, 2, 1).reshape(batch, channels, height, width)
            source = self.sr(source).reshape(batch, channels, -1).permute(0, 2, 1)
            source = self.norm(source)
        kv = self.kv(source).reshape(
            batch, -1, 2, self.num_heads, channels // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        keys, values = kv[0], kv[1]
        attention = ((q @ keys.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        output = (attention @ values).transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj(output), attention


class PVTBlock(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialReductionAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = PVTMLP(dim, int(dim * mlp_ratio))

    def forward(self, x, height, width):
        attended, weights = self.attn(self.norm1(x), height, width)
        x = x + attended
        return x + self.mlp(self.norm2(x), height, width), weights


class PVTEncoderStage(nn.Module):
    def __init__(self, dim, depth, num_heads, sr_ratio, mlp_ratio):
        super().__init__()
        self.blocks = nn.ModuleList([
            PVTBlock(dim, num_heads, sr_ratio, mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, height, width):
        attention = None
        for block in self.blocks:
            x, attention = block(x, height, width)
        x = self.norm(x)
        token_importance = attention.mean(dim=1).mean(dim=1)
        return x, token_importance


class PVTMultiGranularityVisualEncoder(nn.Module):
    """Combines global and attention-selected local features at four resolutions."""

    def __init__(
        self,
        image_size=224,
        embed_dims=(64, 128, 320, 512),
        num_heads=(1, 2, 5, 8),
        depths=(3, 4, 6, 3),
        sr_ratios=(8, 4, 2, 1),
        mlp_ratios=(8, 8, 4, 4),
        output_dim=512,
        local_k=8,
    ):
        super().__init__()
        self.local_k = int(local_k)
        stage_sizes = (image_size, image_size // 4, image_size // 8, image_size // 16)
        kernels = (7, 3, 3, 3)
        strides = (4, 2, 2, 2)
        input_dims = (3, *embed_dims[:-1])
        self.patch_embeds = nn.ModuleList([
            PVTPatchEmbed(stage_sizes[i], kernels[i], strides[i], input_dims[i], embed_dims[i])
            for i in range(4)
        ])
        self.pos_embeds = nn.ParameterList([
            nn.Parameter(torch.zeros(1, layer.num_patches, embed_dims[i]))
            for i, layer in enumerate(self.patch_embeds)
        ])
        self.pos_drops = nn.ModuleList([nn.Dropout(0.0) for _ in range(4)])
        self.stages = nn.ModuleList([
            PVTEncoderStage(
                embed_dims[i], depths[i], num_heads[i], sr_ratios[i], mlp_ratios[i]
            )
            for i in range(4)
        ])
        self.global_proj = nn.ModuleList([
            nn.Linear(dim, output_dim // 2) for dim in embed_dims
        ])
        self.local_proj = nn.ModuleList([
            nn.Linear(dim * self.local_k, output_dim // 2) for dim in embed_dims
        ])
        self.stage_weights = nn.Parameter(torch.ones(4) / 4)
        self.final_proj = nn.Linear(output_dim, output_dim)
        self.apply(self._initialize)
        for embedding in self.pos_embeds:
            nn.init.trunc_normal_(embedding, std=0.02)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, std=math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _local_features(self, x, importance):
        batch, tokens, channels = x.shape
        count = min(self.local_k, tokens)
        indices = torch.topk(importance, count, dim=1).indices
        selected = torch.gather(
            x, 1, indices.unsqueeze(-1).expand(-1, -1, channels)
        )
        if count < self.local_k:
            selected = F.pad(selected, (0, 0, 0, self.local_k - count))
        return selected.reshape(batch, -1)

    def forward(self, images, mask_ratio=0.0):
        del mask_ratio
        batch = images.shape[0]
        x = images
        stage_features = []
        height = width = None
        for index in range(4):
            if index > 0:
                x = x.permute(0, 2, 1).reshape(batch, -1, height, width)
            x, height, width = self.patch_embeds[index](x)
            x = self.pos_drops[index](x + self.pos_embeds[index][:, :x.size(1)])
            x, importance = self.stages[index](x, height, width)
            global_feature = self.global_proj[index](x.mean(dim=1))
            local_feature = self.local_proj[index](self._local_features(x, importance))
            stage_features.append(torch.cat([global_feature, local_feature], dim=-1))
        stacked = torch.stack(stage_features, dim=1)
        weights = F.softmax(self.stage_weights, dim=0).view(1, 4, 1)
        return self.final_proj((stacked * weights).sum(dim=1))


def remap_pvtv2_state_dict(checkpoint):
    state_dict = checkpoint
    for key in ("model", "state_dict"):
        if isinstance(state_dict, dict) and key in state_dict:
            state_dict = state_dict[key]
    remapped = {}
    for key, value in state_dict.items():
        key = key.removeprefix("module.").removeprefix("backbone.")
        key = re.sub(
            r"^patch_embed(\d+)(.*)",
            lambda match: f"patch_embeds.{int(match.group(1)) - 1}{match.group(2)}",
            key,
        )
        key = re.sub(
            r"^block(\d+)\.(\d+)(.*)",
            lambda match: (
                f"stages.{int(match.group(1)) - 1}.blocks."
                f"{match.group(2)}{match.group(3)}"
            ),
            key,
        )
        key = re.sub(
            r"^norm(\d+)\.(\w+)$",
            lambda match: f"stages.{int(match.group(1)) - 1}.norm.{match.group(2)}",
            key,
        )
        key = re.sub(
            r"^pos_embed(\d+)(.*)",
            lambda match: f"pos_embeds.{int(match.group(1)) - 1}{match.group(2)}",
            key,
        )
        remapped[key] = value
    return remapped


def load_pvtv2_weights(model, checkpoint_path):
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"PVTv2 weights not found: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    remapped = remap_pvtv2_state_dict(checkpoint)
    target = model.state_dict()
    loadable = {
        key: value for key, value in remapped.items()
        if key in target and target[key].shape == value.shape
    }
    ignored = sorted(set(remapped) - set(loadable))
    incompatible = model.load_state_dict(loadable, strict=False)
    return {
        "loaded": sorted(loadable),
        "ignored": ignored,
        "missing": list(incompatible.missing_keys),
        "unexpected": list(incompatible.unexpected_keys),
    }


__all__ = [
    "PVTMultiGranularityVisualEncoder",
    "load_pvtv2_weights",
    "remap_pvtv2_state_dict",
]
