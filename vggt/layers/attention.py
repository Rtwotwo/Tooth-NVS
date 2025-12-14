# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

XFORMERS_AVAILABLE = False


@torch.compile
def infer_layer_norm(x, w, b, dims, eps):
    x -= x.mean(dims, True)
    x /= torch.sqrt((x**2).mean(dims, True) + eps)
    x *= w
    x += b
 
def infer_norm(norm_layer, x):
    # torch.cuda.empty_cache()
    if isinstance(norm_layer, nn.LayerNorm):
        assert norm_layer.elementwise_affine == True
        w = norm_layer.weight
        b = norm_layer.bias
        dims = [i for i in range(len(x.shape)-len(w.shape), len(x.shape))]
        infer_layer_norm(x, w, b, dims, norm_layer.eps)
        # torch.cuda.empty_cache()
        return x
    return norm_layer(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

        self.training = False
        self.q_norm.training = False
        self.k_norm.training = False

    def forward(self, x: Tensor, pos=None) -> Tensor:
        B, N, C = x.shape

        # TODO: avoid doubled memory by splitting self.qkv into three separate MLP
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        del qkv

        # TODO: nn.LayerNorm seems to be doing weird stuff and increases VRAM usage even after return
        # maybe hard code running mean/std, and possibly bake them into self.qkv weights?
        # torch.cuda.empty_cache()
        # q = self.q_norm(q).to(q.dtype); torch.cuda.empty_cache()
        # k = self.k_norm(k).to(k.dtype); torch.cuda.empty_cache()
        infer_norm(self.q_norm, q)
        infer_norm(self.k_norm, k)

        if self.rope is not None:
            q = self.rope(q, pos).to(q.dtype)
            k = self.rope(k, pos).to(q.dtype)
            # torch.cuda.empty_cache()

        assert self.fused_attn
        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            # attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        # x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
