import torch
import numpy as np
import time
import torch.nn as nn
from torch.nn import functional as F
import math
from models.ABN import MultiBatchNorm
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import torch.distributions as dist
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import os
import re
from transformers import CLIPTokenizer
from transformers import CLIPModel
import open_clip
import pdb
from collections import OrderedDict
import json

_tokenizer = _Tokenizer()

"""
LoRA: Low-Rank Adaptation utilities
"""


class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.0,
                 dual_lora: bool = False):
        super().__init__()

        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_dim = base_linear.in_features
        out_dim = base_linear.out_features
        self.rank = rank
        self.scaling = float(alpha) / float(rank)
        self.dual_lora = dual_lora  # 是否支持双LoRA（DeLoRA需要）

        # 双LoRA支持：Clean LoRA和Noisy LoRA
        if self.dual_lora:
            self.A_clean = nn.Parameter(torch.zeros(rank, in_dim))
            self.B_clean = nn.Parameter(torch.zeros(out_dim, rank))
            # 初始化Clean/Noisy LoRA权重
            nn.init.kaiming_uniform_(self.A_clean, a=math.sqrt(5))
            nn.init.zeros_(self.B_clean)
        else:
            print("请使用lora")

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def __getattr__(self, name):
        """Delegate attribute access to the base linear layer to support MultiheadAttention"""
        # 避免无限循环：只委托给base中确实存在的属性
        try:
            return super().__getattr__(name)
        except AttributeError:
            # 如果父类没有该属性，再尝试从base中获取
            if hasattr(self, 'base') and hasattr(self.base, name):
                return getattr(self.base, name)
            raise AttributeError(f"'{type(self).__name__}' object and its base module have no attribute '{name}'")

    def forward(self, x: torch.Tensor, use_clean_lora: bool = False, use_noisy_lora: bool = False,
                not_use_lora: bool = False) -> torch.Tensor:

        y = self.base(x)
        # 检查是否设置了属性（用于 get_clean_noisy_logits）
        if hasattr(self, '_use_clean_lora') and self._use_clean_lora is not None:
            use_clean_lora = self._use_clean_lora
        if hasattr(self, '_not_use_lora') and self._not_use_lora is not None:
            not_use_lora = self._not_use_lora

        # 决定使用哪一组LoRA权重
        if self.dual_lora:
            # 双LoRA模式
            if use_clean_lora:
                # 同时使用clean和noisy LoRA（默认行为）
                # 仅使用clean LoRA
                lora_x_clean = F.linear(self.dropout(x), self.A_clean)
                lora_out = F.linear(lora_x_clean, self.B_clean)
            elif not_use_lora:
                lora_out = torch.zeros_like(y)
            else:
                # 默认行为：双LoRA
                lora_x_clean = F.linear(self.dropout(x), self.A_clean)
                lora_out_clean = F.linear(lora_x_clean, self.B_clean)
                lora_out = lora_out_clean
        else:
            print("请使用delora")

        return y + self.scaling * lora_out


def _get_parent_module(root: nn.Module, dotted_name: str) -> nn.Module:
    parent = root
    if dotted_name == "":
        return parent
    for part in dotted_name.split("."):
        parent = getattr(parent, part)
    return parent


def inject_lora_to_visual(visual: nn.Module,
                          target_keywords=None,
                          rank: int = 4,
                          alpha: int = 16,
                          dropout: float = 0.0,
                          verbose: bool = True,
                          dual_lora: bool = False,
                          last_n_layers: int = None):
    """
    Replace matching nn.Linear modules under 'visual' with LoRALinear.
    Default targets cover both open_clip/timm ViT blocks and PyTorch ViT MultiheadAttention.

    Args:
        last_n_layers: If specified, only inject LoRA into the last N layers of the transformer.
                      Only applies to layers named like 'transformer.resblocks.{layer_idx}.*'
    """
    if target_keywords is None:
        target_keywords = ['attn.out_proj']
    replaced = []

    # Determine total number of layers if last_n_layers is specified
    total_layers = None
    if last_n_layers is not None:
        if hasattr(visual, 'transformer') and hasattr(visual.transformer, 'resblocks'):
            total_layers = len(visual.transformer.resblocks)
    for name, module in list(visual.named_modules()):
        if isinstance(module, nn.Linear) and any(k in name for k in target_keywords):
            # Check if we need to filter by layers
            skip_layer = False
            if last_n_layers is not None and total_layers is not None:
                # Parse layer index from module name
                parts = name.split(".")
                layer_idx = None
                for i, part in enumerate(parts):
                    if part == "resblocks" and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                            break
                        except ValueError:
                            pass

                if layer_idx is not None:
                    # Only keep layers in the last N layers
                    if layer_idx < (total_layers - last_n_layers):
                        skip_layer = True

            if not skip_layer:

                parent_path = name.rsplit(".", 1)[0] if "." in name else ""
                child_name = name.split(".")[-1]
                parent = _get_parent_module(visual, parent_path)
                setattr(parent, child_name,
                        LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout, dual_lora=dual_lora))
                replaced.append(name)


    if verbose:
        print(f"[LoRA] Injected into {len(replaced)} Linear layer(s).")
        if dual_lora:
            print(f"[LoRA] Dual LoRA enabled (Clean/Noisy weights)")
        for n in replaced:
            print("  -", n)
    return replaced


def freeze_backbone_except_lora(visual: nn.Module):
    """
    Freeze all parameters under 'visual' except LoRA adapters (A/B).
    """
    for name, param in visual.named_parameters():
        requires = (".A" in name) or (".B" in name) or (".A_clean" in name) or (".B_clean" in name) or (
                    ".A_noisy" in name) or (".B_noisy" in name)
        param.requires_grad_(requires)


def merge_all_lora(visual: nn.Module):
    for m in visual.modules():
        if isinstance(m, LoRALinear):
            m.merge()


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def interpolate_pos_encoding(self, x, w, h):  # x.shape = LBC
        npatch = x.shape[0] - 1
        positional_embedding_ = self.positional_embedding.unsqueeze(0)
        N = positional_embedding_.shape[1] - 1
        if npatch == N and w == h:
            return positional_embedding_[0]
        class_pos_embed = positional_embedding_[:, 0]
        patch_pos_embed = positional_embedding_[:, 1:]
        dim = x.shape[-1]
        w0 = w
        h0 = h
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(1), patch_pos_embed), dim=1)[0]

    def forward(self, x, if_pos=True):
        b, c, h, w = x.shape
        x_local = x.reshape(b, c, h, w).permute(0, 2, 3, 1)

        x_local = F.linear(x_local, self.v_proj.weight, self.v_proj.bias)
        x_local = F.linear(x_local, self.c_proj.weight, self.c_proj.bias)

        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0), x_local


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.attnpool(x)
        x_global, x_local = self.attnpool(x)
        return x_global, x_local


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def attention_weight(self, x: torch.Tensor):  # ADDED
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)[1]

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        y = self.ln_1(x)
        y = y.permute(1, 0, 2)
        y = F.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias)
        N, L, C = y.shape
        y = y.view(N, L, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * N, L, C // 3)
        y = F.linear(y, self.attn.out_proj.weight, self.attn.out_proj.bias)
        q, k, v = y.tensor_split(3, dim=0)
        v = v.permute(1, 0, 2)
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v += x
        v = v + self.mlp(self.ln_2(v))

        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x, q, k, v


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        # self.resblock = ResidualAttentionBlock(width, heads, attn_mask)

    def forward(self, x: torch.Tensor):
        # return self.resblocks(x)
        for i in range(self.layers):
            x, q, k, v = self.resblocks[i](x)
        return x, q, k, v


# class ytVisualWithLocal(nn.Module):
#     def __init__(self, visual_encoder):
#         super().__init__()
#         self.input_resolution = visual_encoder.input_resolution
#         self.output_dim = visual_encoder.output_dim
#         input_resolution = visual_encoder.input_resolution
#         output_dim = visual_encoder.output_dim
#         width = visual_encoder.transformer.width
#         patch_size = 16
#         layers = visual_encoder.transformer.layers
#         heads = visual_encoder.transformer.resblocks[0].attn.num_heads

#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

#         scale = width ** -0.5
#         self.class_embedding = nn.Parameter(scale * torch.randn(width))
#         self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
#         self.ln_pre = LayerNorm(width)

#         self.transformer = Transformer(width, layers, heads)

#         self.ln_post = LayerNorm(width)
#         self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

#     def forward(self, x: torch.Tensor):
#         x = self.conv1(x)  # shape = [*, width, grid, grid]
#         hw_shape = (x.shape[2], x.shape[3])
#         x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
#         x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

#         x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
#         x = x + self.positional_embedding.to(x.dtype)
#         x = self.ln_pre(x)

#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x, q, k, v = self.transformer(x)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         q = q.permute(1, 0, 2)
#         k = k.permute(1, 0, 2)
#         v = v.permute(1, 0, 2)

#         v = self.ln_post(v)

#         q = q[:, 1:]
#         k = k[:, 1:]
#         v = v[:, 1:]

#         out = x[:, 1:]
#         B, _, C = out.shape
#         v = v.reshape(B, hw_shape[0], hw_shape[1], C).contiguous()

#         x = self.ln_post(x[:, 0, :])

#         if self.proj is not None:
#             x = x @ self.proj
#             feat = v @ self.proj
#         return x, feat

class ytVisualWithLocal(nn.Module):
    def __init__(self, visual_encoder):
        super().__init__()
        self.visual = visual_encoder  # 直接接收 open_clip 的 visual 模块

    def forward(self, x):
        """
        返回 CLS embedding 和所有 patch embeddings
        Args:
            model: open_clip VisionTransformer
            images: (B, 3, H, W)
        Returns:
            cls_embeds: (B, D)
            patch_embeds: (B, P, D)  # 不含 CLS
        """

        visual = self.visual
        x = visual.conv1(x)  # shape = [*, width, grid, grid]
        hw_shape = (x.shape[2], x.shape[3])
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        x = torch.cat([visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                        device=x.device), x],
                      dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)

        # transformer
        # x = x.permute(1, 0, 2)  # NLD -> LND
        x = visual.transformer(x)

        # x = x.permute(1, 0, 2)  # LND -> NLD
        x = visual.ln_post(x)

        cls_embeds = x[:, 0, :]  # (B, D)
        patch_embeds = x[:, 1:, :]  # (B, P, D)

        if visual.proj is not None:
            cls_embeds = cls_embeds @ visual.proj
            patch_embeds = patch_embeds @ visual.proj
        return cls_embeds, patch_embeds


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg['N_CTX']
        ctx_init = cfg['CTX_INIT']
        dtype = next(clip_model.parameters()).dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("\"{}\"", "")
            ctx_init = ctx_init.replace(".", "")
            ctx_init = ctx_init.replace("_", " ")
            words = re.findall(r'\b\w+\b', ctx_init)
            n_ctx = len(words)
            print('n_ctx', n_ctx)
            prompt = clip.tokenize(ctx_init).cuda()
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg['CSC']:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = next(clip_model.parameters()).dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class OriginalCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)

        self.logit_scale = clip_model.logit_scale
        self.dtype = next(clip_model.parameters()).dtype
        self.classnames = classnames
        self.clip_model = clip_model
        self.cfg = cfg

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        # template = self.cfg['CTX_INIT']
        # text_inputs = torch.cat([clip.tokenize(template.format(classname)) for classname in self.classnames]).cuda()
        # text_features = self.clip_model.encode_text(text_inputs)
        # image_features = self.clip_model.encode_image(image)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = (logit_scale * image_features @ text_features.t())
        # logits = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return logits, text_features


class NegaPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg['N_CTX']
        ctx_init = cfg['CTX_INIT']
        if cfg['CSC'] > 0:
            ctx_init = None
        n_nega_ctx = cfg['NEGA_CTX']
        self.n_nega_ctx = n_nega_ctx
        self.csc = cfg['CSC']
        self.cfg = cfg
        dtype = next(clip_model.parameters()).dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("\"{}\"", "")
            ctx_init = ctx_init.replace(".", "")
            ctx_init = ctx_init.replace("_", " ")
            words = re.findall(r'\b\w+\b', ctx_init)
            n_ctx = len(words)
            prompt = clip.tokenize(ctx_init)
            prompt = prompt.cuda()
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            # print("prompt.shape", prompt.shape)
            # print(prompt)
            # print("embedding.shape", embedding.shape)
            ctx_vectors = ctx_vectors.view(1, ctx_vectors.shape[0], ctx_vectors.shape[1])  # class_posi, ctx, vector
            ctx_vectors = ctx_vectors.repeat(1 + n_nega_ctx, 1, 1)
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg['CSC'] > 0:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, 1 + n_nega_ctx, n_ctx, ctx_dim, dtype=dtype).cuda()
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(1 + n_nega_ctx, n_ctx, ctx_dim, dtype=dtype).cuda()
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        if ctx_vectors.dim() == 3:
            ctx_positive = ctx_vectors[0:1, :, :]
            ctx_negative = ctx_vectors[1:, :, :]
        else:
            ctx_positive = ctx_vectors[:, 0:1, :, :]
            ctx_negative = ctx_vectors[:, 1:, :, :]
        self.ctx_positive = nn.Parameter(ctx_positive)  # to be optimized

        if ctx_negative.shape[0] == 0:
            ctx_negative = torch.empty(0, dtype=dtype).cuda()
        self.ctx_negative = nn.Parameter(ctx_negative)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        positive_prompts = [prompt_prefix + " " + name for name in classnames]
        negative_prompts = [prompt_prefix + " " + name for name in classnames]

        positive_tokenized_prompts = torch.cat([clip.tokenize(p) for p in positive_prompts]).cuda()
        # print(positive_tokenized_prompts[0])
        negative_tokenized_prompts = torch.cat([clip.tokenize(p) for p in negative_prompts]).cuda()
        # tokenized_prompts:
        # tensor([ <start>    a     photo   of   a  positive [classname] . <end>
        # [49406,   320,  1125,   539,   320,  4844,  1929,   269, 49407, 0 ...,0],
        # [49406,   320,  1125,   539,   320,  4844,  2368,   269, 49407, 0 ...,0],
        # [49406,   320,  1125,   539,   320,  4844,  4558,   269, 49407, 0 ...,0],
        # [49406,   320,  1125,   539,   320,  4844,  6531,   269, 49407, 0 ...,0]])
        with torch.no_grad():
            positive_embedding = clip_model.token_embedding(positive_tokenized_prompts).type(dtype)
            negative_embedding = clip_model.token_embedding(negative_tokenized_prompts).type(dtype)

        positive_embedding = positive_embedding.view(positive_embedding.shape[0], 1, positive_embedding.shape[1],
                                                     positive_embedding.shape[2])
        negative_embedding = negative_embedding.view(negative_embedding.shape[0], 1, negative_embedding.shape[1],
                                                     negative_embedding.shape[2])
        negative_embedding = negative_embedding.repeat(1, n_nega_ctx, 1, 1)
        embedding = torch.cat([positive_embedding, negative_embedding], dim=1)
        positive_tokenized_prompts = positive_tokenized_prompts.view(positive_tokenized_prompts.shape[0], 1,
                                                                     positive_tokenized_prompts.shape[1])
        negative_tokenized_prompts = negative_tokenized_prompts.view(negative_tokenized_prompts.shape[0], 1,
                                                                     negative_tokenized_prompts.shape[1])
        negative_tokenized_prompts = negative_tokenized_prompts.repeat(1, n_nega_ctx, 1)
        tokenized_prompts = torch.cat([positive_tokenized_prompts, negative_tokenized_prompts], dim=1)
        tokenized_prompts = tokenized_prompts.view(tokenized_prompts.shape[0] * tokenized_prompts.shape[1], -1)
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, :, 1 + n_ctx:, :])  # positive prompt CLS, EOS
        if cfg['stage'] >= 2:
            self.register_buffer("positive_token_prefix", embedding[:, :1, :1, :])  # SOS
            self.register_buffer("positive_token_suffix", embedding[:, :1, 1 + n_ctx:, :])  # positive prompt CLS, EOS
            self.register_buffer("negative_token_prefix", embedding[:, 1:, :1, :])  # SOS
            self.register_buffer("negative_token_suffix", embedding[:, 1:, 1 + n_ctx:, :])
            self.positive_tokenized_prompts = positive_tokenized_prompts.view(
                positive_tokenized_prompts.shape[0] * positive_tokenized_prompts.shape[1], -1)
            self.negative_tokenized_prompts = negative_tokenized_prompts.view(
                negative_tokenized_prompts.shape[0] * negative_tokenized_prompts.shape[1], -1)

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = "end"

    def forward(self, modify_to_ori=None):
        # modify_to_ori is a dic that transform the modified labels to original ones
        ctx_positive = self.ctx_positive

        # print('ctx_positive', ctx_positive.shape)
        ctx_negative = self.ctx_negative
        # ctx_negative = ctx_negative[0:1, 0:1, :].repeat(ctx_negative.shape[0], ctx_negative.shape[1], 1)
        # make ctx_negative[0,0,:] to ctx_negative
        if ctx_negative.shape[0] == 0:
            if ctx_positive.dim() == 3:
                ctx = ctx_positive.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
            else:
                ctx = ctx_positive
        else:
            if ctx_positive.dim() == 3:
                diff = ctx_positive.shape[1] - ctx_negative.shape[1]
                additional_rows = torch.zeros((ctx_negative.shape[0], diff, ctx_negative.shape[2])).cuda()
                additional_rows = additional_rows.to(ctx_negative.dtype)
                ctx_negative = torch.cat([additional_rows, ctx_negative], dim=1)
                ctx = torch.cat([ctx_positive, ctx_negative], dim=0)
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
            else:
                ctx = torch.cat([ctx_positive, ctx_negative], dim=1)
        prefix = self.token_prefix
        suffix = self.token_suffix

        if modify_to_ori is not None:
            ori_labels = list(modify_to_ori.values())
            ctx = ctx[ori_labels]
            prefix = prefix[ori_labels]
            suffix = suffix[ori_labels]
        prompts = torch.cat(
            [
                prefix,  # (n_cls,1+n_neg, 1, dim)
                ctx,  # (n_cls,1+n_neg, n_ctx, dim)
                suffix,  # (n_cls,1+n_neg, *, dim)
            ],
            dim=2,
        )
        return prompts

    def foward_positive(self):
        ctx_positive = self.ctx_positive
        if ctx_positive.dim() == 3:
            ctx = ctx_positive.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
        else:
            ctx = ctx_positive
        prefix = self.positive_token_prefix
        suffix = self.positive_token_suffix
        prompts = torch.cat(
            [
                prefix,  # (n_cls,1+n_neg, 1, dim)
                ctx,  # (n_cls,1+n_neg, n_ctx, dim)
                suffix,  # (n_cls,1+n_neg, *, dim)
            ],
            dim=2,
        )
        return prompts

    def foward_negative(self):
        ctx_negative = self.ctx_negative
        if ctx_negative.dim() == 3:
            ctx = ctx_negative.unsqueeze(0).expand(self.n_cls, -1, -1, -1)
        else:
            ctx = ctx_negative
        prefix = self.negative_token_prefix
        suffix = self.negative_token_suffix
        prompts = torch.cat(
            [
                prefix,  # (n_cls,1+n_neg, 1, dim)
                ctx,  # (n_cls,1+n_neg, n_ctx, dim)
                suffix,  # (n_cls,1+n_neg, *, dim)
            ],
            dim=2,
        )
        return prompts

    def update_ctx_positive(self, ctx_posi):
        noise_range = 1e-5
        noise_dist = dist.Uniform(low=-noise_range, high=noise_range, )
        if self.csc == 1:
            ctx_negative_repeated = ctx_posi.repeat(1, self.n_nega_ctx, 1, 1)
        else:
            ctx_negative_repeated = ctx_posi.repeat(self.n_nega_ctx, 1, 1)

        ctx_negative = ctx_negative_repeated + noise_dist.sample(ctx_negative_repeated.shape).to(
            self.ctx_negative.device)
        ctx_negative = ctx_negative.half()

        self.ctx_positive = nn.Parameter(ctx_posi, requires_grad=False)
        self.ctx_negative = nn.Parameter(ctx_negative, requires_grad=True)

    def update_ctx_negative(self, ctx_nega):
        self.ctx_negative = nn.Parameter(ctx_nega, requires_grad=False)

    def freeze_ctx_positive(self):
        self.ctx_positive = nn.Parameter(self.ctx_positive, requires_grad=False)

    def get_ctx_positive(self):
        return self.ctx_positive

    def get_positive_prompts(self, cls_ids):
        """
        单独获取指定类别的正提示词（用于prompt mixup）
        Args:
            cls_ids: [B]  tensor，需要获取正提示词的类别索引（如伪标签y_a、混合类别y_d）
        Returns:
            positive_prompts: [B, 1, seq_len, dim]，指定类别的正提示词嵌入（1对应正提示词维度）
        """
        ctx_positive = self.ctx_positive  # 可学习的正提示词上下文 [n_cls, 1, n_ctx, dim]（CSC=1时）或 [1, n_ctx, dim]（CSC=0时）
        batch_size = cls_ids.size(0)
        device = cls_ids.device

        # 处理CSC（类别专属上下文）和非CSC两种情况
        if self.csc > 0:
            # CSC=1：每个类别有专属正上下文，直接按cls_ids索引
            selected_ctx = ctx_positive[cls_ids]  # [B, 1, n_ctx, dim]
        else:
            # CSC=0：所有类别共享正上下文，广播到batch_size
            selected_ctx = ctx_positive.repeat(batch_size, 1, 1, 1)  # [B, 1, n_ctx, dim]

        # 拼接SOS前缀和CLS/EOS后缀（复用原有token_prefix/token_suffix的正提示词部分）
        # token_prefix: [n_cls, 1+n_nega_ctx, 1, dim] → 取正提示词部分 [n_cls, 1, 1, dim]
        prefix = self.token_prefix[:, :1, :, :]  # 仅保留正提示词的SOS前缀
        suffix = self.token_suffix[:, :1, :, :]  # 仅保留正提示词的CLS/EOS后缀

        # 按cls_ids筛选前缀/后缀（适配batch中不同类别）
        prefix = prefix[cls_ids]  # [B, 1, 1, dim]
        suffix = suffix[cls_ids]  # [B, 1, seq_len-n_ctx-1, dim]

        # 拼接成完整正提示词：prefix（SOS） + selected_ctx（可学习上下文） + suffix（CLS/EOS）
        positive_prompts = torch.cat([prefix, selected_ctx, suffix], dim=2)  # [B, 1, seq_len, dim]
        return positive_prompts

    def mix_positive_prompts(self, cls_a, cls_d, lambda_):
        """
        混合两个类别的正提示词（核心prompt mixup功能）
        Args:
            cls_a: [B] tensor，主类别（伪标签类别）
            cls_d: [B] tensor，混合类别（随机选择的类别）
            lambda_: 混合系数（与data mixup共享同一λ）
        Returns:
            mixed_prompts: [B, 1, seq_len, dim]，混合后的正提示词嵌入
        """
        # 步骤1：获取两个类别的正提示词
        prompt_a = self.get_positive_prompts(cls_a)  # [B, 1, seq_len, dim]（主类别正提示词）
        prompt_d = self.get_positive_prompts(cls_d)  # [B, 1, seq_len, dim]（混合类别正提示词）

        # 步骤2：按λ线性混合（与data mixup保持相同混合比例）
        # lambda_需广播到prompt维度：[B] → [B, 1, 1, 1]
        lambda_broadcast = lambda_.view(-1, 1, 1, 1).to(prompt_a.dtype)
        mixed_prompts = lambda_broadcast * prompt_a + (1 - lambda_broadcast) * prompt_d  # [B, 1, seq_len, dim]
        return mixed_prompts


class NegaTextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.transformer.eval()
        # --- 获取 token_embedding ---
        self.token_embedding = clip_model.token_embedding  # <--- Add this line
        # --- 获取结束 ---
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = next(clip_model.parameters()).dtype
        if (hasattr(clip_model, 'attn_mask')):
            self.attn_mask = clip_model.attn_mask
        else:
            self.attn_mask = None
        # print('attn_mask is ', self.attn_mask)

    def forward(self, prompts, tokenized_prompts):
        if len(prompts.shape) == 4:
            prompts = torch.flatten(prompts, start_dim=0, end_dim=1)
        x = prompts + self.positional_embedding.type(self.dtype)

        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(x.device)
            x = self.transformer(x, self.attn_mask)
        else:
            x = self.transformer(x)

        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

    """
        10.13 修改全局负提示词的初始化
    """

    def encode_global_neg(self, context_tensor, tokenized_prompts):
        """
        使用 learnable context_tensor 编码全局负提示。
        Args:
            context_tensor: [n_global_neg, n_ctx, embed_dim] - 可学习的上下文 token 嵌入
            tokenized_prompts: [n_global_neg, seq_len] - 对应模板文本的 token IDs
        Returns:
            text_features: [n_global_neg, embed_dim] - 全局负提示的文本特征
        """
        # 获取 token IDs 对应的基础嵌入
        # shape: [n_global_neg, seq_len, embed_dim]
        base_embeddings = self.token_embedding(tokenized_prompts).type(self.dtype)

        # --- 关键：将 learnable context_tensor 嵌入到 base_embeddings 中 ---
        # 假设 context_tensor 对应 tokenized_prompts 中最后 n_ctx 个非填充/非EOS token 的位置
        # 需要为每一行找到 EOS token 的位置
        # tokenized_prompts shape: [n_global_neg, seq_len]
        # 找到每一行中 EOS token (49407) 的列索引
        # argmax(dim=-1) 会找到每一行中最大值的索引，对于 CLIP tokenization，EOS (49407) 通常是序列中最大的值之一，并且位于有效内容的末尾。
        # 但注意，如果有多个 49407 (EOS/PAD)，argmax 可能指向最后一个。
        # 或者，我们可以找到每一行第一个 49407 的位置。
        # 使用 argmax 是一种常见且相对鲁棒的方法，因为它能找到序列的“结尾”。
        eos_positions = tokenized_prompts.argmax(dim=1)  # [n_global_neg] - 每一行 EOS 的列索引
        # print(f"EOS positions shape: {eos_positions.shape}, values: {eos_positions[:5]}") # Debug print

        # 计算 context_tensor 应该插入的起始位置 (从 EOS 往前数 n_ctx 个)
        # context_tensor.shape[1] 即为 n_ctx
        # 注意：这里假设 n_ctx <= (eos_pos - start_of_content_pos + 1) for all rows
        # For CLIP-like templates, this is usually true if n_ctx is small enough.
        start_ctx_indices = eos_positions - context_tensor.shape[1]  # [n_global_neg] - 每一行 context 开始的列索引
        # print(f"Start context indices shape: {start_ctx_indices.shape}, values: {start_ctx_indices[:5]}") # Debug print

        # --- 循环方式替换，因为切片需要对每个样本指定不同的位置 ---
        # base_embeddings shape: [n_global_neg, seq_len, embed_dim]
        # context_tensor shape: [n_global_neg, n_ctx, embed_dim]
        for i in range(context_tensor.shape[0]):  # Iterate over n_global_neg
            start_idx = start_ctx_indices[i].item()
            end_idx = eos_positions[i].item()
            base_embeddings[i, start_idx:end_idx, :] = context_tensor[i]  # Replace context tokens for prompt i
        # --- 循环替换结束 ---

        # 标准 CLIP 文本编码器前向过程
        x = base_embeddings + self.positional_embedding.type(self.dtype)

        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(x.device)
            x = self.transformer(x, self.attn_mask)
        else:
            x = self.transformer(x)

        x = self.ln_final(x).type(self.dtype)
        # Use the EOS token position to get the final feature (same as before)
        # tokenized_prompts.argmax(dim=-1) gives the EOS position for each sample
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class NegaPromptCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = NegaPromptLearner(cfg, classnames, clip_model).cuda()

        self.n_nega_ctx = cfg['NEGA_CTX']
        self.stage = cfg['stage']
        self.cfg = cfg  # 添加cfg引用
        self.classnames = classnames  # 添加classnames引用
        self.clip_model = clip_model

        # # ---------------------- 新增：加载caption-level提示词并生成固定锚点 ----------------------
        # # 1. 加载JSON文件中的Top3 caption提示词
        # self.caption_prompt_path = cfg.get('caption_prompt_path', 'cifar80n_top3_prompts.json')
        # with open(self.caption_prompt_path, 'r', encoding='utf-8') as f:
        #     caption_data = json.load(f)
        # self.caption_prompts = caption_data['结果']  # 格式：{class: [prompt1, prompt2, prompt3]}
        #
        # # 2. 生成每个类别的caption锚点（Top3提示词嵌入取平均，冻结）
        # self.caption_anchors = self._generate_caption_anchors(clip_model).cuda()  # [n_class, 512]
        # self.caption_anchors.requires_grad = False  # 固定锚点，不参与训练
        # # -------------------------------------------------------------------------------------

        if 2 <= self.stage < 100:
            n_class = len(self.classnames)
            n_global_neg = int(self.cfg.get('GLOBAL_NEGA_CTX', 20))  # number of global neg prompts

            # --- 从 prompt_learner 获取 ---
            embed_dim = self.prompt_learner.ctx_dim  # <--- 从 prompt_learner 获取 embed_dim
            n_ctx = self.prompt_learner.n_ctx  # <--- 从 prompt_learner 获取 n_ctx
            # --- 获取结束 ---

            _init_noise_std = 0.02  # 初始化噪声标准差

            # 随机初始化全局负提示的上下文 token
            # [n_global_neg, n_ctx, dim]
            global_neg_ctx_init = torch.randn(n_global_neg, n_ctx, embed_dim) * _init_noise_std

            # 将初始化好的上下文 token 作为可学习参数
            self.global_neg_ctx = nn.Parameter(global_neg_ctx_init, requires_grad=True)

            # 定义用于 tokenization 的模板文本
            import clip
            _tokenizer = clip.tokenize  # 获取 CLIP 的 tokenizer
            template_text = "a photo of an unknown object"  # 语义起点，但核心靠 self.global_neg_ctx 学习
            # Tokenize the template text n_global_neg times
            # 注意：这里 tokenized 的长度需要与 n_ctx + len(class_token) + padding 对应
            # 但全局负提示没有特定的 class_token，所以我们需要一个固定长度的 prompt 模板
            # 例如，如果 CLIP prompt 模板是 "a photo of a [CLASS]"，其 token 长度是固定的 (e.g., 77)
            # 我们需要为 "a photo of an unknown object" 生成 tokenized_prompts
            # 这个 tokenized_prompts 的长度应该与 CLIP 的 max_length 相同 (通常是 77)
            # CLIP.tokenize 会自动处理 padding 和 EOS token
            self.global_neg_tokenized_prompts = _tokenizer([template_text] * n_global_neg).to(
                'cuda')  # [n_global_neg, seq_len] (e.g., [n_global_neg, 77])

        else:
            self.global_neg_ctx = None
            self.global_neg_tokenized_prompts = None

        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.ytVisualWithLocal = ytVisualWithLocal(clip_model.visual)

        # === LoRA 注入（仅视觉端），默认仅 qkv，r=8, α=16 ===
        try:
            # 检查是否启用 DeLoRA，若是则启用双 LoRA
            dual_lora = self.cfg.get('delora', False)
            # 当启用 DeLoRA 时，固定 prompt_learner 的参数，维持本来的a photo of a {class}的结构
            if dual_lora:
                print(
                    f"[DeLoRA Mode] Freezing prompt parameters to maintain original 'a photo of a {{class}}' structure")

                # Only inject LoRA into the last 3 layers of the transformer
                # inject_lora_to_visual(self.image_encoder,
                #                       target_keywords=['attn.qkv', 'mlp.c_fc', 'mlp.c_proj', 'attn.out_proj'], rank=8,
                #                       alpha=16, dropout=0.0, verbose=True, dual_lora=dual_lora, last_n_layers=3)

                inject_lora_to_visual(self.image_encoder,
                                      target_keywords=['attn.qkv', 'mlp.c_fc', 'mlp.c_proj', 'attn.out_proj'], rank=cfg.get('rank', 2),
                                      alpha=8, dropout=0.1, verbose=True, dual_lora=dual_lora, last_n_layers=cfg.get('partial', 1))
                # 只冻结视觉骨干，保留 LoRA (A/B) 可训练；不影响文本侧与提示词参数
                freeze_backbone_except_lora(self.image_encoder)
        except Exception as e:
            print(f"[LoRA] Injection skipped due to error: {e}")

        self.text_encoder = NegaTextEncoder(clip_model).cuda()
        # self.text_encoder = TextEncoder(clip_model)
        # self.weight_yes = self.merge_yes_feature(classnames, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = next(clip_model.parameters()).dtype
        self.classnames = classnames
        self.positive_text_features = None

    def get_lora_logits(self, data):
        """
        获取 Clean LoRA 应用时的 logits
        输入: data
        返回值：clean lora 和 固定提示词的logits

        """
        logit_scale = self.logit_scale.exp()

        # 检查是否启用双 LoRA
        lora_layers = []
        has_dual_lora = False
        for m in self.image_encoder.modules():
            if isinstance(m, LoRALinear) and m.dual_lora:
                lora_layers.append(m)
                has_dual_lora = True
        if not has_dual_lora or not lora_layers:
            # 如果模型不支持双 LoRA，返回 None
            return None

        # 获取文本特征 之前的依赖模型本身，我这是一个模型，这样做就有正提示词了。
        prompts = self.prompt_learner(modify_to_ori=None)
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        # torch.save(text_features.cpu(), "text_features.pt")
        # print("save!")
        # exit(0)

        ### 这里相当于二阶段
        # text_features = torch.load("/data/tyang/aaaa_now/compare/text_features.pt", map_location='cuda')

        # === 获取 Clean LoRA 的 logits ===
        # 设置所有 LoRALinear 层使用 Clean LoRA
        for m in lora_layers:
            m._use_clean_lora = True
            m._not_use_lora = False

        # 前向传播获取图像特征


        image_features_clean = self.image_encoder(data.type(self.dtype))
        image_features_clean = torch.nn.functional.normalize(image_features_clean, dim=-1)
        logits_clean = logit_scale * image_features_clean @ text_features.t()



        # === 重置 LoRALinear 层的设置 ===
        for m in lora_layers:
            m._use_clean_lora = None
            m._merge_lora = None

        return logits_clean

    def get_original_logits(self, data):

        """
        计算不经过Lora模块得到的image和不固定提示的logits
        """

        lora_layers = []
        logit_scale = self.logit_scale.exp()
        for m in self.image_encoder.modules():
            if isinstance(m, LoRALinear) and m.dual_lora:
                lora_layers.append(m)

        # 设置所有 LoRALinear 层不使用 LoRA
        for m in lora_layers:
            m._use_clean_lora = False
            m._not_use_lora = True

        # 获取文本特征
        prompts = self.prompt_learner(modify_to_ori=None)
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = torch.nn.functional.normalize(text_features, dim=-1)

        image_features = self.image_encoder(data.type(self.dtype))
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
        logits = logit_scale * image_features @ text_features.t()

        # 重置 LoRALinear 层的设置
        for m in lora_layers:
            m._use_clean_lora = None
            m._not_use_lora = None

        return logits


    def forward_negative(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        negative_prompts = self.prompt_learner.foward_negative()
        negative_tokenized_prompts = self.prompt_learner.negative_tokenized_prompts
        negative_text_features = self.text_encoder(negative_prompts,
                                                   negative_tokenized_prompts)  # (1000*n_nega_ctx) * 512)
        positive_text_features = self.positive_text_features  # 1000*512
        # fusion the text_features that positive, negative, positive, negative, ...
        positive_text_features = positive_text_features.view(positive_text_features.shape[0], 1, -1)
        negative_text_features = negative_text_features.view(positive_text_features.shape[0], self.n_nega_ctx, -1)
        text_features = torch.cat([positive_text_features, negative_text_features], dim=1)
        text_features = text_features.view(text_features.shape[0] * text_features.shape[1], -1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = (logit_scale * image_features @ text_features.t())
        return logits, text_features

    def forward(self, image, modify_to_ori=None):
        # if self.stage == 3:
        #     return self.forward_negative(image)

        prompts = self.prompt_learner(modify_to_ori)
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)


        image_features = self.image_encoder(image.type(self.dtype))

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        logits = (logit_scale * image_features @ text_features.t())
        # if 2 <= self.stage < 100:
        #     # 返回 logits, 原有text_features, 全局负提示特征
        #     logits_global_neg = (logit_scale * image_features @ global_neg_features.t())
        #     logits_local_neg = (logit_scale * locals_features @ global_neg_features.t())
        #     return logits, text_features, logits_global_neg, global_neg_features, logits_local, logits_local_neg

        return logits

    def forward_joint_warmup(self, image, modify_to_ori=None):
        """
        专门用于联合预热训练的forward函数：
        计算经过LoRA的视觉特征与正提示文本特征的相似度
        """
        # 1. 获取正提示文本特征
        prompts = self.prompt_learner(modify_to_ori)
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # 2. 获取经过LoRA的视觉特征
        image_features, locals_features = self.ytVisualWithLocal(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 3. 计算相似度 (使用CLIP的logit缩放)
        logit_scale = self.logit_scale.exp()
        logits = (logit_scale * image_features @ text_features.t())

        # 仅返回我们需要的结果：logits用于CE损失计
        return logits

    def forward_test(self, image, text_features=None):
        image_features, locals_features = self.ytVisualWithLocal(image.type(self.dtype))

        # image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        locals_features = locals_features / locals_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_global = (logit_scale * image_features @ text_features.t())
        logits_local = (logit_scale * locals_features @ text_features.t())

        return logits_global, logits_local, text_features

    def forward_global(self, image, global_neg_features=None):
        image_features, _ = self.ytVisualWithLocal(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_global = (logit_scale * image_features @ global_neg_features.t())
        return logits_global

    def get_visual_features(self, image):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def get_ctx_posi(self, ctx_posi):
        self.prompt_learner.update_ctx_positive(ctx_posi)
        # get positive_text_features
        prompts = self.prompt_learner.foward_positive()
        tokenized_prompts = self.prompt_learner.positive_tokenized_prompts
        self.positive_text_features = self.text_encoder(prompts, tokenized_prompts)

    def get_ctx_nega(self, ctx_nega):
        self.prompt_learner.update_ctx_negative(ctx_nega)

    def freeze_ctx_posi(self):
        self.prompt_learner.freeze_ctx_positive()

    def radius(self):
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        n_nega_ctx = self.cfg['NEGA_CTX']
        ensemble_text_features = text_features.view(int(text_features.shape[0] / (1 + n_nega_ctx)), 1 + n_nega_ctx, -1)
        positive_text_features = ensemble_text_features[:, 0, :]
        negative_text_features = ensemble_text_features[:, 1:, :]
        radius = torch.Tensor(positive_text_features.shape[0], n_nega_ctx)
        logit_scale = self.logit_scale.exp()
        for i in range(positive_text_features.shape[0]):
            positive_feature = positive_text_features[i, :]
            negative_features = negative_text_features[i, :, :]

            cos_sim = torch.nn.functional.cosine_similarity(negative_features, positive_feature.unsqueeze(0), dim=1)
            one_radius = 1 - cos_sim

            # one_radius = logit_scale*positive_feature @ negative_features.t()

            radius[i, :] = one_radius

        return radius

    def draw_tsne_plot(self, testloader, outloader, log_dir, expr_name, epoch):
        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)
        text_features = text_features.reshape(prompts.shape[0], prompts.shape[1], text_features.shape[-1])
        pos_feature = text_features[:, 0:1, :].cpu()
        pos_feature = pos_feature / pos_feature.norm(dim=-1, keepdim=True)
        neg_feature = text_features[:, 1:, :].cpu()
        neg_feature = neg_feature / neg_feature.norm(dim=-1, keepdim=True)
        pos_label = torch.arange(pos_feature.shape[0])[..., None]  # shape = [nclass, 1]
        neg_label = torch.full((neg_feature.shape[0], neg_feature.shape[1]),
                               pos_feature.shape[0])  # shape = [nclass, n_nega]

        n_class = pos_feature.shape[0]

        all_image_feature = torch.Tensor()
        all_image_label = torch.Tensor()
        for data, labels in testloader:
            data, labels = data.cuda(), labels.cuda()
            with torch.set_grad_enabled(False):
                image_features = self.image_encoder(data.type(self.dtype)).cpu()
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                all_image_feature = torch.cat([all_image_feature, image_features], dim=0)
                all_image_label = torch.cat([all_image_label, labels.cpu()], dim=0)

        all_text_feature = torch.Tensor()
        all_text_feature = torch.cat([all_text_feature, pos_feature], dim=1)
        all_text_feature = all_text_feature.view(-1, all_text_feature.shape[-1])

        all_text_label = torch.Tensor()
        all_text_label = torch.cat([all_text_label, pos_label], dim=1)
        all_text_label = all_text_label.view(-1)

        total_feature = torch.cat([all_text_feature, all_image_feature], dim=0)
        total_label = torch.cat([all_text_label, -1 * (all_image_label + 1)], dim=0)

        X = total_feature.detach().numpy()
        tsne_model = TSNE(metric="precomputed", n_components=2, init="random", perplexity=30)
        distance_matrix = pairwise_distances(X, X, metric='cosine', n_jobs=-1)

        data = torch.Tensor(tsne_model.fit_transform(distance_matrix))
        target = total_label
        dataset = TensorDataset(data, target)
        loader = DataLoader(dataset, batch_size=256)
        plt.figure()
        for x, y in loader:
            # 样本点显示
            idx_pos_text = (y < n_class) & (y >= 0)  # 正向样本
            idx_nega_text = (y >= n_class)  # 负向样本
            idx_pos_image = (y < 0) & (y >= -n_class)
            idx_nega_image = (y < -n_class)

            plt.scatter(x[idx_pos_text, 0], x[idx_pos_text, 1], marker='o', c=y[idx_pos_text], alpha=0.2,
                        cmap=plt.cm.get_cmap("plasma", n_class + 1), label='pos')
            plt.scatter(x[idx_nega_text, 0], x[idx_nega_text, 1], marker='o', c=y[idx_nega_text], alpha=0.2,
                        cmap=plt.cm.get_cmap("summer", n_class + 1), label='nega')
            plt.scatter(x[idx_pos_image, 0], x[idx_pos_image, 1], marker='x', c=-1 * y[idx_pos_image] - 1, alpha=0.4,
                        cmap=plt.cm.get_cmap("plasma", n_class + 1), label='pos')
            plt.scatter(x[idx_nega_image, 0], x[idx_nega_image, 1], marker='x', c=-1 * y[idx_nega_image] - 1, alpha=0,
                        cmap=plt.cm.get_cmap("summer", n_class + 1), label='nega')
        # handles, labels = plt.gca().get_legend_handles_labels()
        # plt.legend(handles, labels)
        dir_path = os.path.join(log_dir, 'tsne', expr_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        plt.savefig(os.path.join(dir_path, 'tsne_plot_epoch_{}.pdf'.format(epoch)))
        plt.close()
