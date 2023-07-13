from typing import Optional, Tuple
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPT2Config:
    model_name: str = 'gpt2'

    num_layers: int = 12
    num_attn_heads: int = 12
    hidden_size: int = 768
    vocab_size: int = 50257
    block_size: int = 1024

    norm_eps: float = 1e-5
    embed_dropout: float = 0.0
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0


# num_layers, num_attn_heads and hidden_size are determined from model_type
gpt2_meta = {
    'gpt2': dict(model_name='gpt2', num_layers=12, num_attn_heads=12, hidden_size=768),  # 124M params
    'gpt2-medium': dict(model_name='gpt2-medium', num_layers=24, num_attn_heads=16, hidden_size=1024),  # 350M params
    'gpt2-large': dict(model_name='gpt2-large', num_layers=36, num_attn_heads=20, hidden_size=1280),  # 774M params
    'gpt2-xl': dict(model_name='gpt2-xl', num_layers=48, num_attn_heads=25, hidden_size=1600),  # 1558M params
}


def get_gpt2_config(
    model_type: str,
    embed_dropout: float = 0.0,
    attn_dropout: float = 0.0,
    resid_dropout: float = 0.0,
) -> GPT2Config:
    """Central place to get GPT2 model configurations."""
    assert model_type in list(gpt2_meta.keys())

    _args = gpt2_meta[model_type]

    return GPT2Config(
        model_name=_args['model_name'],
        num_layers=_args['num_layers'],
        num_attn_heads=_args['num_attn_heads'],
        hidden_size=_args['hidden_size'],
        vocab_size=50257,  # can also round to 128x393 for better performance
        embed_dropout=embed_dropout,
        attn_dropout=attn_dropout,
        resid_dropout=resid_dropout,
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=True)
        # output projection
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        # regularization
        self.resid_dropout = nn.Dropout(config.resid_dropout)
        self.attn_dropout = config.attn_dropout

        self.num_attn_heads = config.num_attn_heads
        self.hidden_size = config.hidden_size

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # batch size, sequence length, embedding dimensionality (hidden_size)
        (B, T, C) = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.hidden_size, dim=2)
        # (B, nh, T, hs)
        k = k.view(B, T, self.num_attn_heads, C // self.num_attn_heads).transpose(1, 2)
        # (B, nh, T, hs)
        q = q.view(B, T, self.num_attn_heads, C // self.num_attn_heads).transpose(1, 2)
        # (B, nh, T, hs)
        v = v.view(B, T, self.num_attn_heads, C // self.num_attn_heads).transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels.
        # However we have to manually compute the attention mask to correctly handle padding,
        # since we can't use both attn_mask and is_causal=True at the same time,
        # this is very slow and consumes more GPU RAM
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_dropout if self.training else 0,
            is_causal=True if attn_mask is None else False,
            # scale=1 / math.sqrt(C // self.num_attn_heads),
        )

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.c_proj(y)
        return self.resid_dropout(y)


class MLP(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, 4 * config.hidden_size, bias=True)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.hidden_size, config.hidden_size, bias=True)
        self.resid_dropout = nn.Dropout(config.resid_dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.resid_dropout(x)


class EncoderBlock(nn.Module):
    """Encoder block for GPT-2 model"""

    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.mh_attn = CausalSelfAttention(config=config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp = MLP(config=config)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.mh_attn(self.ln_1(x), attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2Torso(nn.Module):
    """GPT-2 model torso without output head."""

    def __init__(
        self,
        model_type: str,
        embed_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = get_gpt2_config(
            model_type=model_type,
            embed_dropout=embed_dropout,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
        )
        self.model_name = self.config.model_name
        self.vocab_size = self.config.vocab_size
        self.block_size = self.config.block_size

        self.token_embed = nn.Embedding(self.config.vocab_size, self.config.hidden_size)  # token embedding
        self.position_embed = nn.Embedding(self.config.block_size, self.config.hidden_size)  # position embedding
        self.token_embed_drop = nn.Dropout(self.config.embed_dropout)
        self.position_embed_drop = nn.Dropout(self.config.embed_dropout)
        # transformer encoder layers
        self.layers = nn.ModuleList([EncoderBlock(config=self.config) for _ in range(self.config.num_layers)])
        # Post layernorm after transformer decoder blocks
        self.post_ln = nn.LayerNorm(self.config.hidden_size, eps=self.config.norm_eps)

        # init all weights
        _init_weights(self)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.num_layers))

    def get_metadata(self):
        return self.config

    def forward(self, token_ids: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        device = token_ids.device
        b, t = token_ids.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, maximum sequence length is {self.config.block_size}"
        # Is this still correct when we use padding to make batch of sequences have the same length?
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

        tok_embed = self.token_embed(token_ids)  # token embeddings of shape (b, t, hidden_size)
        tok_embed = self.token_embed_drop(tok_embed)
        pos_embed = self.position_embed(pos)  # position embeddings of shape (1, t, hidden_size)
        pos_embed = self.position_embed_drop(pos_embed)  # position embeddings of shape (1, t, hidden_size)

        x = tok_embed + pos_embed

        for block in self.layers:
            x = block(x, attn_mask)

        x = self.post_ln(x)

        return x


# class GPT2Model(nn.Module):
#     """GPT-2 model with LM head to predict next token."""

#     def __init__(
#         self,
#         model_type: str,
#         embed_dropout: float = 0.0,
#         attn_dropout: float = 0.0,
#         resid_dropout: float = 0.0,
#     ) -> None:
#         super().__init__()
#         self.config = get_gpt2_config(
#             model_type=model_type,
#             embed_dropout=embed_dropout,
#             attn_dropout=attn_dropout,
#             resid_dropout=resid_dropout,
#         )
#         self.model_name = self.config.model_name
#         self.vocab_size = self.config.vocab_size
#         self.block_size = self.config.block_size

#         self.token_embed = nn.Embedding(self.config.vocab_size, self.config.hidden_size)  # token embedding
#         self.position_embed = nn.Embedding(self.config.block_size, self.config.hidden_size)  # position embedding
#         self.embed_drop = nn.Dropout(self.config.embed_dropout)
#         self.layers = nn.ModuleList([EncoderBlock(self.config) for _ in range(self.config.num_layers)])
#         # Post layernorm after transformer decoder blocks
#         self.post_ln = nn.LayerNorm(self.config.hidden_size, eps=self.config.norm_eps)
#         # Final output layer don't have bias
#         self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

#         # init all weights
#         _init_weights(self)
#         # apply special scaled init to the residual projections, per GPT-2 paper
#         for pn, p in self.named_parameters():
#             if pn.endswith("c_proj.weight"):
#                 torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.num_layers))

#     def get_num_params(self):
#         return sum(p.numel() for p in self.parameters() if p.requires_grad)

#     def get_metadata(self):
#         return self.config

#     def forward(self, token_ids: torch.Tensor, attn_mask: torch.Tensor = None, is_inference: bool = False) -> torch.Tensor:
#         device = token_ids.device
#         b, t = token_ids.size()
#         assert (
#             t <= self.config.block_size
#         ), f"Cannot forward sequence of length {t}, maximum sequence length is {self.config.block_size}"
#         # Is this still correct when we use padding to make batch of sequences have the same length?
#         pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

#         tok_embed = self.token_embed(token_ids)  # token embeddings of shape (b, t, hidden_size)
#         pos_embed = self.position_embed(pos)  # position embeddings of shape (1, t, hidden_size)
#         x = self.embed_drop(tok_embed + pos_embed)

#         for block in self.layers:
#             x = block(x, attn_mask)

#         x = self.post_ln(x)

#         if is_inference:
#             x = x[:, -1, :]

#         logits = self.lm_head(x)

#         return logits


class GPT2LMHeadModel(nn.Module):
    """GPT-2 model with LM head to predict next token."""

    def __init__(
        self,
        model_type: str,
        embed_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.transformer = GPT2Torso(
            model_type=model_type,
            embed_dropout=embed_dropout,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
        )

        self.config = self.transformer.config

        self.model_name = self.config.model_name
        self.vocab_size = self.config.vocab_size
        self.block_size = self.config.block_size

        # LM head to predict next token
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        _init_weights(self.lm_head)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_metadata(self):
        return self.config

    def forward(self, token_ids: torch.Tensor, attn_mask: torch.Tensor = None, is_inference: bool = False) -> torch.Tensor:
        x = self.transformer(token_ids=token_ids, attn_mask=attn_mask)

        if is_inference:
            x = x[:, -1, :]

        logits = self.lm_head(x)

        return logits


class GPT2ScalarModel(nn.Module):
    """GPT-2 model with scalar head."""

    def __init__(
        self,
        model_type: str,
        embed_dropout: float = 0.0,
        attn_dropout: float = 0.0,
        resid_dropout: float = 0.0,
    ) -> None:
        self.transformer = GPT2Torso(
            model_type=model_type,
            embed_dropout=embed_dropout,
            attn_dropout=attn_dropout,
            resid_dropout=resid_dropout,
        )

        self.config = self.transformer.config

        self.model_name = self.config.model_name
        self.vocab_size = self.config.vocab_size
        self.block_size = self.config.block_size

        # Scalar head to predict reward or value.
        self.scalar_head = nn.Linear(self.config.hidden_size, 1)

        init_std = 1.0 / math.sqrt(self.config.hidden_size + 1)
        torch.nn.init.normal_(self.head.weight, std=init_std)
        torch.nn.init.zeros_(self.head.bias)

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_metadata(self):
        return self.config

    def forward(self, token_ids: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.transformer(token_ids=token_ids, attn_mask=attn_mask)

        value = self.scalar_head(x)
        return value


def _init_weights(module: torch.nn.Module) -> None:
    if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def create_optimizer(
    model: torch.nn.Module, lr: float, eps: float, weight_decay: float, betas: Tuple[float], fused: bool
) -> torch.optim.AdamW:
    """
    Returns the PyTorch AdamW optimizer for the model,
    where we skip apply weight decay to layer norm, embedding, and all bias,
    and apply weight decay to the reset of parameters.
    """

    # filter out those do not require gradients
    params_dict = {p_name: params for p_name, params in model.named_parameters() if params.requires_grad}

    # Create empty lists to store parameters for weight decay and no weight decay.
    decay = []
    no_decay = []

    for p_name, params in params_dict.items():
        # Check for parameters corresponding to torch.nn.LayerNorm or torch.nn.Embedding.
        # Note we use hard-coded names where 'ln' is for LayerNorm, and 'embed' is for Embedding, this works better with FSDP
        if (
            p_name.endswith("bias")
            or p_name.endswith("ln_1.weight")
            or p_name.endswith("ln_2.weight")
            or p_name.endswith("post_ln.weight")
            or p_name.endswith("position_embed.weight")
            or p_name.endswith("token_embed.weight")
        ):
            no_decay.append(params)
        else:
            decay.append(params)

    num_decay_params = sum(p.numel() for p in decay)
    num_nodecay_params = sum(p.numel() for p in no_decay)
    total_num_params = sum(p.numel() for p in params_dict.values())
    assert num_decay_params + num_nodecay_params == total_num_params

    print(f"--> num decayed parameter tensors: {len(decay)}, with {num_decay_params:,} parameters")
    print(f"--> num non-decayed parameter tensors: {len(no_decay)}, with {num_nodecay_params:,} parameters")

    # create the pytorch optimizer object
    optim_groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=lr, eps=eps, betas=betas, fused=fused)
    return optimizer
