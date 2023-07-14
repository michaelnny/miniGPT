"""Merge LoRA fine-tunned checkpoint and pretrained checkpoint into a single checkpoint file"""
import sys
import time
import os
from pathlib import Path
from typing import Optional


import torch
import torch.nn as nn

from models import GPT2LMHeadModel, GPT2ScalarModel, lora


def del_lora_state_dict(model: nn.Module):
    base_model_dict = model.state_dict()
    key_to_delete = [k for k in base_model_dict if "lora_" in k]
    for del_key in key_to_delete:
        del base_model_dict[del_key]
    return base_model_dict


def lora_model_lookup(checkpoint: dict) -> int:
    """Returns the LoRA rank from the adapter checkpoint."""
    return checkpoint["transformer.layers.0.mh_attn.c_attn.lora_B"].shape[1]


def merge_lora_checkpoint(
    model_type: str,
    lora_ckpt_path: str,
    pretrained_ckpt_path: str,
    save_path: str,
    is_scalar_head: bool = False,
) -> None:
    """Merges LoRA weights with pretrained base model.

    Args:
        model_type: The GPT-2 model type, supports 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'.
        lora_ckpt_path: Path to the checkpoint with trained LoRA weights, which are the output of
            `finetune_lora.py`.
        pretrained_ckpt_path: The pretrained checkpoint used in side the `finetune_lora.py` when start the fine-tuning.
        save_path: target path to save the merged stat_dict.
    """

    assert model_type in ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')

    if not os.path.exists(lora_ckpt_path):
        raise ValueError(f"LoRA checkpoint file {lora_ckpt_path} does not exist, aborting...")
    if not os.path.exists(pretrained_ckpt_path):
        raise ValueError(f"Pretrained checkpoint file {pretrained_ckpt_path} does not exist, aborting...")

    if os.path.exists(save_path):
        print(f"The checkpoint file {save_path} already exists, aborting...")
        return

    print("Loading model checkpoints ...")

    pretrained_checkpoint = torch.load(pretrained_ckpt_path)
    lora_checkpoint = torch.load(lora_ckpt_path)

    # find the rank from LoRA checkpoint
    rank = lora_model_lookup(lora_checkpoint)

    with lora(r=rank, alpha=16, dropout=0.0, enabled=True):
        if is_scalar_head:
            model = GPT2ScalarModel(model_type)
        else:
            model = GPT2LMHeadModel(model_type)
        # 1. Load the pretrained weights
        model.load_state_dict(pretrained_checkpoint, strict=False)
        # 2. Load the fine-tuned lora weights
        model.load_state_dict(lora_checkpoint, strict=False)

    model.eval()
    merged_model_dict = del_lora_state_dict(model)
    print("Saving LoRA to base model weights ...")
    torch.save(merged_model_dict, save_path)
    print(f"Merged model state dict saved at {save_path}")


if __name__ == "__main__":
    merge_lora_checkpoint(
        model_type="gpt2-xl",
        lora_ckpt_path="./checkpoints/finetune_lora/lora_model_gpt2-xl-iter-1000.pt",
        pretrained_ckpt_path="./checkpoints/gpt2-xl-openai-pretrained.pt",
        save_path="./checkpoints/gpt2-xl-finetune-iter-1000-merged.pt",
    )
