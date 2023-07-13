import os
import itertools
import functools
from typing import Tuple
import tqdm
import random
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist

from torch.utils.data.distributed import DistributedSampler

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    StateDictType,
)

from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler


from models import GPT2LMHeadModel, EncoderBlock, create_optimizer
from custom_dataset import FineTuneDataset

from configs.finetune import config as cfg

from utils import (
    CosineDecayWithWarmupLRScheduler,
    Memory_Maximizer,
    load_full_state_model_checkpoint,
    load_full_state_optimizer_checkpoint,
    save_full_state_model_checkpoint,
    save_full_state_optimizer_checkpoint,
    format_to_gb,
)


def setup():
    # initialize the process group
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    print(f"clearing cache for rank {rank}")
    torch.cuda.empty_cache()


# FSDP activation checkpointing
non_reentrant_wrapper = functools.partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)


def apply_fsdp_checkpointing(model):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    check_fn = lambda submodule: isinstance(submodule, EncoderBlock)

    apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)


def create_trace_profiler(tb_trace_dir):
    torch_profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=2, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_trace_dir),
        profile_memory=True,
        with_stack=False,
        record_shapes=True,
    )

    return torch_profiler


def compute_finetune_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    assert len(logits.shape) == 3  # [B, max_seq_length, vocab_size]
    assert len(targets.shape) == len(mask.shape) == 2  # [B, max_seq_length]
    assert logits.shape[0] == targets.shape[0] == mask.shape[0]

    B, T, *_ = logits.shape

    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')

    assert not torch.any(torch.isnan(loss))

    loss = loss.view(B, T)

    assert loss.shape == mask.shape

    # loss mask is defined as: -1s are prompt tokens, 1s are completion tokens, and 0s the padding tokens
    # note here prompt is less important than completion
    weights = mask.float().masked_fill(mask == -1, cfg.prompt_loss_weight).masked_fill(mask == 1, cfg.completion_loss_weight)
    loss *= weights

    loss = torch.mean(loss)

    return loss


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> Tuple[int, float]:
    assert len(logits.shape) == 3  # [B, max_seq_length, vocab_size]
    assert len(targets.shape) == 2  # [B, max_seq_length]
    assert targets.shape == mask.shape  # [B, max_seq_length]
    assert logits.shape[0] == targets.shape[0]

    # loss mask is defined as: -1s are prompt tokens, 1s are completion tokens, and 0s the padding tokens
    # only include completion when compute accuracy
    weights = mask.float().masked_fill(mask == -1, 0)

    # get the index of the max log-probability
    pred = torch.softmax(logits, dim=-1).argmax(dim=-1)

    correct = pred.eq(targets.view_as(pred)).float()

    # only consider completion when compute metrics
    correct *= weights
    num_accurate = correct.sum().item()
    num_samples = weights.bool().sum().item()

    return (num_accurate, num_samples)


def run_single_train_step(
    model,
    rank,
    world_size,
    train_loader,
    optimizer,
    scheduler,
    scaler=None,
    return_stats=False,
):
    """A single training iteration consists of N micro batch * M gradient accumulation steps.

    ```
    optimizer.zero_grad()
    for step in range(gradient_accum_steps):
        data, target = next(train_loader)
        output = model(data)
        loss = compute_pre_train_loss(output, target)
        loss.backward()

    optimizer.step()
    ```

    """

    local_rank = int(os.environ["LOCAL_RANK"])

    if return_stats:
        fsdp_metrics = torch.zeros(5).to(local_rank)

    # prepare for next update
    optimizer.zero_grad(set_to_none=True)

    for x, y, attn_mask, loss_mask in itertools.islice(train_loader, cfg.gradient_accum_steps):
        x, y, loss_mask = (
            x.to(local_rank, non_blocking=True),
            y.to(local_rank, non_blocking=True),
            loss_mask.to(local_rank, non_blocking=True),
        )

        # use pre-computed attention mask to handle padded sequence,
        # this is very slow (2-3x slower), also requires huge amount of GPU RAM...
        if attn_mask is not None:
            attn_mask = attn_mask.to(local_rank, non_blocking=True)

        output = model(x, attn_mask)

        loss = compute_finetune_loss(output, y, loss_mask)
        # scale the loss to account for gradient accumulation
        scaled_loss = loss / cfg.gradient_accum_steps

        if scaler is not None:  # when using float16
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        if return_stats:
            num_acc, num_samples = compute_metrics(output, y, loss_mask)
            fsdp_metrics[0] += loss.item()  # sum up batch loss
            fsdp_metrics[1] += np.exp(loss.item())  # sum up perplexity
            fsdp_metrics[2] += 1  # increase number of batches
            fsdp_metrics[3] += num_acc  # sum up number of accurate prediction tokens
            fsdp_metrics[4] += num_samples  # sum up number of tokens

    if scaler is not None:  # when using float16
        if cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)  # unscale before clip gradients
            # FSDP needs to use this method to clip gradient norm instead of torch.nn.utils.clip_grad_norm_
            model.clip_grad_norm_(cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()  # adjust scaling for next batch
    else:
        if cfg.grad_clip != 0.0:
            model.clip_grad_norm_(cfg.grad_clip)
        optimizer.step()

    scheduler.step()  # call lr scheduler on a step-by-step basis instead of epoch

    if return_stats:
        dist.all_reduce(fsdp_metrics, op=dist.ReduceOp.SUM)
        train_loss = fsdp_metrics[0] / fsdp_metrics[2]
        train_perplexity = fsdp_metrics[1] / fsdp_metrics[2]
        train_accuracy = 100 * fsdp_metrics[3] / fsdp_metrics[4]

        lr = optimizer.param_groups[0]["lr"]
        return {
            "loss": train_loss.item(),
            "accuracy": train_accuracy.item(),
            "perplexity": train_perplexity.item(),
            "learning_rate": lr,
        }
    else:
        return None


def run_evaluation_steps(model, rank, world_size, eval_loader):
    """Run M evaluation iterations"""
    model.eval()  # set model in evaluation mode

    local_rank = int(os.environ["LOCAL_RANK"])

    fsdp_metrics = torch.zeros(5).to(local_rank)

    inner_pbar = None
    if rank == 0:
        inner_pbar = tqdm.tqdm(range(cfg.eval_iters), colour="green", desc="Evaluation iterations")

    with torch.no_grad():
        for x, y, attn_mask, loss_mask in itertools.islice(eval_loader, cfg.eval_iters):
            x, y, loss_mask = (
                x.to(local_rank, non_blocking=True),
                y.to(local_rank, non_blocking=True),
                loss_mask.to(local_rank, non_blocking=True),
            )
            # use pre-computed attention mask to handle padded sequence,
            # this is very slow (2-3x slower), also requires huge amount of GPU RAM...
            if attn_mask is not None:
                attn_mask = attn_mask.to(local_rank, non_blocking=True)

            output = model(x, attn_mask)

            loss = compute_finetune_loss(output, y, loss_mask)

            num_acc, num_samples = compute_metrics(output, y, loss_mask)
            fsdp_metrics[0] += loss.item()  # sum up batch loss
            fsdp_metrics[1] += np.exp(loss.item())  # sum up perplexity
            fsdp_metrics[2] += 1  # increase number of batches
            fsdp_metrics[3] += num_acc  # sum up number of accurate prediction tokens
            fsdp_metrics[4] += num_samples  # sum up number of tokens

            if inner_pbar is not None:
                inner_pbar.update(1)

    dist.all_reduce(fsdp_metrics, op=dist.ReduceOp.SUM)
    eval_loss = fsdp_metrics[0] / fsdp_metrics[2]
    eval_perplexity = fsdp_metrics[1] / fsdp_metrics[2]
    eval_accuracy = 100 * fsdp_metrics[3] / fsdp_metrics[4]

    if inner_pbar is not None:
        inner_pbar.close()

    model.train()  # set model in training mode after evaluation runs

    return {"loss": eval_loss.item(), "accuracy": eval_accuracy.item(), 'perplexity': eval_perplexity.item()}


def rank0_logger(msg, rank):
    if rank == 0:
        print(msg)


def _collate_fn(batch):
    """
    Custom collate function to pad the sequence to maximum length in the batch,
    which is much faster than pad the sequence to some global max sequence length.

    In addition, it will compute the attention mask and loss mask for the batch.
    """

    batch_size = len(batch)
    if batch_size == 1:  # not need to pad, just process the samples
        prompt, completion = batch[0]
        completion_len = len(completion)
        assert completion_len < cfg.max_seq_length

        sequence = torch.concat((prompt, completion), dim=0)
        seq_length = len(sequence)

        if seq_length > cfg.max_seq_length:
            sequence = sequence[-(cfg.max_seq_length) :]

        x = sequence[:-1]  # [seq_length - 1], exclude the last token for x, so x, y pair have the same length
        y = sequence[1:]  # [seq_length - 1],  shift target one step to the right

        # create a loss mask for the positions for the prompts, completion
        # where -1s are prompt tokens, 1s are completion tokens
        loss_mask = torch.ones_like(y).to(dtype=torch.int)
        loss_mask[:-completion_len] = -1  # prompt tokens

        # add batch dimension
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        loss_mask = loss_mask.unsqueeze(0)

        # the SDPA module with casual attention is much faster if we don't pre-compute the mask
        attn_mask = None

        return x, y, attn_mask, loss_mask

    batch_seq_lengths = [len(item[0]) + len(item[1]) for item in batch]

    max_batch_seq_length = min(cfg.max_seq_length, max(batch_seq_lengths))

    # concatenate prompt, completion together
    batch_sequences = torch.full((batch_size, max_batch_seq_length), cfg.pad_id, dtype=torch.long)

    # where -1s are prompt tokens, 1s are completion tokens, and 0s are padding tokens
    mask = torch.full((batch_size, max_batch_seq_length), 0, dtype=torch.long)

    for i, (prompt, completion) in enumerate(batch):
        # need prompt, completion lengths to compute loss mask
        prompt_len, completion_len = len(prompt), len(completion)

        assert completion_len < max_batch_seq_length

        if prompt_len + completion_len > max_batch_seq_length:
            chunk_size = max_batch_seq_length - completion_len
            prompt = prompt[-chunk_size:]
            prompt_len = len(prompt)

        seq = torch.concat((prompt, completion), dim=0)

        seq_len = len(seq)

        assert seq_len > 0 and seq_len <= max_batch_seq_length

        if seq_len < max_batch_seq_length:
            if cfg.pad_left:
                # example: [1, 2, 3] -> [0, 0, 1, 2, 3]
                batch_sequences[i, -seq_len:] = seq

                # example: [1, 2, 3] (where 1, 2 are prompt, and 3 is completion) -> [0, 0, -1, -1, 1]
                mask[i, -(prompt_len + completion_len) : -completion_len] = -1
                mask[i, -completion_len:] = 1
            else:
                # example: [1, 2, 3] -> [1, 2, 3, 0, 0]
                batch_sequences[i, :seq_len] = seq

                # example: [1, 2, 3] (where 1, 2 are prompt, and 3 is completion) -> [-1, -1, 1, 0, 0]
                mask[i, :prompt_len] = -1
                mask[i, prompt_len : prompt_len + completion_len] = 1
        else:
            batch_sequences[i, :] = seq

            # example: [1, 2, 3, 4, 5] (where 1, 2, 3 are prompt, and 4, 5 are completion) -> [-1, -1, -1, 1, 1]
            mask[i, :prompt_len] = -1
            mask[i, prompt_len:] = 1

    x = batch_sequences[:, :-1]  # [batch_size, max_batch_seq_length - 1]
    y = batch_sequences[:, 1:]  # [batch_size, max_batch_seq_length - 1]

    # shift to right to align with y
    loss_mask = torch.clone(mask)[:, 1:]

    # create attention mask
    # BUG in SDPA module when use -inf or bool mask will cause NaNs
    attn_mask = torch.full((batch_size, 1, max_batch_seq_length - 1, max_batch_seq_length - 1), float(-1e10))

    for i, seq_len in enumerate(batch_seq_lengths):
        # casual mask looks like:
        # [0, -inf, -inf]
        # [0, 0, -inf]
        # [0, 0, 0]
        casual_mask = torch.triu(attn_mask[i, :, :seq_len, :seq_len], diagonal=1)
        if cfg.pad_left:
            attn_mask[i, :, -seq_len:, -seq_len:] = casual_mask
        else:
            attn_mask[i, :, :seq_len, :seq_len] = casual_mask

    return x, y, attn_mask, loss_mask


def fsdp_main():
    assert cfg.micro_batch_size >= 1
    assert cfg.gradient_accum_steps >= 1
    assert cfg.log_interval >= 10

    if not os.path.exists(cfg.pretrain_ckpt_file):
        raise ValueError(f'Invalid pretrained checkpoint "{cfg.pretrain_ckpt_file}", aborting...')

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    setup()

    logger = functools.partial(rank0_logger, rank=rank)

    # --------------- Load datasets ---------------

    logger("\nLoading datasets ...")

    train_dataset = FineTuneDataset(data_sources=cfg.train_datasources)

    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)

    cuda_kwargs = {
        "collate_fn": _collate_fn,
        "num_workers": cfg.dataloader_workers,
        "pin_memory": True,
        "shuffle": False,
    }

    train_kwargs = {"batch_size": cfg.micro_batch_size, "sampler": train_sampler}

    train_kwargs.update(cuda_kwargs)
    train_loader = DataLoader(train_dataset, **train_kwargs)

    logger(f"--> Train dataset metadata:\n{train_dataset.get_metadata()}")

    # create evaluation dataset on demand
    eval_loader = None
    if cfg.eval_interval > 0:
        eval_dataset = FineTuneDataset(data_sources=cfg.eval_datasources)

        eval_kwargs = {"batch_size": cfg.micro_batch_size}
        eval_kwargs.update(cuda_kwargs)

        eval_loader = DataLoader(eval_dataset, **eval_kwargs)

        logger(f"--> Evaluation dataset metadata:\n{eval_dataset.get_metadata()}")

    # --------------- Setup model and optimizer ---------------

    logger("\nInitialize model and FSDP ...")

    torch.cuda.set_device(local_rank)
    clear_gpu_cache(local_rank)

    model = GPT2LMHeadModel(
        model_type=cfg.model_type,
        embed_dropout=cfg.embed_dropout,
        attn_dropout=cfg.attn_dropout,
        resid_dropout=cfg.resid_dropout,
    )
    
    logger(f"--> model metadata:\n{model.get_metadata()}")
    logger(f"--> number of parameters: {model.get_num_params() / 1e6:.2f} million")

    # Load model checkpoint before passing into FSDP
    if os.path.exists(cfg.pretrain_ckpt_file):
        if cfg.checkpoint_type == StateDictType.FULL_STATE_DICT:
            load_full_state_model_checkpoint(model, rank, cfg.pretrain_ckpt_file)
        elif cfg.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
            pass

    scaler = None
    mixed_precision_policy = None  # defaults to fp32

    if cfg.mixed_precision:
        bf16_ready = torch.version.cuda and torch.cuda.is_bf16_supported() and dist.is_nccl_available()
        if bf16_ready:
            logger(f"--> bFloat16 enabled for mixed precision ...")
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                # Gradient communication precision.
                reduce_dtype=torch.bfloat16,
                # Buffer precision.
                buffer_dtype=torch.bfloat16,
            )
        else:
            logger(f"--> float16 enabled for mixed precision ...")
            # requires grad scaler
            scaler = ShardedGradScaler()
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.float16,
                # Gradient communication precision.
                reduce_dtype=torch.float16,
                # Buffer precision.
                buffer_dtype=torch.float16,
            )
    else:
        logger(f"--> fallback to float32 ...")

    gpt2_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={EncoderBlock})

    model = FSDP(
        model,
        auto_wrap_policy=gpt2_auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        backward_prefetch=cfg.backward_prefetch,
        forward_prefetch=cfg.forward_prefetch,
        cpu_offload=CPUOffload(offload_params=False),
        sharding_strategy=cfg.sharding_strategy,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=cfg.limit_all_gathers,
        use_orig_params=True,  # need this since we're only use weight decay for some of the params
    )

    if cfg.fsdp_activation_checkpointing:
        logger("--> applying FSDP activation checkpointing ...")
        apply_fsdp_checkpointing(model)

    logger(f"--> FSDP model:\n{model}")

    if cfg.compile_model:
        logger(f"--> compile model using torch.compile() ...")
        model = torch.compile(model)

    logger("\nInitialize optimizer ...")

    optimizer = create_optimizer(
        model=model,
        lr=cfg.init_lr,
        eps=cfg.adamw_eps,
        weight_decay=cfg.weight_decay,
        betas=cfg.adamw_betas,
        fused=cfg.adamw_fused,
    )

    scheduler = CosineDecayWithWarmupLRScheduler(
        optimizer=optimizer,
        min_lr=cfg.min_lr,
        max_lr=cfg.max_lr,
        warmup_steps=cfg.warmup_steps,
        max_decay_steps=cfg.max_decay_steps,
    )

    # a lazy way to initialize scheduler when resume training, as FSDP does not support save and load scheduler state yet
    if cfg.start_from_iter > 0:
        for _ in range(0, cfg.start_from_iter):
            scheduler.step()

        if rank == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            logger(f'Current learning rate is: {current_lr:.6f}')

    # --------------- Start Training ---------------

    logger(f"\nStarting to run {cfg.max_train_iters - cfg.start_from_iter} training iterations ...")

    torch_profiler = None
    # Careful as the logs will grow very fast
    if cfg.use_profiler:
        torch_profiler = create_trace_profiler(os.path.join(cfg.log_dir, "profile_traces"))

    tb_writer = None
    memmax = None
    mem_alloc_tracker = None
    inner_pbar = None
    train_stats = eval_stats = None

    if rank == 0:
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.ckpt_dir, exist_ok=True)

        if cfg.use_tensorboard:
            tb_writer = SummaryWriter(os.path.join(cfg.log_dir, model.model_name))

        if cfg.track_gpu_mem_usage:
            memmax = Memory_Maximizer()
            mem_alloc_tracker = []
            memmax.start()

        inner_pbar = tqdm.tqdm(range(cfg.max_train_iters), colour="blue", desc="Training iterations")

    model.train()
    for iter in range(cfg.start_from_iter + 1, cfg.max_train_iters + 1):
        train_stats = run_single_train_step(
            model=model,
            rank=rank,
            world_size=world_size,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            return_stats=iter % cfg.log_interval == 0 or iter == cfg.max_train_iters,
        )

        if inner_pbar is not None:
            inner_pbar.update(1)

        if torch_profiler is not None:
            torch_profiler.step()

        # logging
        if train_stats is not None and rank == 0:
            logger(
                f'Training iteration {iter}: train loss: {train_stats["loss"]:.4f}, '
                f'train accuracy: {train_stats["accuracy"]:.2f}%, train perplexity: {train_stats["perplexity"]:.2f}, learning rate: {train_stats["learning_rate"]:.10f}'
            )

            if tb_writer is not None:
                tb_writer.add_scalar("train/loss", train_stats["loss"], iter)
                tb_writer.add_scalar("train/accuracy", train_stats["accuracy"], iter)
                tb_writer.add_scalar("train/perplexity", train_stats["perplexity"], iter)
                tb_writer.add_scalar("train/learning_rate", train_stats["learning_rate"], iter)

            if cfg.track_gpu_mem_usage:
                memmax.update()
                mem_alloc_tracker.append(format_to_gb(torch.cuda.memory_allocated()))

        # checkpointing
        if cfg.ckpt_interval > 0 and iter % cfg.ckpt_interval == 0 or iter == cfg.max_train_iters:
            # save model state
            if cfg.checkpoint_type == StateDictType.FULL_STATE_DICT:
                save_full_state_model_checkpoint(
                    model, rank, os.path.join(cfg.ckpt_dir, f"model_{model.model_name}-iter-{iter}.pt")
                )
            elif cfg.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                pass

            # save optimizer state separately to save resource incase model is too big
            if cfg.save_optimizer:
                if cfg.checkpoint_type == StateDictType.FULL_STATE_DICT:
                    save_full_state_optimizer_checkpoint(
                        model,
                        optimizer,
                        rank,
                        os.path.join(cfg.ckpt_dir, f"optim_{model.model_name}-iter-{iter}.pt"),
                    )
                elif cfg.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                    pass

        # evaluation steps
        if cfg.eval_iters > 0 and (cfg.eval_interval > 0 and iter % cfg.eval_interval == 0 or iter == cfg.max_train_iters):
            eval_stats = run_evaluation_steps(model=model, rank=rank, world_size=world_size, eval_loader=eval_loader)

            if rank == 0:
                logger(
                    f'Training iteration {iter}: evaluation loss: {eval_stats["loss"]:.4f}, '
                    f'evaluation accuracy: {eval_stats["accuracy"]:.2f}%, evaluation perplexity: {eval_stats["perplexity"]:.2f}'
                )

                if tb_writer is not None:
                    tb_writer.add_scalar("eval/loss", eval_stats["loss"], iter)
                    tb_writer.add_scalar("eval/accuracy", eval_stats["accuracy"], iter)
                    tb_writer.add_scalar("eval/perplexity", eval_stats["perplexity"], iter)

    if rank == 0:
        # training is done...show some training stats.
        if cfg.track_gpu_mem_usage:
            memmax.stop()
            logger(f"Total memory allocated: {mem_alloc_tracker}")
            logger(f"CUDA Memory Summary After Last training:\n{torch.cuda.memory_summary()}")

    # all done, set barrier to ensure all GPU's complete, and then cleanup
    dist.barrier()
    cleanup()


if __name__ == "__main__":
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # torch.set_float32_matmul_precision("high")

    # torch.backends.cuda.enable_flash_sdp(True)
    # torch.backends.cuda.enable_mem_efficient_sdp(True)
    # torch.backends.cuda.enable_math_sdp(True)

    fsdp_main()
