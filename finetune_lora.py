import os
import itertools
import functools
from typing import Tuple
import tqdm
import random
import numpy as np
from contextlib import nullcontext

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

from configs.finetune_lora import config as cfg

from utils import (
    CosineDecayWithWarmupLRScheduler,
    Memory_Maximizer,
    save_lora_model_checkpoint,
    format_to_gb,
)


from finetune import _collate_fn, compute_finetune_loss, compute_metrics, create_trace_profiler

from models.lora import lora, lora_state_dict, mark_only_lora_as_trainable


def setup():
    # initialize the process group
    dist.init_process_group('nccl')


def cleanup():
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    print(f'clearing cache for rank {rank}')
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
    check_fn = lambda submodule: isinstance(submodule, EncoderBlock)  # noqa: E731

    apply_activation_checkpointing(model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)


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

    local_rank = int(os.environ['LOCAL_RANK'])

    if return_stats:
        metrics = torch.zeros(5).to(local_rank)

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
            metrics[0] += loss.item()  # sum up batch loss
            metrics[1] += np.exp(loss.item())  # sum up perplexity
            metrics[2] += 1  # increase number of batches
            metrics[3] += num_acc  # sum up number of accurate prediction tokens
            metrics[4] += num_samples  # sum up number of tokens

    if scaler is not None:  # when using float16
        if cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)  # unscale before clip gradients
            #  FSDP needs to use this method to clip gradient norm instead of torch.nn.utils.clip_grad_norm_
            model.clip_grad_norm_(cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()  # adjust scaling for next batch
    else:
        if cfg.grad_clip != 0.0:
            model.clip_grad_norm_(cfg.grad_clip)

        optimizer.step()

    scheduler.step()  # call lr scheduler on a step-by-step basis instead of epoch

    if return_stats:
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        train_loss = metrics[0] / metrics[2]
        train_perplexity = metrics[1] / metrics[2]
        train_accuracy = 100 * metrics[3] / metrics[4]

        lr = optimizer.param_groups[0]['lr']
        return {
            'loss': train_loss.item(),
            'accuracy': train_accuracy.item(),
            'perplexity': train_perplexity.item(),
            'learning_rate': lr,
        }
    else:
        return None


def run_evaluation_steps(model, rank, world_size, eval_loader):
    """Run M evaluation iterations"""
    model.eval()  # set model in evaluation mode

    local_rank = int(os.environ['LOCAL_RANK'])

    metrics = torch.zeros(5).to(local_rank)

    inner_pbar = None
    if rank == 0:
        inner_pbar = tqdm.tqdm(range(cfg.eval_iters), colour='green', desc='Evaluation iterations')

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
            metrics[0] += loss.item()  # sum up batch loss
            metrics[1] += np.exp(loss.item())  # sum up perplexity
            metrics[2] += 1  # increase number of batches
            metrics[3] += num_acc  # sum up number of accurate prediction tokens
            metrics[4] += num_samples  # sum up number of tokens

            if inner_pbar is not None:
                inner_pbar.update(1)

    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    eval_loss = metrics[0] / metrics[2]
    eval_perplexity = metrics[1] / metrics[2]
    eval_accuracy = 100 * metrics[3] / metrics[4]

    if inner_pbar is not None:
        inner_pbar.close()

    model.train()  # set model in training mode after evaluation runs

    return {'loss': eval_loss.item(), 'accuracy': eval_accuracy.item(), 'perplexity': eval_perplexity.item()}


def rank0_logger(msg, rank):
    if rank == 0:
        print(msg)


def fsdp_main():
    assert cfg.micro_batch_size >= 1
    assert cfg.gradient_accum_steps >= 1
    assert cfg.log_interval >= 10

    if not os.path.exists(cfg.pretrain_ckpt_file):
        raise ValueError(f'Invalid pretrained checkpoint "{cfg.pretrain_ckpt_file}", aborting...')

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    setup()

    logger = functools.partial(rank0_logger, rank=rank)

    # --------------- Load datasets ---------------

    logger('\nLoading datasets ...')

    train_dataset = FineTuneDataset(data_sources=cfg.train_datasources)

    train_sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)

    cuda_kwargs = {
        'collate_fn': _collate_fn,
        'num_workers': cfg.dataloader_workers,
        'batch_size': cfg.micro_batch_size,
        'pin_memory': True,
        'shuffle': False,
    }

    train_kwargs = {'sampler': train_sampler}
    train_kwargs.update(cuda_kwargs)
    train_loader = DataLoader(train_dataset, **train_kwargs)
    logger(f'--> Train dataset metadata:\n{train_dataset.get_metadata()}')

    # create evaluation dataset on demand
    eval_loader = None
    if cfg.eval_interval > 0:
        eval_dataset = FineTuneDataset(data_sources=cfg.eval_datasources)
        eval_loader = DataLoader(eval_dataset, **cuda_kwargs)
        logger(f'--> Evaluation dataset metadata:\n{eval_dataset.get_metadata()}')

    # --------------- Setup model and optimizer ---------------

    logger('\nInitialize model and FSDP ...')

    torch.cuda.set_device(local_rank)
    clear_gpu_cache(local_rank)

    with lora(r=cfg.lora_r, alpha=cfg.lora_alpha, dropout=cfg.lora_dropout, enabled=True):
        model = GPT2LMHeadModel(
            model_type=cfg.model_type,
            embed_dropout=cfg.embed_dropout,
            attn_dropout=cfg.attn_dropout,
            resid_dropout=cfg.resid_dropout,
        )

        logger(f'--> model metadata:\n{model.get_metadata()}')
        logger(f'--> number of parameters: {model.get_num_params() / 1e6:.2f} million')

        # Load model checkpoint before passing into FSDP
        if os.path.exists(cfg.pretrain_ckpt_file):
            logger(f'--> load pretrained checkpoint {cfg.pretrain_ckpt_file}')
            model_state = torch.load(cfg.pretrain_ckpt_file)
            # strict=False because missing keys due to LoRA weights not contained in checkpoint state
            model.load_state_dict(model_state, strict=False)
            del model_state

    mark_only_lora_as_trainable(model, bias=cfg.train_bias)

    scaler = None
    mixed_precision_policy = None  # defaults to fp32

    if cfg.mixed_precision:
        bf16_ready = torch.version.cuda and torch.cuda.is_bf16_supported() and dist.is_nccl_available()
        if bf16_ready:
            logger('--> bFloat16 enabled for mixed precision ...')
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                # Gradient communication precision.
                reduce_dtype=torch.bfloat16,
                # Buffer precision.
                buffer_dtype=torch.bfloat16,
            )
        else:
            logger('--> float16 enabled for mixed precision ...')
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
        logger('--> fallback to float32 ...')

    _auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={EncoderBlock})

    model = FSDP(
        model,
        auto_wrap_policy=_auto_wrap_policy,
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
        logger('--> applying FSDP activation checkpointing ...')
        apply_fsdp_checkpointing(model)

    logger(f'--> FSDP model:\n{model}')

    if cfg.compile_model:
        logger('--> compile model using torch.compile() ...')
        model = torch.compile(model)

    logger('\nInitialize optimizer ...')

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

    # --------------- Start Training ---------------

    logger(f'\nStarting to run {cfg.max_train_iters} training iterations ...')

    torch_profiler = None
    # Careful as the logs will grow very fast
    if cfg.use_profiler:
        torch_profiler = create_trace_profiler(os.path.join(cfg.log_dir, 'profile_traces'))

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

        inner_pbar = tqdm.tqdm(range(cfg.max_train_iters), colour='blue', desc='Training iterations')

    model.train()
    for iter in range(1, cfg.max_train_iters + 1):
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
                tb_writer.add_scalar('train/loss', train_stats['loss'], iter)
                tb_writer.add_scalar('train/accuracy', train_stats['accuracy'], iter)
                tb_writer.add_scalar('train/perplexity', train_stats['perplexity'], iter)
                tb_writer.add_scalar('train/learning_rate', train_stats['learning_rate'], iter)

            if cfg.track_gpu_mem_usage:
                memmax.update()
                mem_alloc_tracker.append(format_to_gb(torch.cuda.memory_allocated()))

        # checkpointing
        if cfg.ckpt_interval > 0 and iter % cfg.ckpt_interval == 0 or iter == cfg.max_train_iters:
            # save model state
            save_lora_model_checkpoint(
                model=model,
                rank=rank,
                ckpt_save_path=os.path.join(cfg.ckpt_dir, f'lora_model_{model.model_name}-iter-{iter}.pt'),
                bias=cfg.train_bias,
            )

        # evaluation steps
        if cfg.eval_iters > 0 and (cfg.eval_interval > 0 and iter % cfg.eval_interval == 0 or iter == cfg.max_train_iters):
            eval_stats = run_evaluation_steps(model=model, rank=rank, world_size=world_size, eval_loader=eval_loader)

            if rank == 0:
                logger(
                    f'Training iteration {iter}: evaluation loss: {eval_stats["loss"]:.4f}, '
                    f'evaluation accuracy: {eval_stats["accuracy"]:.2f}%, evaluation perplexity: {eval_stats["perplexity"]:.2f}'
                )

                if tb_writer is not None:
                    tb_writer.add_scalar('eval/loss', eval_stats['loss'], iter)
                    tb_writer.add_scalar('eval/accuracy', eval_stats['accuracy'], iter)
                    tb_writer.add_scalar('eval/perplexity', eval_stats['perplexity'], iter)

    if rank == 0:
        # training is done...show some training stats.
        if cfg.track_gpu_mem_usage:
            memmax.stop()
            logger(f'Total memory allocated: {mem_alloc_tracker}')
            logger(f'CUDA Memory Summary After Last training:\n{torch.cuda.memory_summary()}')

    # all done, set barrier to ensure all GPU's complete, and then cleanup
    dist.barrier()
    cleanup()


if __name__ == '__main__':
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    torch.set_float32_matmul_precision('high')

    # torch.backends.cuda.enable_flash_sdp(True)
    # torch.backends.cuda.enable_mem_efficient_sdp(True)
    # torch.backends.cuda.enable_math_sdp(True)

    fsdp_main()
