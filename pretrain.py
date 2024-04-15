import os
import functools
from typing import Tuple, Dict
import tqdm
import random
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist

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

# import custom objects
from models import GPT2LMHeadModel, EncoderBlock, create_optimizer
from custom_dataset import BlendedDataset

from configs.pretrain import config as cfg

from utils import (
    CosineDecayWithWarmupLRScheduler,
    Memory_Maximizer,
    StatsTracker,
    load_full_state_model_checkpoint,
    load_full_state_optimizer_checkpoint,
    save_full_state_model_checkpoint,
    save_full_state_optimizer_checkpoint,
    format_to_gb,
    create_logger,
)


# helper functions for FSDP
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


def compute_pre_train_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    assert len(logits.shape) == 3  # [B, max_seq_length, vocab_size]
    assert len(targets.shape) == 2  # [B, max_seq_length]
    assert logits.shape[0] == targets.shape[0]

    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


@torch.no_grad()
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[int, float]:
    assert len(logits.shape) == 3  # [B, max_seq_length, vocab_size]
    assert len(targets.shape) == 2  # [B, max_seq_length]
    assert logits.shape[0] == targets.shape[0]

    # get the index of the max log-probability
    pred = torch.softmax(logits, dim=-1).argmax(dim=-1)

    num_accurate = pred.eq(targets.view_as(pred)).sum().item()
    num_samples = targets.shape[0] * targets.shape[1]

    return num_accurate, num_samples


def train_step(
    model: GPT2LMHeadModel,
    batch: Tuple[torch.Tensor],
    scaler: ShardedGradScaler,
    gradient_accum_steps: int,
    local_rank: int,
    tracker: StatsTracker,
) -> None:
    """Run a single training step, where we do a forward + backward passes, but do no update parameters"""

    assert gradient_accum_steps >= 1

    x, y = batch

    x, y = x.to(local_rank, non_blocking=True), y.to(local_rank, non_blocking=True)

    output = model(x)
    loss = compute_pre_train_loss(output, y)
    # scale the loss to account for gradient accumulation
    scaled_loss = loss / cfg.gradient_accum_steps

    if scaler is not None:  # when using float16
        scaler.scale(scaled_loss).backward()
    else:
        scaled_loss.backward()

    num_acc, num_samples = compute_metrics(output.detach(), y.detach())
    tracker.update(loss.detach(), num_acc, num_samples)


def update_step(
    model: GPT2LMHeadModel,
    optimizer: torch.optim.AdamW,
    scheduler: CosineDecayWithWarmupLRScheduler,
    grad_clip: float,
    scaler: ShardedGradScaler = None,
) -> None:
    """Run a single parameter update step"""
    if grad_clip > 0.0:
        if scaler is not None:  # when using float16
            scaler.unscale_(optimizer)  # unscale before clip gradients

        # FSDP needs to use this method to clip gradient norm instead of torch.nn.utils.clip_grad_norm_
        model.clip_grad_norm_(grad_clip)

    if scaler is not None:  # when using float16
        scaler.step(optimizer)
        scaler.update()  # adjust scaling for next batch
    else:
        optimizer.step()

    # prepare for next update
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()  # call lr scheduler on a step-by-step basis instead of epoch


@torch.no_grad()
def run_evaluation_steps(
    model: GPT2LMHeadModel,
    loader: DataLoader,
    steps: int,
    local_rank: int,
    tracker: StatsTracker,
) -> None:
    """Run M evaluation steps"""
    model.eval()  # set model in evaluation mode

    inner_pbar = None
    if local_rank == 0:
        inner_pbar = tqdm.tqdm(range(cfg.eval_iters), colour='green', desc='Evaluation steps')

    for i, (x, y) in enumerate(loader):
        x, y = x.to(local_rank, non_blocking=True), y.to(local_rank, non_blocking=True)

        output = model(x)
        loss = compute_pre_train_loss(output, y)

        num_acc, num_samples = compute_metrics(output.detach(), y.detach())
        tracker.update(loss.detach(), num_acc, num_samples)

        if inner_pbar is not None:
            inner_pbar.update(1)

        if i + 1 >= steps:
            break

    if inner_pbar is not None:
        inner_pbar.close()


def fsdp_main():  # noqa: C901
    assert cfg.start_from_iter >= 0
    assert cfg.train_batch_size >= 1
    assert cfg.gradient_accum_steps >= 1
    assert cfg.log_interval >= 10

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    setup()

    logger = create_logger(rank=rank)

    # --------------- Load datasets ---------------

    logger.info('Loading datasets ...')
    train_dataset = BlendedDataset(
        data_sources=cfg.train_datasources,
        max_seq_length=cfg.max_seq_length,
        rank=rank,
        world_size=world_size,  # shard the dataset
        seed=int(cfg.seed + rank),
    )

    # Our custom IterableDatasets already have sharding and shuffle mechanism implemented
    cuda_kwargs = {
        'num_workers': cfg.dataloader_workers,
        'batch_size': cfg.train_batch_size,
        'pin_memory': True,
        'shuffle': False,
        'sampler': None,
    }
    train_loader = DataLoader(train_dataset, **cuda_kwargs)
    logger.info(f'--> Train dataset metadata:\n{train_dataset.get_metadata()}')

    # create evaluation dataset on demand
    eval_loader = None
    if cfg.eval_interval > 0:
        eval_dataset = BlendedDataset(
            data_sources=cfg.eval_datasources,
            max_seq_length=cfg.max_seq_length,
            rank=rank,
            world_size=world_size,  # shard the dataset
            seed=int(cfg.seed + rank),
        )
        eval_loader = DataLoader(eval_dataset, **cuda_kwargs)
        logger.info(f'--> Evaluation dataset metadata:\n{eval_dataset.get_metadata()}')

    batch_size = int(cfg.train_batch_size * cfg.gradient_accum_steps)
    steps_per_epoch = len(train_loader) // cfg.gradient_accum_steps
    max_train_steps = steps_per_epoch * cfg.num_epochs

    # --------------- Setup model and optimizer ---------------

    logger.info('Initialize model and FSDP ...')

    torch.cuda.set_device(local_rank)
    clear_gpu_cache(local_rank)

    model = GPT2LMHeadModel(
        model_type=cfg.model_type,
        embed_dropout=cfg.embed_dropout,
        attn_dropout=cfg.attn_dropout,
        resid_dropout=cfg.resid_dropout,
    )

    logger.info(f'--> model metadata:\n{model.get_metadata()}')
    logger.info(f'--> number of parameters: {model.get_num_params() / 1e6:.2f} million')

    # Load model checkpoint before passing into FSDP
    if cfg.load_model_ckpt and os.path.exists(cfg.load_model_ckpt):
        if cfg.checkpoint_type == StateDictType.FULL_STATE_DICT:
            load_full_state_model_checkpoint(model, rank, cfg.load_model_ckpt)
        elif cfg.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
            pass

    scaler = None
    mixed_precision_policy = None  # defaults to fp32

    if cfg.mixed_precision:
        bf16_ready = torch.version.cuda and torch.cuda.is_bf16_supported() and dist.is_nccl_available()
        if bf16_ready:
            logger.info('--> bFloat16 enabled for mixed precision ...')
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                # Gradient communication precision.
                reduce_dtype=torch.bfloat16,
                # Buffer precision.
                buffer_dtype=torch.bfloat16,
            )
        else:
            logger.info('--> float16 enabled for mixed precision ...')
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
        logger.info('--> fallback to float32 ...')

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
        logger.info('--> applying FSDP activation checkpointing ...')
        apply_fsdp_checkpointing(model)

    logger.info(f'--> FSDP model:\n{model}')

    if cfg.compile_model:
        logger.info('--> compile model using torch.compile() ...')
        model = torch.compile(model)

    logger.info('Initialize optimizer ...')

    optimizer = create_optimizer(
        model=model,
        lr=cfg.init_lr,
        eps=cfg.adamw_eps,
        weight_decay=cfg.weight_decay,
        betas=cfg.adamw_betas,
        fused=cfg.adamw_fused,
    )

    # Load optimizer checkpoint
    if cfg.load_optim_ckpt and os.path.exists(cfg.load_optim_ckpt):
        if cfg.checkpoint_type == StateDictType.FULL_STATE_DICT:
            load_full_state_optimizer_checkpoint(model, optimizer, rank, cfg.load_model_ckpt)
        elif cfg.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
            pass

    scheduler = CosineDecayWithWarmupLRScheduler(
        optimizer=optimizer,
        min_lr=cfg.min_lr,
        max_lr=cfg.max_lr,
        warmup_steps=cfg.warmup_steps,
        max_decay_steps=max_train_steps,
    )

    # a lazy way to initialize scheduler when resume training, as FSDP does not support save and load scheduler state yet
    if cfg.start_from_iter > 0:
        for _ in range(0, cfg.start_from_iter):
            scheduler.step()

        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f'Current learning rate is: {current_lr:.6f}')

    # --------------- Start Training ---------------

    logger.info(f'Starting to run {max_train_steps - cfg.start_from_iter} training steps ...')

    torch_profiler = None
    # Careful as the logs will grow very fast
    if cfg.use_profiler:
        torch_profiler = create_trace_profiler(os.path.join(cfg.log_dir, 'profile_traces'))

    tb_writer = None
    memmax = None
    mem_alloc_tracker = None
    inner_pbar = None
    train_stats = None
    eval_stats = None
    train_steps = 0

    if rank == 0:
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.ckpt_dir, exist_ok=True)

        if cfg.use_tensorboard:
            tb_writer = SummaryWriter(os.path.join(cfg.log_dir, model.model_name))

        if cfg.track_gpu_mem_usage:
            memmax = Memory_Maximizer()
            mem_alloc_tracker = []
            memmax.start()

        inner_pbar = tqdm.tqdm(range(max_train_steps), colour='blue', desc='Training steps')

    train_tracker = StatsTracker(True, local_rank)
    val_tracker = StatsTracker(True, local_rank)

    for epoch in range(1, cfg.num_epochs + 1):  # for each epoch
        logger.info(f'Start epoch {epoch}')
        model.train()
        train_tracker.reset()
        val_tracker.reset()

        for iter, batch in enumerate(train_loader):  # for each batch in current epoch
            train_step(model, batch, scaler, cfg.gradient_accum_steps, local_rank, train_tracker)

            if iter % cfg.gradient_accum_steps == 0:
                update_step(model, optimizer, scheduler, cfg.grad_clip, scaler)
                inner_pbar.update(1)
                train_steps += 1

                if torch_profiler is not None:
                    torch_profiler.step()

                # logging training statistics
                if train_steps % cfg.log_interval == 0:
                    train_stats = train_tracker.get_dict()
                    train_stats['learning_rate'] = optimizer.param_groups[0]['lr']
                    log_statistics(tb_writer, train_steps, train_stats, True)
                    train_tracker.reset()

                    if cfg.track_gpu_mem_usage:
                        memmax.update()
                        mem_alloc_tracker.append(format_to_gb(torch.cuda.memory_allocated()))

                # checkpointing
                if cfg.ckpt_interval > 0 and iter % cfg.ckpt_interval == 0 or iter == max_train_steps:
                    # save model state
                    if cfg.checkpoint_type == StateDictType.FULL_STATE_DICT:
                        save_full_state_model_checkpoint(
                            model, rank, os.path.join(cfg.ckpt_dir, f'model_{model.model_name}-iter-{iter}.pt')
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
                            os.path.join(cfg.ckpt_dir, f'optim_{model.model_name}-iter-{iter}.pt'),
                        )
                    elif cfg.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                        pass

                # evaluation steps
                if cfg.eval_iters > 0 and (cfg.eval_interval > 0 and iter % cfg.eval_interval == 0 or iter == max_train_steps):
                    val_tracker.reset()
                    model.eval()
                    run_evaluation_steps(model, eval_loader, cfg.eval_iters, local_rank, val_tracker)
                    model.train()
                    eval_stats = val_tracker.get_dict()
                    log_statistics(tb_writer, train_steps, eval_stats, False)

    if rank == 0:
        # training is done...show some training stats.
        if cfg.track_gpu_mem_usage:
            memmax.stop()
            logger.info(f'Total memory allocated: {mem_alloc_tracker}')
            logger.info(f'CUDA Memory Summary After Last training:\n{torch.cuda.memory_summary()}')

    # all done, set barrier to ensure all GPU's complete, and then cleanup
    dist.barrier()
    cleanup()


def log_statistics(tb_writer: SummaryWriter, train_steps: int, stats: Dict, is_training: bool) -> None:
    if tb_writer is not None:
        tb_tag = 'train' if is_training else 'val'
        for k, v in stats.items():
            tb_writer.add_scalar(f'{tb_tag}/{k}', v, train_steps)


if __name__ == '__main__':
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    torch.set_float32_matmul_precision('high')

    torch.backends.cuda.enable_flash_sdp(True)
    # torch.backends.cuda.enable_mem_efficient_sdp(True)
    # torch.backends.cuda.enable_math_sdp(False)

    fsdp_main()
