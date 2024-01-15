from typing import Tuple
from dataclasses import dataclass

from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

from custom_dataset import DataSource


@dataclass
class config:
    """Pre-training configurations, where we use blended data sources, larger learning rates"""

    # model definition, the details (number of layers, heads etc.) are defined in models/gpt2_model.py --> gpt2_meta
    model_type: str = 'gpt2-xl'  # gpt2, gpt2-medium, gpt2-large, gpt2-xl

    # datasets
    train_datasources: Tuple[DataSource] = (
        DataSource(
            name='openwebtext2',
            weights=0.6,
            data_file='./datasets/openwebtext2/train.npy',
            metadata_file='./datasets/openwebtext2/train_meta.pkl',
        ),
        DataSource(
            name='enwiki',
            weights=0.3,
            data_file='./datasets/enwiki/train.npy',
            metadata_file='./datasets/enwiki/train_meta.pkl',
        ),
        DataSource(
            name='zhwiki',
            weights=0.1,
            data_file='./datasets/zhwiki/train.npy',
            metadata_file='./datasets/zhwiki/train_meta.pkl',
        ),
    )
    eval_datasources: Tuple[DataSource] = (
        DataSource(
            name='openwebtext2',
            weights=0.6,
            data_file='./datasets/openwebtext2/eval.npy',
            metadata_file='./datasets/openwebtext2/eval_meta.pkl',
        ),
        DataSource(
            name='enwiki',
            weights=0.3,
            data_file='./datasets/enwiki/eval.npy',
            metadata_file='./datasets/enwiki/eval_meta.pkl',
        ),
        DataSource(
            name='zhwiki',
            weights=0.1,
            data_file='./datasets/zhwiki/eval.npy',
            metadata_file='./datasets/zhwiki/eval_meta.pkl',
        ),
    )
    dataloader_workers: int = 2
    max_seq_length: int = 1024  # sequence length for model input

    # training and evaluation loops
    num_epochs: int = 10
    train_batch_size: int = 6
    # accumulate gradients so for each iteration, the actual batch size is = train_batch_size x gradient_accum_steps
    gradient_accum_steps: int = 20
    eval_interval: int = 2000  # run M evaluation iterations after every N training iterations
    eval_iters: int = 100
    log_interval: int = 100  # log training metrics (loss, accuracy) every N training iterations
    ckpt_interval: int = 5000  # save model and optionally optimizer checkpoints every N training iterations
    save_optimizer: bool = True

    # learning rate scheduler
    init_lr: float = 4e-6  # initial learning rate
    max_lr: float = 4e-4  # max learning rate when warm up
    min_lr: float = 4e-5  # min learning rate after decay
    warmup_steps: int = 2000

    # AdamW optimizer
    weight_decay: float = 0.01
    adamw_betas: Tuple = (0.9, 0.95)
    adamw_eps: float = 1e-8
    adamw_fused: bool = True

    grad_clip: float = 1.0

    # dropout regularization
    embed_dropout: float = 0.1
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1

    # FSDP module
    mixed_precision: bool = True  # default to BF16, but if no native GPU support detected, will use FP16.
    # when use fsdp_activation_checkpointing, will see 30-50% training slowdown, but can free up ~30% GPU RAM thus we can use larger batch size
    fsdp_activation_checkpointing: bool = False
    limit_all_gathers: bool = True
    forward_prefetch: bool = True
    backward_prefetch: BackwardPrefetch = BackwardPrefetch.BACKWARD_PRE  # BACKWARD_PRE, BACKWARD_POST
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD  # FULL_SHARD, HYBRID_SHARD, SHARD_GRAD_OP, NO_SHARD
    checkpoint_type: StateDictType = StateDictType.FULL_STATE_DICT  # alternatively can use SHARDED_STATE_DICT to avoid OOMs
    compile_model: bool = False  # BUG in torch 2.0.1 when working with FSDP, not support python 3.11

    # resume training
    start_from_iter: int = 0  # default starts from iteration 0
    load_model_ckpt: str = ''  # load model state from checkpoint file
    load_optim_ckpt: str = ''  # load optimizer state from checkpoint file

    # others
    seed: int = 127
    log_dir: str = './logs/pretrain'  # save logs and traces
    ckpt_dir: str = './checkpoints/pretrain'
    use_tensorboard: bool = True
    use_profiler: bool = False  # use torch profiler to monitoring traces
    track_gpu_mem_usage: bool = False  # track GPU memory allocation statistics
