from typing import Tuple
from dataclasses import dataclass

from torch.distributed.fsdp import ShardingStrategy, BackwardPrefetch
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


@dataclass
class config:
    """fine-tuning configurations, where we use smaller learning rates, and less training steps"""

    # model definition, the details (number of layers, heads etc.) are defined in models/gpt2_model.py --> gpt2_meta
    model_type: str = "gpt2-xl"  # gpt2, gpt2-medium, gpt2-large, gpt2-xl

    pretrain_ckpt_file = "./checkpoints/gpt2-xl-openai-pretrained.pt"  # load pretrained checkpoint

    # datasets
    train_datasources: Tuple[str] = (
        './datasets/SQuAD/train_v2.0.pkl',
        './datasets/MARCO_QnA/train_v2.1.pkl',
        './datasets/dolly/train.pkl',
        './datasets/commonsense_dialogues/train.pkl',
        './datasets/mathematics_dataset_v1.0/train.pkl',
    )
    eval_datasources: Tuple[str] = (
        './datasets/SQuAD/dev_v2.0.pkl',
        './datasets/MARCO_QnA/dev_v2.1.pkl',
        './datasets/dolly/dev.pkl',
        './datasets/commonsense_dialogues/dev.pkl',
        './datasets/mathematics_dataset_v1.0/dev.pkl',
    )
    dataloader_workers: int = 2

    max_seq_length: int = 1024  # use smaller sequence length to save GPU RAM
    pad_id: int = 50256  # here we use eot_token id, since GPT-2 model don't have a pad token

    # training and evaluation loops
    max_train_iters: int = 10000  # training samples * epochs / batch size, 500000 training samples, with batch size of 120, 4000 iters = one epoch
    micro_batch_size: int = 6  # on single RTX 3090, use 6 for gpt-xl, 26 for gpt2-medium
    # accumulate gradients so for each iteration, the actual batch size is = micro_batch_size x gradient_accum_steps
    gradient_accum_steps: int = 20
    eval_interval: int = 200  # run M evaluation iterations after every N training iterations
    eval_iters: int = 100  # large size since micro_batch_size is very small
    log_interval: int = 100  # log training metrics (loss, accuracy) every N training iterations
    ckpt_interval: int = 500  # save model and optionally optimizer checkpoints every N training iterations
    save_optimizer: bool = False  # only valid if ckpt_interval > 0

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    train_bias: str = 'lora_only'  # none, lora_only, all

    # learning rate scheduler
    init_lr: float = 1e-8  # initial learning rate
    max_lr: float = 2e-6  # max learning rate when warm up, 0.02 x pre-training learning rate
    min_lr: float = 2e-7  # min learning rate after decay
    warmup_steps: int = 100
    max_decay_steps: int = 10000

    # prompt is less important than completion
    prompt_loss_weight: float = 0.01
    completion_loss_weight: float = 1.0

    # AdamW optimizer
    weight_decay: float = 0.0
    adamw_betas: Tuple = (0.9, 0.95)
    adamw_eps: float = 1e-8
    adamw_fused: bool = True

    grad_clip: float = 1.0

    # dropout regularization
    embed_dropout: float = 0.1
    attn_dropout: float = 0.2
    resid_dropout: float = 0.2

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
    log_dir: str = './logs/finetune_lora'  # save logs and traces
    ckpt_dir: str = './checkpoints/finetune_lora'
    use_tensorboard: bool = True
    use_profiler: bool = False  # use torch profiler to monitoring traces
    track_gpu_mem_usage: bool = False  # track GPU memory allocation statistics
