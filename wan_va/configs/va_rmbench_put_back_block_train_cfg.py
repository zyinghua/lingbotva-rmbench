# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Single-task post-training config for 50 RMBench put_back_block episodes."""

import os

from easydict import EasyDict

from .va_rmbench_put_back_block_cfg import va_rmbench_put_back_block_cfg


va_rmbench_put_back_block_train_cfg = EasyDict()
va_rmbench_put_back_block_train_cfg.update(va_rmbench_put_back_block_cfg)
va_rmbench_put_back_block_train_cfg.__name__ = (
    "Config: VA RMBench put_back_block train"
)

_data_root = os.environ.get(
    "RMBENCH_LINGBOTVA_DATA_ROOT",
    "/datasets/RMBench-data/lingbotva-rmbench",
)
va_rmbench_put_back_block_train_cfg.dataset_path = os.path.join(
    _data_root, "put_back_block"
)
va_rmbench_put_back_block_train_cfg.empty_emb_path = os.path.join(
    va_rmbench_put_back_block_train_cfg.dataset_path, "empty_emb.pt"
)
va_rmbench_put_back_block_train_cfg.save_root = os.environ.get(
    "VA_SAVE_ROOT", "./train_out/rmbench-put-back-block-50-from-base-3k"
)

va_rmbench_put_back_block_train_cfg.enable_wandb = os.environ.get(
    "VA_ENABLE_WANDB", "0"
).lower() in {"1", "true", "yes"}
va_rmbench_put_back_block_train_cfg.dataset_init_worker = int(
    os.environ.get("VA_DATASET_INIT_WORKERS", "1")
)
va_rmbench_put_back_block_train_cfg.load_worker = int(
    os.environ.get("VA_LOAD_WORKERS", "0")
)
va_rmbench_put_back_block_train_cfg.save_interval = int(
    os.environ.get("VA_SAVE_INTERVAL", "1000")
)
va_rmbench_put_back_block_train_cfg.gc_interval = 50
va_rmbench_put_back_block_train_cfg.cfg_prob = 0.1

# Match the released RoboTwin post-training optimizer recipe.
va_rmbench_put_back_block_train_cfg.learning_rate = 1e-5
va_rmbench_put_back_block_train_cfg.beta1 = 0.9
va_rmbench_put_back_block_train_cfg.beta2 = 0.95
va_rmbench_put_back_block_train_cfg.weight_decay = 0.1
va_rmbench_put_back_block_train_cfg.warmup_steps = 10
va_rmbench_put_back_block_train_cfg.batch_size = 1
va_rmbench_put_back_block_train_cfg.gradient_accumulation_steps = int(
    os.environ.get("VA_GRAD_ACCUM", "10")
)
va_rmbench_put_back_block_train_cfg.num_steps = int(
    os.environ.get("VA_NUM_STEPS", "1500")
)
