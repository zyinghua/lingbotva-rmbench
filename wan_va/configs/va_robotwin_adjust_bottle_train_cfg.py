# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Single-task post-training config for 50 clean RoboTwin adjust_bottle demos.

This inherits the released RoboTwin model/action/camera settings, but points
dataset discovery at the exact task directory so adding another dataset beside
it cannot silently turn this run into multi-task training.
"""

import os

from easydict import EasyDict

from .va_robotwin_train_cfg import va_robotwin_train_cfg


va_robotwin_adjust_bottle_train_cfg = EasyDict(
    __name__="Config: VA RoboTwin adjust_bottle 50-demo train"
)
va_robotwin_adjust_bottle_train_cfg.update(va_robotwin_train_cfg)

_data_root = os.environ.get(
    "ROBOTWIN_LINGBOTVA_DATA_ROOT",
    "/datasets/robotwin2.0-fastwam/lingbotva-robotwin",
)
va_robotwin_adjust_bottle_train_cfg.dataset_path = os.path.join(
    _data_root,
    "lerobot_robotwin_eef_clean_50",
    "adjust_bottle-demo_clean_collect_200-50",
)
va_robotwin_adjust_bottle_train_cfg.empty_emb_path = os.path.join(
    _data_root, "empty_emb.pt"
)

# Training starts from the model selected by va_robotwin_cfg by default. This
# environment override lets a reproduction run select a different base without
# changing the already-validated inference config.
va_robotwin_adjust_bottle_train_cfg.wan22_pretrained_model_name_or_path = (
    os.environ.get(
        "VA_PRETRAINED_MODEL_PATH",
        va_robotwin_adjust_bottle_train_cfg.wan22_pretrained_model_name_or_path,
    )
)

va_robotwin_adjust_bottle_train_cfg.save_root = os.environ.get(
    "VA_SAVE_ROOT", "./train_out/robotwin-adjust-bottle-50"
)
va_robotwin_adjust_bottle_train_cfg.enable_wandb = os.environ.get(
    "VA_ENABLE_WANDB", "0"
).lower() in {"1", "true", "yes"}
va_robotwin_adjust_bottle_train_cfg.dataset_init_worker = int(
    os.environ.get("VA_DATASET_INIT_WORKERS", "1")
)
va_robotwin_adjust_bottle_train_cfg.load_worker = int(
    os.environ.get("VA_LOAD_WORKERS", "0")
)
va_robotwin_adjust_bottle_train_cfg.save_interval = int(
    os.environ.get("VA_SAVE_INTERVAL", "1000")
)
va_robotwin_adjust_bottle_train_cfg.gc_interval = 50
va_robotwin_adjust_bottle_train_cfg.cfg_prob = 0.1

# Paper post-training recipe for a 50-demonstration task.
va_robotwin_adjust_bottle_train_cfg.learning_rate = 1e-5
va_robotwin_adjust_bottle_train_cfg.beta1 = 0.9
va_robotwin_adjust_bottle_train_cfg.beta2 = 0.95
va_robotwin_adjust_bottle_train_cfg.weight_decay = 0.1
va_robotwin_adjust_bottle_train_cfg.warmup_steps = 10
va_robotwin_adjust_bottle_train_cfg.batch_size = 1
va_robotwin_adjust_bottle_train_cfg.gradient_accumulation_steps = 1
va_robotwin_adjust_bottle_train_cfg.num_steps = int(
    os.environ.get("VA_NUM_STEPS", "3000")
)
