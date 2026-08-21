# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""LingBot-VA config for RMBench ``put_back_block``.

RMBench inherits RoboTwin's Aloha embodiment, three-camera T-shaped latent
layout, and 16-D dual-EEF action representation. The only task-specific model
setting is action normalization, computed from all 50 converted episodes
(17,562 aligned rows) after the loader's episode-relative pose conversion.
"""

import os

from easydict import EasyDict

from .va_robotwin_cfg import va_robotwin_cfg


va_rmbench_put_back_block_cfg = EasyDict()
va_rmbench_put_back_block_cfg.update(va_robotwin_cfg)
va_rmbench_put_back_block_cfg.__name__ = "Config: VA RMBench put_back_block"
va_rmbench_put_back_block_cfg.infer_mode = "server"

# Train from the released LingBot-VA backbone, not the RoboTwin-posttrained
# checkpoint. The same override points evaluation at a produced checkpoint.
va_rmbench_put_back_block_cfg.wan22_pretrained_model_name_or_path = os.environ.get(
    "VA_PRETRAINED_MODEL_PATH",
    "/workspace/lingbotva-rmbench/ckpts/lingbot-va-base",
)

va_rmbench_put_back_block_cfg.action_norm_method = "quantiles"
va_rmbench_put_back_block_cfg.norm_stat = {
    "q01": [
        -0.0011135616898536682,
        -1.0132789611816406e-06,
        -0.016312502026557922,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -0.30552986592054365,
        0.0001441428065299988,
        -0.040277838706970215,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    "q99": [
        0.047633704543113706,
        0.2145732745528221,
        0.07280539393424985,
        1.0,
        1.0,
        1.0,
        1.0,
        -9.665995836257964e-05,
        0.31461397886276243,
        0.09563056826591486,
        1.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
    ],
}
