# LingBot-VA model environment (post-training + inference server).
#
# STRICT: every pip command below is the official upstream README, verbatim and
# in order -- README "Quick Start > Installation" and "Post-Training >
# Additional Dependencies". No added package, no added pin, no reordering.
#
#     docker build -f docker/Dockerfile -t lingbotva-rmbench:latest docker/
#
# Environment-only: repo, checkpoints and datasets are bind-mounted at runtime
# (see docker/README.md), so code edits never require a rebuild.
#
# README requirements: Python == 3.10.16, PyTorch == 2.9.0, CUDA 12.6.
# Ubuntu 22.04 ships Python 3.10.12 -- same minor version.

FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1

# System packages only -- the README assumes a host that already has these.
# python-is-python3: the launch scripts invoke bare `python`.
# build-essential/ninja: flash-attn compiles from source.
# libgl1/libglib: shared libraries opencv-python links against.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip python-is-python3 \
        git ca-certificates build-essential ninja-build \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- README > Quick Start > Installation ------------------------------------
RUN pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
        --index-url https://download.pytorch.org/whl/cu126

RUN pip install websockets einops diffusers==0.36.0 transformers==4.55.2 \
        accelerate msgpack opencv-python matplotlib ftfy easydict

# MAX_JOBS caps the flash-attn compile; lower it if the build OOMs the host.
ARG MAX_JOBS=8
ENV MAX_JOBS=${MAX_JOBS}
RUN pip install flash-attn --no-build-isolation

# --- README > Post-Training > Additional Dependencies -----------------------
RUN pip install lerobot==0.3.3 scipy wandb --no-deps

# ---------------------------------------------------------------------------
# KNOWN CONSEQUENCE of the --no-deps line above -- NOT patched here, because
# the README does not patch it.
#
# --no-deps installs those three packages alone, so their own imports are
# unsatisfied: `import wandb` (unconditional at wan_va/train.py:6) needs click,
# gitpython, sentry-sdk, platformdirs, protobuf; lerobot needs datasets, av,
# h5py, jsonlines. Training therefore fails at import until they are present.
#
# Fix it at RUN time, inside the container, if it bites:
#
#     pip install datasets av h5py jsonlines \
#         click sentry-sdk gitpython platformdirs protobuf
#
# ...or set enable_wandb=False in the train config, which removes only the
# wandb half of the problem (train.py imports wandb regardless of the flag).
# ---------------------------------------------------------------------------

WORKDIR /workspace/lingbot-va
CMD ["/bin/bash"]
