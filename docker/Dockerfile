# LingBot-VA environment: post-training, inference server, and -- after the
# one run-time step in docker/README.md -- RoboTwin evaluation as well.
#
# STRICT: every pip command below is the official upstream README, verbatim and
# in order -- README "Quick Start > Installation" and "Post-Training >
# Additional Dependencies". No added package, no added pin, no reordering.
#
# The RoboTwin simulator stack is deliberately NOT built in. It is a different
# concern with its own pins (scipy==1.10.1 against this env's unpinned scipy),
# it is only needed when you want simulator success rates, and it wants a
# RoboTwin checkout that is bind-mounted anyway. Install it inside the running
# container instead -- one command, documented in docker/README.md.
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

# SAPIEN headless rendering needs the NVIDIA graphics capability in addition
# to the CUDA image's default compute/utility capabilities.  `--gpus all` is
# still required when the container is created (see docker/README.md).
ENV NVIDIA_DRIVER_CAPABILITIES=all

# System packages only -- the README assumes a host that already has these.
# python-is-python3: the launch scripts invoke bare `python`.
# build-essential/ninja: flash-attn compiles from source.
# libgl1/libglib/libx11/libxext: shared libraries opencv and SAPIEN link against.
# libegl1/libegl-mesa0: EGL loader + vendor directory required by SAPIEN 3.0.0b1.
# unzip: RoboTwin's asset downloader extracts its three downloaded archives.
#
# The vulkan trio is README "Evaluation on RoboTwin-2.0" step 1, baked in here
# so that adding the RoboTwin python stack at run time (see docker/README.md)
# needs no root apt inside a live container. They are inert if you only train.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip python-is-python3 \
        git ca-certificates build-essential ninja-build \
        ffmpeg libgl1 libglib2.0-0 libx11-6 libxext6 \
        libegl1 libegl-mesa0 unzip \
        libvulkan1 mesa-vulkan-drivers vulkan-tools \
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

WORKDIR /workspace/lingbotva-rmbench
CMD ["/bin/bash"]
