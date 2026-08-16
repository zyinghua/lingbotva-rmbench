# Docker setup

**One image**, `lingbotva-rmbench`, built from `Dockerfile`. It is the
containerised form of the official upstream `Robbyant/lingbot-va` README —
"Quick Start > Installation" plus "Post-Training > Additional Dependencies" —
and it runs post-training and the inference server.

RoboTwin evaluation runs in the *same* container after one run-time install
step (below). It is not baked into the image because it has its own pins, it is
only needed when you want simulator success rates, and it needs a RoboTwin
checkout that is bind-mounted anyway.

**No deviations.** Every `pip` command in the Dockerfile is the README's,
verbatim and in order — no added package, no added pin, no reordering. Only
system (`apt`) packages sit on top, because the README assumes a host that
already has an interpreter, a compiler, and the shared libraries its pip
packages link against.

The image is **environment-only**: the repo, checkpoints, datasets and the
RoboTwin checkout are bind-mounted at run time, so code edits never require a
rebuild.

### One thing the README leaves broken

`pip install lerobot==0.3.3 scipy wandb --no-deps` installs those three packages
*alone*, so their own imports are unsatisfied. `wan_va/train.py:6` does
`import wandb` unconditionally, so **training fails at import** on a literal
build. The Dockerfile documents this rather than patching it; run this inside
the container when it bites:

```bash
pip install datasets av h5py jsonlines \
    click sentry-sdk gitpython platformdirs protobuf
```

Setting `enable_wandb=False` in the train config is not sufficient on its own —
`train.py` imports wandb regardless of the flag.

## Build

```bash
docker build -f docker/Dockerfile -t lingbotva-rmbench:latest docker/
```

flash-attn compiles from source if no wheel matches; pass
`--build-arg MAX_JOBS=4` on small-RAM hosts.

All commands below are run from the **repo root** (`lingbot-va-rmbench/`), which
is what `$PWD` refers to in the bind mounts.

## Run: training

```bash
docker run --gpus all -it --shm-size 64g --network host \
    -v "$PWD":/workspace/lingbot-va \
    -v /path/to/ckpts:/workspace/lingbot-va/ckpts \
    -v /path/to/datasets:/datasets \
    -e WANDB_API_KEY -e WANDB_BASE_URL -e WANDB_TEAM_NAME -e WANDB_PROJECT \
    lingbotva-rmbench:latest
# inside:
NGPU=8 CONFIG_NAME='robotwin_train' bash script/run_va_posttrain.sh --save-root train_out/my_task
```

- `--shm-size` is required: the DataLoader runs 16 workers over shared memory;
  the docker default (64 MB) kills them.
- `run_va_posttrain.sh` respects `WANDB_*` from the environment (falls back to
  placeholders); with no wandb account set `enable_wandb=False` in the train config.
- Config edits (dataset_path, checkpoint path) are on the host — the repo is a mount.

## Run: inference server

```bash
docker run --gpus all -it --network host \
    -v "$PWD":/workspace/lingbot-va \
    -v /path/to/ckpts:/workspace/lingbot-va/ckpts \
    lingbotva-rmbench:latest \
    bash evaluation/robotwin/launch_server.sh
```

`--network host` lets the client reach the server on `localhost:29536`.

## RoboTwin evaluation: one-time setup inside the container

Start the container with the RoboTwin checkout mounted and Vulkan enabled:

```bash
docker run --gpus all -it --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v "$PWD":/workspace/lingbot-va \
    -v /path/to/RoboTwin:/workspace/RoboTwin \
    lingbotva-rmbench:latest
```

Then, **inside** it, run README "Evaluation on RoboTwin-2.0 > Preparing the
Environment" steps 3–5 (step 1's vulkan packages are already in the image;
step 2's clone is the mount above):

```bash
pip install \
    transforms3d==0.4.2 sapien==3.0.0b1 scipy==1.10.1 mplib==0.2.1 \
    gymnasium==0.29.1 trimesh==4.4.3 open3d==0.18.0 imageio==2.34.2 \
    pydantic zarr openai huggingface_hub==0.36.2 h5py \
    azure==4.0.0 azure-ai-inference "pyglet<2" wandb moviepy imageio \
    termcolor av matplotlib ffmpeg
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
```

Assets (README step 6) are downloaded once into the mounted checkout:

```bash
cd /workspace/RoboTwin && bash script/_download_assets.sh
```

Then launch server and client from two shells in the same container:

```bash
bash evaluation/robotwin/launch_server.sh
ROBOTWIN_ROOT=/workspace/RoboTwin bash evaluation/robotwin/launch_client.sh results/ adjust_bottle
```

### Two things to know

- **`scipy` gets downgraded to 1.10.1.** The model env installs `scipy`
  unpinned; RoboTwin pins 1.10.1, and pip will move it. Both users of scipy in
  this repo — `wan_va/dataset/lerobot_latent_dataset.py:15` and
  `evaluation/robotwin/eval_polict_client_openpi.py:39` — only want
  `scipy.spatial.transform.Rotation`, which 1.10.1 has, so this is expected to be
  harmless. `numpy` follows it down to 1.26.x, which is what upstream's own
  `requirements.txt` pins anyway. If it ever does bite, put the RoboTwin stack in
  its own venv instead:
  `python -m venv --system-site-packages /workspace/rt-venv && /workspace/rt-venv/bin/pip install ...`
- **Container-local installs are lost when the container is removed.** Either
  keep the container around (`docker start -ai <name>` instead of `docker run`),
  or `docker commit` it once the RoboTwin stack is in, or put the venv on a
  mounted volume as above.

- `NVIDIA_DRIVER_CAPABILITIES=all` makes the NVIDIA runtime inject the Vulkan
  ICD, which SAPIEN's headless renderer needs. Verify with `vulkaninfo --summary`
  inside the container before debugging SAPIEN itself.
- `ROBOTWIN_ROOT` is read by `evaluation/robotwin/eval_polict_client_openpi.py`
  (defaults to `/workspace/RoboTwin`).
