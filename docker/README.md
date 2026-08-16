# Docker setup

Containerised form of the **official LingBot-VA install instructions** (upstream
`Robbyant/lingbot-va` README) — nothing else. Two images, matching the
two-environment split that README requires:

| Image | Dockerfile | Follows | Runs |
|---|---|---|---|
| `lingbotva-rmbench` | `Dockerfile` | README "Quick Start > Installation" + "Post-Training > Additional Dependencies" | post-training + inference server |
| `lingbotva-rmbench-robotwin` | `Dockerfile.robotwin` | README "Evaluation on RoboTwin-2.0 > Preparing the Environment" (steps 1–5) | RoboTwin 2.0 evaluation client |

**No deviations.** Every `pip` command in both files is the README's, verbatim
and in order — no added package, no added pin, no reordering. Only system
(`apt`) packages are supplied on top, because the README assumes a host that
already has an interpreter, a compiler and the shared libraries its pip packages
link against.

Both are **environment-only**: the repo, checkpoints, datasets and the RoboTwin
checkout are bind-mounted at runtime, so code edits never require a rebuild.

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
docker build -f docker/Dockerfile          -t lingbotva-rmbench:latest       docker/
docker build -f docker/Dockerfile.robotwin -t lingbotva-rmbench-robotwin:latest docker/
```

flash-attn / pytorch3d compile from source if no wheel matches; pass
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

## Run: inference server (same image)

```bash
docker run --gpus all -it --network host \
    -v "$PWD":/workspace/lingbot-va \
    -v /path/to/ckpts:/workspace/lingbot-va/ckpts \
    lingbotva-rmbench:latest \
    bash evaluation/robotwin/launch_server.sh
```

`--network host` lets the client reach the server on `localhost:29536`.

## Run: RoboTwin evaluation client

Server and client must share the machine (README requirement); with host
networking two containers count as sharing.

```bash
docker run --gpus all -it --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v "$PWD":/workspace/lingbot-va \
    -v /path/to/RoboTwin:/workspace/RoboTwin \
    lingbotva-rmbench-robotwin:latest
# inside:
ROBOTWIN_ROOT=/workspace/RoboTwin bash evaluation/robotwin/launch_client.sh results/ adjust_bottle
```

- `NVIDIA_DRIVER_CAPABILITIES=all` makes the NVIDIA runtime inject the Vulkan
  ICD, which SAPIEN's headless renderer needs. Verify with `vulkaninfo --summary`
  inside the container before debugging SAPIEN itself.
- The RoboTwin checkout (`git checkout 2eeec322`) and its assets
  (`bash script/_download_assets.sh`) live on the host and are mounted in.
- `ROBOTWIN_ROOT` is read by `evaluation/robotwin/eval_polict_client_openpi.py`
  (defaults to `/workspace/RoboTwin`).
