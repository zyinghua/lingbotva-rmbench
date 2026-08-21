# RMBench -> LingBot-VA data pipeline

Converts raw RMBench tasks (RoboTwin 2.0 HDF5) into the latent LeRobot format
LingBot-VA post-trains on. Written from three verifiable sources only:

1. **This repo's code** — the dataset loader
   (`wan_va/dataset/lerobot_latent_dataset.py`) defines what training reads;
   the inference server (`wan_va/wan_va_server.py`) defines the encoders that
   extraction must mirror (`_encode_obs`, `_get_t5_prompt_embeds`).
2. **The official released dataset** `robbyant/robotwin-clean-and-aug-lerobot`,
   inspected file by file. Measured ground truth: fps 50, stride 4 sampling
   (`fps` field = 12), sampled frames truncated to `4k+1`, cam_high 256x320 →
   latent 16x20, wrists 128x160 → 8x10, latents flattened `[N, 48]` bf16,
   `text_emb [512, 4096]` bf16 in every camera file, `empty_emb.pt` at root.
3. **RMBench raw data** (`TianxingChen/RMBench`): per-episode HDF5 with
   `endpose/*` (EEF xyz+quat+gripper), `joint_action/*`, per-camera JPEG
   streams; instructions in `instructions/episodeN.json`.

## Steps (per task)

```bash
# 1. raw HDF5 -> LeRobot v2.1 (parquet + per-camera mp4s + meta with action_config)
python tools/rmbench/raw_to_lerobot.py \
    --raw-root /datasets/RMBench-data/data --task put_back_block \
    --out /datasets/lingbot-va-rmbench/put_back_block \
    --instruction-file RMBench/description/task_instruction/put_back_block.json

# 2. action normalization quantiles (episode-relative poses, canonical 30-dim)
python tools/rmbench/compute_norm_stat.py \
    --dataset /datasets/lingbot-va-rmbench/put_back_block

# 3. VAE latents + text embeddings + empty_emb.pt   (the only GPU step, 1 GPU)
python tools/rmbench/extract_latents.py \
    --dataset /datasets/lingbot-va-rmbench/put_back_block \
    --model /workspace/lingbotva-rmbench/ckpts/lingbot-va-base \
    --write-empty-emb

# 4. read-only structural validation before training
python tools/rmbench/validate_dataset.py \
    --dataset /datasets/lingbot-va-rmbench/put_back_block \
    --expect-episodes 50
```

## Then: configs

Copy `wan_va/configs/va_robotwin_cfg.py` + `va_robotwin_train_cfg.py` to
`va_rmbench_<task>_cfg.py` / `..._train_cfg.py` and change:

- `norm_stat` — paste the block step 2 printed (**the** silent-failure field:
  wrong quantiles raise nothing, the policy just mis-scales every motion)
- `dataset_path` / `empty_emb_path` (train cfg)
- `save_root` — per-task; checkpoints are named by step only and collide otherwise
- `num_steps` / `save_interval` — scale to ~50 episodes, not 50k steps
- register both in `wan_va/configs/__init__.py` (`VA_CONFIGS`)

Everything else (cameras, T-shape, `action_per_frame=16`, `attn_window`,
snr shifts) is inherited unchanged: same embodiment, same 50 fps platform,
`env_type='robotwin_tshape'` is what makes the loader, latent tiling and eval
client work with zero code changes.

## Invariants worth knowing

- `action_per_frame = stride * 4`. Stride 4 ⇒ 16. Change one, change both.
- Latent files store the **resized** video dims (256x320 / 128x160), not raw.
- Wrist latents tile as [left|right] along width, stacked above cam_high along
  height → 24x20 per frame; the two wrist widths must sum to cam_high's width,
  which the height//2, width//2 rule guarantees.
- Actions in parquet stay **absolute**; the loader converts to episode-relative
  at read time. norm_stat must be computed on the *relative* values (step 2
  does), never on the raw columns.
- The LeRobot row alignment follows the released RoboTwin data: row `t` stores
  observation/state from raw frame `t`, but its supervised action is the EEF
  target at raw frame `t+1`. Therefore each exported episode has `raw_T - 1`
  rows and its video omits the final raw frame.
- RMBench's HDF5 JPEGs preserve simulator RGB numerically despite passing
  through OpenCV. The converter handles the required RGB->BGR conversion only
  at the MP4 writer boundary; do not add another channel swap.
- `deps`: h5py, cv2, pyarrow, scipy — all present in the image after the
  documented post-`--no-deps` fix; step 3 additionally needs torch + the
  checkpoint's vae/text_encoder/tokenizer.
