"""Extract Wan2.2 VAE latents + UMT5 text embeddings for an exported RMBench task.

Produces latents/chunk-000/<cam_key>/episode_{idx:06d}_{start}_{end}.pth, the
format LingBot-VA trains from. Every number below is pinned by two sources:

  * this repo's inference server (wan_va/wan_va_server.py), which this script
    mirrors operation for operation so train-time latents and deployment-time
    encoding cannot disagree:
      - _encode_obs, env_type='robotwin_tshape': cam_high resized to
        (height, width), both wrists to (height//2, width//2); F.interpolate
        bilinear align_corners=False on 0-255 floats, THEN /255*2-1; streaming
        causal VAE encode; mu half of the output; normalized as
        (mu - latents_mean) / latents_std  (normalize_latents with 1/std).
      - _get_t5_prompt_embeds: prompt_clean, tokenizer max_length=512, UMT5
        last_hidden_state, valid rows kept and zero-padded back to 512.
  * the officially released robotwin dataset (robbyant/robotwin-clean-and-aug-
    lerobot), whose latent files were inspected directly: recorded at 50 fps,
    sampled with stride 4 (fps field = 12), sampled count truncated to 4k+1
    (139 raw -> 35 -> 33 -> 9 latent frames), cam_high 256x320 -> latent 16x20,
    wrists 128x160 -> 8x10, latent stored flattened [N, 48] bfloat16,
    text_emb [512, 4096] bfloat16 duplicated into every camera's file.

The 4k+1 truncation makes the causal VAE's output length 1 + (T-1)/4 exact.
Note action_per_frame in the config is tied to the stride here:
action_per_frame = stride * 4 (va_robotwin_cfg: 16 at stride 4).

Usage
-----
    python tools/rmbench/extract_latents.py \
        --dataset /datasets/RMBench-data/lingbotva-rmbench/put_back_block \
        --model   /workspace/lingbotva-rmbench/ckpts/lingbot-va-base \
        --write-empty-emb
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "wan_va"))

from diffusers.pipelines.wan.pipeline_wan import prompt_clean          # noqa: E402
from modules.utils import (                                            # noqa: E402
    WanVAEStreamingWrapper, load_text_encoder, load_tokenizer, load_vae)

CAM_KEYS = ["observation.images.cam_high",
            "observation.images.cam_left_wrist",
            "observation.images.cam_right_wrist"]
AV1_CODEC_ID = av.codec.Codec("av1", "r").id


def plan_frame_ids(start, end, stride):
    """Plan the largest causal-VAE-compatible 4k+1 subsequence."""
    if not 0 <= start < end:
        raise ValueError(f"invalid action segment [{start},{end})")
    frame_ids = list(range(start, end, stride))
    if not frame_ids:
        raise ValueError(f"action segment [{start},{end}) yields no sampled frames")
    keep = len(frame_ids) - ((len(frame_ids) - 1) % 4)
    return frame_ids[:keep]


def validate_extraction_plan(dataset, info, episodes, stride):
    """Fail before model loading if metadata cannot satisfy the training loader."""
    if info.get("fps") != 50:
        raise ValueError(f"dataset fps is {info.get('fps')}, expected RoboTwin's 50")
    if info.get("total_episodes") != len(episodes):
        raise ValueError(
            f"info.total_episodes={info.get('total_episodes')}, but metadata has "
            f"{len(episodes)} episodes"
        )
    action_feature = info.get("features", {}).get("action", {})
    if (
        action_feature.get("dtype") != "float32"
        or action_feature.get("shape") != [16]
    ):
        raise ValueError(
            f"dataset action feature is {action_feature}; expected EEF float32[16]"
        )
    for camera in CAM_KEYS:
        feature = info.get("features", {}).get(camera, {})
        feature_info = feature.get("info", {})
        if (
            feature.get("dtype") != "video"
            or feature.get("shape") != [3, 480, 640]
            or feature_info.get("video.codec") != "av1"
            or feature_info.get("video.pix_fmt") != "yuv420p"
            or feature_info.get("video.fps") != 50
        ):
            raise ValueError(f"{camera}: incompatible video metadata {feature}")

    planned_segments = 0
    for episode in episodes:
        episode_index = episode["episode_index"]
        episode_length = episode["length"]
        parquet_path = (
            dataset / "data" / "chunk-000"
            / f"episode_{episode_index:06d}.parquet"
        )
        if not parquet_path.is_file():
            raise FileNotFoundError(f"missing {parquet_path}")
        action_table = pq.read_table(parquet_path, columns=["action"])
        parquet_rows = action_table.num_rows
        if parquet_rows != episode_length:
            raise ValueError(
                f"{parquet_path}: {parquet_rows} rows, metadata says {episode_length}"
            )
        action_type = action_table["action"].type
        if getattr(action_type, "list_size", None) != 16:
            raise ValueError(
                f"{parquet_path}: action column type is {action_type}, expected "
                "fixed-size list[16]"
            )

        action_configs = episode.get("action_config")
        if not isinstance(action_configs, list) or not action_configs:
            raise ValueError(f"episode {episode_index}: missing action_config")
        for segment in action_configs:
            start, end = segment["start_frame"], segment["end_frame"]
            if not 0 <= start < end <= episode_length:
                raise ValueError(
                    f"episode {episode_index}: invalid segment [{start},{end}) "
                    f"for length {episode_length}"
                )
            text = segment.get("action_text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"episode {episode_index} [{start},{end}): empty action_text"
                )

            frame_ids = plan_frame_ids(start, end, stride)
            if len(frame_ids) < 5:
                raise ValueError(
                    f"episode {episode_index} [{start},{end}) has only "
                    f"{len(frame_ids)} aligned sampled frames"
                )

            latent_frames = 1 + (len(frame_ids) - 1) // 4
            required_actions = latent_frames * stride * 4
            # _action_post_process drops rows before frame_ids[0], prepends one
            # stride*4 zero-action block, then requires exactly this many rows.
            available_actions = stride * 4 + end - frame_ids[0]
            if available_actions < required_actions:
                raise ValueError(
                    f"episode {episode_index} [{start},{end}): loader needs "
                    f"{required_actions} action rows after padding, but only "
                    f"{available_actions} are available"
                )
            planned_segments += 1

        for camera in CAM_KEYS:
            video_path = (
                dataset / "videos" / "chunk-000" / camera
                / f"episode_{episode_index:06d}.mp4"
            )
            if not video_path.is_file():
                raise FileNotFoundError(f"missing {video_path}")
            with av.open(str(video_path), "r") as container:
                try:
                    stream = container.streams.video[0]
                except IndexError as error:
                    raise ValueError(f"{video_path}: no video stream") from error
                codec_context = stream.codec_context
                codec = codec_context.codec
                properties = {
                    "codec_id": codec.id,
                    "codec_name": codec.name,
                    "pix_fmt": codec_context.pix_fmt,
                    "fps": float(stream.average_rate),
                    "height": codec_context.height,
                    "width": codec_context.width,
                }
            expected = {
                "codec_id": AV1_CODEC_ID,
                "pix_fmt": "yuv420p",
                "fps": 50.0,
                "height": 480,
                "width": 640,
            }
            comparable = {key: properties[key] for key in expected}
            if comparable != expected:
                raise ValueError(
                    f"{video_path}: stream properties {properties}, expected {expected}"
                )

    return planned_segments


def encode_text(tokenizer, text_encoder, text, device):
    """Mirror of VA_Server._get_t5_prompt_embeds for a single prompt."""
    prompt = [prompt_clean(text)]
    ti = tokenizer(prompt, padding="max_length", max_length=512, truncation=True,
                   add_special_tokens=True, return_attention_mask=True,
                   return_tensors="pt")
    ids, mask = ti.input_ids.to(device), ti.attention_mask.to(device)
    seq_len = int(mask.gt(0).sum())
    with torch.no_grad():
        emb = text_encoder(ids, mask).last_hidden_state[0]             # [512, 4096]
    emb = torch.cat([emb[:seq_len], emb.new_zeros(512 - seq_len, emb.shape[1])])
    return emb.to(torch.bfloat16).cpu()


def read_video_frames(path, frame_ids):
    """Decode selected RGB frames with PyAV, as LeRobot does.

    The official RoboTwin videos are AV1. The OpenCV build in the target image
    cannot decode them reliably, while PyAV uses the available libdav1d stack.
    """
    wanted = set(frame_ids)
    found = {}
    last_wanted = max(wanted)
    with av.open(str(path), "r") as container:
        for idx, frame in enumerate(container.decode(video=0)):
            if idx in wanted:
                found[idx] = frame.to_ndarray(format="rgb24")
            if idx >= last_wanted:
                break
    missing = [idx for idx in frame_ids if idx not in found]
    if missing:
        raise RuntimeError(
            f"{path}: missing {len(missing)} requested frames; first missing={missing[0]}"
        )
    return np.stack([found[idx] for idx in frame_ids])                  # (T,H,W,3) uint8


@torch.inference_mode()
def encode_clips(vae, clips, height, width, device, dtype):
    """Encode a same-length camera batch as 1 frame followed by 4-frame chunks.

    ``AutoencoderKLWan._encode`` uses this exact causal rhythm. Feeding an entire
    video to a fresh ``WanVAEStreamingWrapper`` in one call is incorrect: the
    first call primes the temporal-downsampling caches rather than applying all
    temporal convolutions.
    """
    tensors = []
    expected_frames = None
    for frames in clips:
        if expected_frames is None:
            expected_frames = len(frames)
        elif len(frames) != expected_frames:
            raise ValueError("all clips in a VAE batch must have the same length")
        x = torch.from_numpy(frames).float().permute(3, 0, 1, 2)       # C,T,H,W
        x = F.interpolate(x, size=(height, width), mode="bilinear",
                          align_corners=False)
        tensors.append(x)
    x = torch.stack(tensors, dim=0) / 255.0 * 2.0 - 1.0               # B,C,T,h,w
    x = x.to(device=device, dtype=dtype)

    streaming = WanVAEStreamingWrapper(vae)                            # independent causal cache
    enc_parts = [streaming.encode_chunk(x[:, :, :1])]
    for start in range(1, x.shape[2], 4):
        chunk = x[:, :, start:start + 4]
        if chunk.shape[2] != 4:
            raise ValueError(
                f"sampled video must have 4k+1 frames, got {x.shape[2]}"
            )
        enc_parts.append(streaming.encode_chunk(chunk))
    enc = torch.cat(enc_parts, dim=2)
    mu, _ = torch.chunk(enc, 2, dim=1)
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(mu.device)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(mu.device)
    return ((mu.float() - mean) * (1.0 / std)).to(mu)


def save_atomic(payload, path):
    """Do not leave a truncated .pth that a resumed run would mistake as complete."""
    temp = path.with_name(path.name + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def latent_payload(lat, text_emb, text, frame_ids, start, end, ori_fps, stride,
                   video_height, video_width):
    """Build the per-camera payload consumed by LatentLeRobotDataset."""
    C, Fh, Hh, Wh = lat.shape
    flat = lat.permute(1, 2, 3, 0).reshape(-1, C).to(torch.bfloat16).cpu()
    return {
        "latent": flat,                       # [F*h*w, 48] bf16, f-major
        "latent_num_frames": Fh,
        "latent_height": Hh,
        "latent_width": Wh,
        "video_num_frames": len(frame_ids),
        "video_height": video_height,
        "video_width": video_width,
        "text_emb": text_emb,                 # [512, 4096] bf16
        "text": text,
        "frame_ids": frame_ids,
        "start_frame": start,
        "end_frame": end,
        # The released RoboTwin cache stores the integer field 12 for 50/4.
        "fps": int(ori_fps / stride),
        "ori_fps": ori_fps,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="exported LeRobot task root")
    ap.add_argument("--model", required=True,
                    help="LingBot-VA checkpoint root holding vae/, text_encoder/, tokenizer/")
    ap.add_argument("--height", type=int, default=256, help="cam_high target height (va_robotwin_cfg)")
    ap.add_argument("--width", type=int, default=320, help="cam_high target width")
    ap.add_argument("--stride", type=int, default=4,
                    help="frame sampling stride; 4 matches the official dataset (50 -> 12.5 fps) "
                         "and pins action_per_frame = stride*4 = 16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--write-empty-emb", action="store_true",
                    help="also write <dataset>/empty_emb.pt (UMT5 of the empty string, for CFG)")
    ap.add_argument("--force", action="store_true", help="re-extract existing files")
    args = ap.parse_args()

    ds = Path(args.dataset)
    device, dtype = torch.device(args.device), torch.bfloat16
    if args.stride != 4:
        raise SystemExit(
            "this RoboTwin/RMBench config fixes action_per_frame=16, so latent stride must be 4"
        )
    if args.height % 16 or args.width % 16:
        raise SystemExit("--height and --width must both be divisible by the Wan VAE factor 16")
    info = json.loads((ds / "meta" / "info.json").read_text())
    ori_fps = info["fps"]
    with (ds / "meta" / "episodes.jsonl").open(encoding="utf-8") as handle:
        episodes = [json.loads(line) for line in handle if line.strip()]
    planned_segments = validate_extraction_plan(ds, info, episodes, args.stride)
    print(
        f"[latents] preflight passed: {len(episodes)} episodes / "
        f"{planned_segments} segments / {len(CAM_KEYS) * planned_segments} outputs",
        flush=True,
    )

    # Encode every unique prompt once, then release UMT5 before loading the VAE.
    # This lowers peak VRAM and avoids repeating the same task prompt 50 times.
    texts = sorted({
        segment["action_text"]
        for episode in episodes
        for segment in episode["action_config"]
    })
    print(
        f"[latents] loading text stack from {args.model} "
        f"({len(texts)} unique prompt(s))",
        flush=True,
    )
    tokenizer = load_tokenizer(os.path.join(args.model, "tokenizer"))
    text_encoder = load_text_encoder(os.path.join(args.model, "text_encoder"),
                                     torch_dtype=dtype, torch_device=device)
    text_encoder.eval()
    text_cache = {
        text: encode_text(tokenizer, text_encoder, text, device)
        for text in texts
    }

    if args.write_empty_emb:
        save_atomic(
            encode_text(tokenizer, text_encoder, "", device), ds / "empty_emb.pt"
        )
        print(f"[latents] wrote {ds / 'empty_emb.pt'}")

    del text_encoder, tokenizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"[latents] loading VAE from {args.model}", flush=True)
    vae = load_vae(os.path.join(args.model, "vae"), torch_dtype=dtype, torch_device=device)
    vae.eval()

    for cam in CAM_KEYS:
        (ds / "latents" / "chunk-000" / cam).mkdir(parents=True, exist_ok=True)

    for ep in episodes:
        idx = ep["episode_index"]
        for seg in ep["action_config"]:
            s, e, text = seg["start_frame"], seg["end_frame"], seg["action_text"]
            frame_ids = plan_frame_ids(s, e, args.stride)
            n = len(frame_ids)
            latent_T = 1 + (n - 1) // 4

            text_emb = text_cache[text]
            out_paths = [
                ds / "latents" / "chunk-000" / cam
                / f"episode_{idx:06d}_{s}_{e}.pth"
                for cam in CAM_KEYS
            ]

            # Head camera: independent full-resolution VAE stream.
            if args.force or not out_paths[0].exists():
                head_frames = read_video_frames(
                    ds / "videos" / "chunk-000" / CAM_KEYS[0]
                    / f"episode_{idx:06d}.mp4", frame_ids)
                head_lat = encode_clips(
                    vae, [head_frames], args.height, args.width, device, dtype
                )[0]
                assert head_lat.shape[1] == latent_T, (head_lat.shape[1], latent_T)
                save_atomic(
                    latent_payload(
                        head_lat, text_emb, text, frame_ids, s, e, ori_fps,
                        args.stride, args.height, args.width,
                    ),
                    out_paths[0],
                )

            # The server encodes left/right wrists together as one half-resolution
            # batch. Do the same here, then save the two batch items separately.
            wrist_needed = [args.force or not path.exists() for path in out_paths[1:]]
            if any(wrist_needed):
                wrist_frames = [
                    read_video_frames(
                        ds / "videos" / "chunk-000" / cam
                        / f"episode_{idx:06d}.mp4", frame_ids
                    )
                    for cam in CAM_KEYS[1:]
                ]
                wrist_latents = encode_clips(
                    vae, wrist_frames, args.height // 2, args.width // 2,
                    device, dtype,
                )
                assert wrist_latents.shape[2] == latent_T, (
                    wrist_latents.shape[2], latent_T
                )
                for wrist_i, needed in enumerate(wrist_needed):
                    if not needed:
                        continue
                    save_atomic(
                        latent_payload(
                            wrist_latents[wrist_i], text_emb, text, frame_ids,
                            s, e, ori_fps, args.stride,
                            args.height // 2, args.width // 2,
                        ),
                        out_paths[wrist_i + 1],
                    )
            print(f"  ep{idx:06d} [{s},{e})  sampled={n}  latent_frames={latent_T}")

    print("[latents] done")


if __name__ == "__main__":
    main()
