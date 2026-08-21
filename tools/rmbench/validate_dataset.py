"""Validate an RMBench task before LingBot-VA post-training.

This is deliberately read-only. It checks the LeRobot v2.1 files, all three
camera videos, the Wan latent payloads, temporal alignment, and empty CFG text
embedding. It does not initialize the transformer or write anything.

Usage:
    python tools/rmbench/validate_dataset.py \
        --dataset /datasets/RMBench-data/lingbotva-rmbench/put_back_block \
        --expect-episodes 50
"""

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch


CAM_KEYS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
ACTION_NAMES = (
    [f"left_{name}" for name in ("x", "y", "z", "q1", "q2", "q3", "q4")]
    + ["left_gripper"]
    + [f"right_{name}" for name in ("x", "y", "z", "q1", "q2", "q3", "q4")]
    + ["right_gripper"]
)
USED_ACTION_CHANNEL_IDS = list(range(0, 7)) + [28] + list(range(7, 14)) + [29]
CANONICAL_QUATERNION_IDS = {3, 4, 5, 6, 10, 11, 12, 13}
CANONICAL_GRIPPER_IDS = {28, 29}
AV1_CODEC_ID = av.codec.Codec("av1", "r").id
LATENT_HW = {
    "observation.images.cam_high": (16, 20),
    "observation.images.cam_left_wrist": (8, 10),
    "observation.images.cam_right_wrist": (8, 10),
}
REQUIRED_LATENT_KEYS = {
    "latent", "latent_num_frames", "latent_height", "latent_width",
    "video_num_frames", "video_height", "video_width", "text_emb", "text",
    "frame_ids", "start_frame", "end_frame", "fps", "ori_fps",
}


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fail(message):
    raise SystemExit(f"VALIDATION FAILED: {message}")


def planned_frame_ids(start, end, stride=4):
    frame_ids = list(range(start, end, stride))
    keep = len(frame_ids) - ((len(frame_ids) - 1) % 4)
    return frame_ids[:keep]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--expect-episodes", type=int)
    args = parser.parse_args()

    root = Path(args.dataset)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    episodes = load_jsonl(root / "meta" / "episodes.jsonl")
    tasks = load_jsonl(root / "meta" / "tasks.jsonl")
    stats = load_jsonl(root / "meta" / "episodes_stats.jsonl")

    if info.get("codebase_version") != "v2.1":
        fail(f"codebase_version={info.get('codebase_version')!r}, expected 'v2.1'")
    if args.expect_episodes is not None and len(episodes) != args.expect_episodes:
        fail(f"found {len(episodes)} episodes, expected {args.expect_episodes}")
    if len(stats) != len(episodes):
        fail(f"episodes_stats has {len(stats)} rows for {len(episodes)} episodes")
    if [ep["episode_index"] for ep in episodes] != list(range(len(episodes))):
        fail("episode indices are not contiguous from zero")
    if info.get("total_episodes") != len(episodes):
        fail("info.total_episodes disagrees with episodes.jsonl")
    if info.get("total_tasks") != len(tasks):
        fail("info.total_tasks disagrees with tasks.jsonl")

    action_feature = info.get("features", {}).get("action", {})
    state_feature = info.get("features", {}).get("observation.state", {})
    if (
        action_feature.get("dtype") != "float32"
        or action_feature.get("shape") != [16]
        or action_feature.get("names") != [ACTION_NAMES]
    ):
        fail(f"action feature is {action_feature}, expected float32[16]")
    if (
        state_feature.get("dtype") != "float32"
        or state_feature.get("shape") != [16]
        or state_feature.get("names") != [ACTION_NAMES]
    ):
        fail(f"observation.state feature is {state_feature}, expected float32[16]")
    for camera in CAM_KEYS:
        if info.get("features", {}).get(camera, {}).get("dtype") != "video":
            fail(f"missing video feature {camera}")

    empty_path = root / "empty_emb.pt"
    if not empty_path.is_file():
        fail(f"missing {empty_path}")
    empty = torch.load(empty_path, map_location="cpu", weights_only=False)
    if (
        tuple(empty.shape) != (512, 4096)
        or empty.dtype != torch.bfloat16
        or not torch.isfinite(empty).all()
    ):
        fail(f"empty_emb is shape={tuple(empty.shape)} dtype={empty.dtype}")

    norm_path = root / "norm_stat.json"
    if not norm_path.is_file():
        fail(f"missing {norm_path}; run compute_norm_stat.py before training")
    norm = json.loads(norm_path.read_text(encoding="utf-8"))
    norm_datasets = norm.get("datasets")
    if not isinstance(norm_datasets, list) or len(norm_datasets) != 1:
        fail(f"{norm_path}: expected provenance for exactly one task dataset")
    if Path(norm_datasets[0]).resolve() != root.resolve():
        fail(
            f"{norm_path}: computed for {norm_datasets[0]}, not the validated "
            f"dataset {root}"
        )
    q01 = np.asarray(norm.get("q01"), dtype=np.float64)
    q99 = np.asarray(norm.get("q99"), dtype=np.float64)
    if q01.shape != (30,) or q99.shape != (30,):
        fail(f"{norm_path}: q01/q99 must both have 30 values")
    if not np.isfinite(q01).all() or not np.isfinite(q99).all():
        fail(f"{norm_path}: q01/q99 contains NaN or infinity")
    unused_ids = sorted(set(range(30)) - set(USED_ACTION_CHANNEL_IDS))
    if np.any(q01[unused_ids] != 0) or np.any(q99[unused_ids] != 0):
        fail(f"{norm_path}: unused canonical channels must remain zero")
    for channel in CANONICAL_QUATERNION_IDS:
        if (q01[channel], q99[channel]) != (-1.0, 1.0):
            fail(f"{norm_path}: quaternion channel {channel} must be [-1,1]")
    for channel in CANONICAL_GRIPPER_IDS:
        if (q01[channel], q99[channel]) != (0.0, 1.0):
            fail(f"{norm_path}: gripper channel {channel} must be [0,1]")
    if np.any(q99[USED_ACTION_CHANNEL_IDS] < q01[USED_ACTION_CHANNEL_IDS]):
        fail(f"{norm_path}: a used channel has q99 below q01")
    if norm.get("episodes") not in (None, len(episodes)):
        fail(f"{norm_path}: episode count disagrees with dataset metadata")

    total_frames = 0
    latent_files = 0
    latent_frame_counts = []
    for episode in episodes:
        episode_index = episode["episode_index"]
        length = episode["length"]
        total_frames += length

        parquet = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        table = pq.read_table(parquet, columns=["observation.state", "action"])
        if table.num_rows != length:
            fail(f"{parquet}: {table.num_rows} rows, metadata says {length}")
        state = np.stack(table["observation.state"].to_numpy()).astype(np.float32)
        action = np.stack(table["action"].to_numpy()).astype(np.float32)
        if state.shape != (length, 16) or not np.isfinite(state).all():
            fail(f"{parquet}: invalid observation.state shape/values {state.shape}")
        if action.shape != (length, 16) or not np.isfinite(action).all():
            fail(f"{parquet}: invalid action shape/values {action.shape}")
        if not np.array_equal(action[:-1], state[1:]):
            fail(f"{parquet}: action[t] is not exactly observation.state[t+1]")

        for camera in CAM_KEYS:
            video = root / "videos" / "chunk-000" / camera / f"episode_{episode_index:06d}.mp4"
            try:
                with av.open(str(video), "r") as container:
                    stream = container.streams.video[0]
                    codec_context = stream.codec_context
                    codec = codec_context.codec
                    actual = {
                        "codec_id": codec.id,
                        "codec_name": codec.name,
                        "pix_fmt": codec_context.pix_fmt,
                        "fps": float(stream.average_rate),
                        "height": codec_context.height,
                        "width": codec_context.width,
                    }
                    keyframes = []
                    video_frames = 0
                    for video_frames, frame in enumerate(
                        container.decode(video=0), start=1
                    ):
                        if frame.key_frame:
                            keyframes.append(video_frames - 1)
            except (av.error.FFmpegError, IndexError) as error:
                fail(f"cannot decode {video}: {error}")

            expected = {
                "codec_id": AV1_CODEC_ID, "pix_fmt": "yuv420p", "fps": 50.0,
                "height": 480, "width": 640,
            }
            comparable = {key: actual[key] for key in expected}
            if comparable != expected:
                fail(f"{video}: stream properties {actual}, expected {expected}")
            if video_frames != length:
                fail(f"{video}: {video_frames} frames, expected {length}")
            expected_keyframes = list(range(0, length, 2))
            if keyframes != expected_keyframes:
                fail(
                    f"{video}: AV1 keyframes do not follow GOP 2; first actual="
                    f"{keyframes[:10]}, expected={expected_keyframes[:10]}"
                )

            feature = info["features"][camera]
            feature_info = feature["info"]
            if (
                feature["shape"] != [3, 480, 640]
                or feature_info["video.codec"] != "av1"
                or feature_info["video.pix_fmt"] != "yuv420p"
                or feature_info["video.fps"] != 50
                or feature_info["video.height"] != 480
                or feature_info["video.width"] != 640
            ):
                fail(f"{camera}: video feature metadata is inconsistent: {feature}")

        configs = episode.get("action_config")
        if not isinstance(configs, list) or not configs:
            fail(f"episode {episode_index}: missing action_config")
        for segment in configs:
            start, end = segment["start_frame"], segment["end_frame"]
            if not (0 <= start < end <= length):
                fail(f"episode {episode_index}: invalid segment [{start},{end})")
            expected_frame_ids = planned_frame_ids(start, end)
            if len(expected_frame_ids) < 5:
                fail(
                    f"episode {episode_index} [{start},{end}): only "
                    f"{len(expected_frame_ids)} compatible sampled frames"
                )
            reference_text_embedding = None
            for camera in CAM_KEYS:
                path = (
                    root / "latents" / "chunk-000" / camera
                    / f"episode_{episode_index:06d}_{start}_{end}.pth"
                )
                if not path.is_file():
                    fail(f"missing {path}")
                payload = torch.load(path, map_location="cpu", weights_only=False)
                missing = REQUIRED_LATENT_KEYS - set(payload)
                if missing:
                    fail(f"{path}: missing keys {sorted(missing)}")

                frame_ids = list(payload["frame_ids"])
                if frame_ids != expected_frame_ids:
                    fail(
                        f"{path}: frame_ids differ from the exact stride-4, "
                        "maximal 4k+1 extraction plan"
                    )
                if payload["start_frame"] != start or payload["end_frame"] != end:
                    fail(f"{path}: stored segment bounds disagree with action_config")

                latent_frames = (len(frame_ids) - 1) // 4 + 1
                height, width = LATENT_HW[camera]
                latent = payload["latent"]
                expected_shape = (latent_frames * height * width, 48)
                if (
                    tuple(latent.shape) != expected_shape
                    or latent.dtype != torch.bfloat16
                    or not torch.isfinite(latent).all()
                ):
                    fail(
                        f"{path}: latent shape/dtype {tuple(latent.shape)}/{latent.dtype}, "
                        f"expected {expected_shape}/bfloat16"
                    )
                if (
                    payload["latent_num_frames"] != latent_frames
                    or payload["latent_height"] != height
                    or payload["latent_width"] != width
                ):
                    fail(f"{path}: latent metadata disagrees with tensor")
                expected_video_hw = (256, 320) if camera == CAM_KEYS[0] else (128, 160)
                if (
                    payload["video_num_frames"] != len(frame_ids)
                    or (payload["video_height"], payload["video_width"]) != expected_video_hw
                    or payload["ori_fps"] != 50
                    or payload["fps"] != 12
                ):
                    fail(f"{path}: sampled-video metadata is inconsistent")
                text_emb = payload["text_emb"]
                if (
                    tuple(text_emb.shape) != (512, 4096)
                    or text_emb.dtype != torch.bfloat16
                    or not torch.isfinite(text_emb).all()
                ):
                    fail(f"{path}: invalid text_emb {tuple(text_emb.shape)}/{text_emb.dtype}")
                if reference_text_embedding is None:
                    reference_text_embedding = text_emb
                elif not torch.equal(text_emb, reference_text_embedding):
                    fail(f"{path}: text_emb differs across cameras for one segment")
                if payload["text"] != segment["action_text"]:
                    fail(f"{path}: text disagrees with action_config")
                latent_files += 1
                latent_frame_counts.append(latent_frames)

    if info.get("total_frames") != total_frames:
        fail(f"info.total_frames={info.get('total_frames')}, summed episodes={total_frames}")
    if norm.get("frames") not in (None, total_frames):
        fail(f"{norm_path}: frame count disagrees with dataset metadata")
    if info.get("total_videos") != len(episodes) * len(CAM_KEYS):
        fail(
            f"info.total_videos={info.get('total_videos')}, expected "
            f"{len(episodes) * len(CAM_KEYS)}"
        )

    expected_latents = sum(len(ep["action_config"]) for ep in episodes) * len(CAM_KEYS)
    if latent_files != expected_latents:
        fail(f"validated {latent_files} latent files, expected {expected_latents}")
    actual_latents = len(list((root / "latents").rglob("*.pth")))
    actual_parquets = len(list((root / "data").rglob("*.parquet")))
    actual_videos = len(list((root / "videos").rglob("*.mp4")))
    if actual_latents != expected_latents:
        fail(f"found {actual_latents} latent files on disk, expected {expected_latents}")
    if actual_parquets != len(episodes):
        fail(f"found {actual_parquets} parquet files, expected {len(episodes)}")
    if actual_videos != len(episodes) * len(CAM_KEYS):
        fail(f"found {actual_videos} videos, expected {len(episodes) * len(CAM_KEYS)}")

    print(
        "VALID: "
        f"episodes={len(episodes)} frames={total_frames} videos={len(episodes) * 3} "
        f"latents={latent_files} latent_frames=[{min(latent_frame_counts)},"
        f"{max(latent_frame_counts)}] empty_emb={tuple(empty.shape)}"
    )


if __name__ == "__main__":
    main()
