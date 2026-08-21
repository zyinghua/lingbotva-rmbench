"""Convert one raw RMBench task (demo_clean) into the LeRobot v2.1 layout LingBot-VA trains on.

Every format decision below is derived from two verifiable sources:

  * the loader in this repo -- wan_va/dataset/lerobot_latent_dataset.py reads
    meta/{info.json,episodes.jsonl}, per-episode parquet (column 'action' only),
    and requires an ``action_config`` list per episode (parse_meta);
    _action_post_process for env_type='robotwin_tshape' slices the action as
    [:, :7] left pose / [:, 7:8] left gripper / [:, 8:15] right pose /
    [:, 15:16] right gripper  -> the exported action MUST be
    [left_xyz+quat(7), left_grip(1), right_xyz+quat(7), right_grip(1)] = 16 dims.
  * the officially released dataset robbyant/robotwin-clean-and-aug-lerobot
    (inspected file by file): fps=50, codebase v2.1, per-camera mp4s under
    videos/chunk-000/<cam>/episode_{idx:06d}.mp4, observation.state = the same
    16-dim EEF stack in absolute coordinates, episodes_stats.jsonl with
    min/max/mean/std/count per feature (image stats over a frame subsample,
    shaped [3,1,1] in [0,1]).

Raw RMBench episodes (data/<task>/demo_clean/data/episodeN.hdf5) carry everything
needed: endpose/{left,right}_endpose (T,7 xyz+quat), endpose/{left,right}_gripper
(T,), joint_action/vector (T,14), and per-camera JPEG streams under
observation/{head,left,right}_camera/rgb. The natural-language condition comes
from ``instructions/episodeN.json`` (field ``seen``), when that generated file is
present. A task-level RMBench instruction JSON or a literal instruction can be
passed as an explicit fallback.

Camera name mapping (RoboTwin convention, order = va_robotwin_cfg.obs_cam_keys):
    observation.images.cam_high        <- observation/head_camera
    observation.images.cam_left_wrist  <- observation/left_camera
    observation.images.cam_right_wrist <- observation/right_camera

As in the released RoboTwin LeRobot data, exported row ``t`` contains the image
and EEF state from raw frame ``t`` and the action target from raw frame ``t+1``.
The RoboTwin-derived RMBench conversion drops the terminal observation and the
initial action, so each raw episode becomes ``raw_length - 1`` aligned rows.

Usage
-----
    python tools/rmbench/raw_to_lerobot.py \
        --raw-root /datasets/RMBench-data/data \
        --task put_back_block \
        --out  /datasets/RMBench-data/lingbotva-rmbench/put_back_block \
        --instruction-file RMBench/description/task_instruction/put_back_block.json
"""

import argparse
import json
from pathlib import Path

import av
import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CAM_MAP = {
    "observation.images.cam_high": "head_camera",
    "observation.images.cam_left_wrist": "left_camera",
    "observation.images.cam_right_wrist": "right_camera",
}
VIDEO_HEIGHT = 480
VIDEO_WIDTH = 640
VIDEO_CODEC = "libsvtav1"
VIDEO_PIX_FMT = "yuv420p"
VIDEO_OPTIONS = {"g": "2", "crf": "30"}
AV1_CODEC_ID = av.codec.Codec("av1", "r").id
ACTION_NAMES = (
    [f"left_{c}" for c in ("x", "y", "z", "q1", "q2", "q3", "q4")] + ["left_gripper"]
    + [f"right_{c}" for c in ("x", "y", "z", "q1", "q2", "q3", "q4")] + ["right_gripper"]
)


def _first_instruction(path, instruction_type):
    payload = json.loads(path.read_text(encoding="utf-8"))
    choices = payload.get(instruction_type)
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], str):
        raise ValueError(
            f"{path}: expected a non-empty string list at key {instruction_type!r}"
        )
    return choices[0].strip()


def episode_instruction(src, raw_idx, args):
    """Resolve the full-episode prompt, preferring episode-specific RMBench text."""
    if args.instruction:
        return args.instruction.strip()

    episode_path = src / "instructions" / f"episode{raw_idx}.json"
    if episode_path.is_file():
        return _first_instruction(episode_path, args.instruction_type)

    if args.instruction_file:
        return _first_instruction(Path(args.instruction_file), args.instruction_type)

    raise FileNotFoundError(
        f"missing {episode_path}. Pass --instruction-file pointing to "
        f"RMBench/description/task_instruction/{args.task}.json, or pass --instruction. "
        "language_annotation.json contains low-level subtask durations, not the "
        "full instruction used by LingBot-VA evaluation."
    )


def decode_rgb(encoded, source):
    """Decode one RMBench byte payload, preserving its legacy RGB channel order.

    RMBench passes simulator RGB arrays directly to ``cv2.imencode``. Although
    OpenCV calls the decoded array BGR, its numeric channel order here is still
    the original RGB. Treating it as ordinary BGR would swap red and blue.
    """
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode RGB bytes from {source}")
    return image


def feat_stats(arr):
    """min/max/mean/std/count in the shape episodes_stats.jsonl uses."""
    a = np.asarray(arr, dtype=np.float64).reshape(len(arr), -1)
    return {
        "min": a.min(0).tolist(), "max": a.max(0).tolist(),
        "mean": a.mean(0).tolist(), "std": a.std(0).tolist(), "count": [len(a)],
    }


def img_stats(frames):
    """Per-channel stats over a subsample of decoded frames, values in [0,1], shape [3,1,1]."""
    a = np.stack(frames).astype(np.float64) / 255.0          # (n, H, W, 3)
    def per(function):
        return [[[float(value)]] for value in function(a, axis=(0, 1, 2))]

    return {"min": per(np.min), "max": per(np.max), "mean": per(np.mean),
            "std": per(np.std), "count": [len(frames)]}


def image_stat_indices(length):
    """Mirror LeRobot v0.3.3 compute_stats.sample_indices."""
    min_samples = min(100, length)
    num_samples = max(min_samples, min(int(length ** 0.75), 10_000))
    return set(np.round(np.linspace(0, length - 1, num_samples)).astype(int).tolist())


def downsample_stat_image(image, target_size=150, max_size_threshold=300):
    """Mirror LeRobot's inexpensive spatial subsampling for image statistics."""
    height, width = image.shape[:2]
    if max(width, height) < max_size_threshold:
        return image
    factor = int(width / target_size) if width > height else int(height / target_size)
    return image[::factor, ::factor]


def encode_av1_video(video_path, rgb_frames, fps):
    """Encode RGB frames with LeRobot v0.3.3's default AV1 settings.

    LeRobot's reference path writes RGB PNGs and then calls PyAV with
    libsvtav1/yuv420p, GOP 2 and CRF 30. Feeding the same RGB arrays directly
    to PyAV is equivalent and avoids a temporary image directory.
    """
    tmp_path = video_path.with_name(f"{video_path.stem}.tmp{video_path.suffix}")
    with av.open(str(tmp_path), "w") as output:
        stream = output.add_stream(
            VIDEO_CODEC,
            fps,
            options=VIDEO_OPTIONS,
        )
        stream.pix_fmt = VIDEO_PIX_FMT
        stream.width = VIDEO_WIDTH
        stream.height = VIDEO_HEIGHT

        for rgb in rgb_frames:
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)

    if not tmp_path.is_file() or tmp_path.stat().st_size == 0:
        raise RuntimeError(f"AV1 encoder produced no data at {tmp_path}")
    tmp_path.replace(video_path)


def inspect_video(video_path):
    """Return decoded stream properties and frame count using LeRobot's PyAV stack."""
    with av.open(str(video_path), "r") as container:
        stream = container.streams.video[0]
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
        keyframes = []
        frame_count = 0
        for frame_count, frame in enumerate(container.decode(video=0), start=1):
            if frame.key_frame:
                keyframes.append(frame_count - 1)
        properties["frames"] = frame_count
        properties["keyframes"] = keyframes
    return properties


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", required=True,
                    help="RMBench data root holding <task>/<config>/data/episodeN.hdf5")
    ap.add_argument("--task", required=True)
    ap.add_argument("--config", default="demo_clean")
    ap.add_argument("--out", required=True, help="output LeRobot dataset root")
    ap.add_argument("--fps", type=int, default=50,
                    help="recording fps; 50 for the RoboTwin 2.0 platform (official dataset: fps=50)")
    ap.add_argument("--instruction-type", default="seen", choices=("seen", "unseen"),
                    help="which list to use in RMBench instruction JSON files")
    prompt = ap.add_mutually_exclusive_group()
    prompt.add_argument("--instruction-file",
                        help="task-level RMBench JSON used only when per-episode instructions are absent")
    prompt.add_argument("--instruction",
                        help="literal full-episode instruction used for every episode")
    ap.add_argument("--limit", type=int, default=None, help="first N episodes only (smoke test)")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing files in an already exported dataset")
    args = ap.parse_args()

    src = Path(args.raw_root) / args.task / args.config
    out = Path(args.out)
    eps = sorted((src / "data").glob("episode*.hdf5"),
                 key=lambda p: int(p.stem.replace("episode", "")))
    if args.limit:
        eps = eps[: args.limit]
    if not eps:
        raise SystemExit(f"no episodes under {src}/data")
    known_output_files = (
        list((out / "meta").glob("*"))
        + list((out / "data").rglob("*.parquet"))
        + list((out / "videos").rglob("*.mp4"))
        + list((out / "latents").rglob("*.pth"))
        + [path for path in (out / "empty_emb.pt", out / "norm_stat.json") if path.exists()]
    )
    if known_output_files and not args.overwrite:
        raise SystemExit(
            f"{out} already contains dataset files; pass --overwrite to replace "
            "its known export files, or use a fresh --out directory"
        )
    if args.overwrite:
        expected = {f"episode_{i:06d}" for i in range(len(eps))}
        stale = [
            p for p in (out / "data" / "chunk-000").glob("episode_*.parquet")
            if p.stem not in expected
        ]
        for cam in CAM_MAP:
            stale.extend(
                p for p in (out / "videos" / "chunk-000" / cam).glob("episode_*.mp4")
                if p.stem not in expected
            )
        stale_derived = list((out / "latents").rglob("*.pth"))
        if (out / "norm_stat.json").exists():
            stale_derived.append(out / "norm_stat.json")
        if stale or stale_derived:
            examples = stale[:3] + stale_derived[:3]
            raise SystemExit(
                "refusing to mix a new export with stale episode or derived files. "
                f"Use a fresh --out directory. Examples: {examples}"
            )
    print(
        f"[raw2lerobot] {args.task}: {len(eps)} episodes from {src}",
        flush=True,
    )

    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for cam in CAM_MAP:
        (out / "videos" / "chunk-000" / cam).mkdir(parents=True, exist_ok=True)

    episodes, stats_rows, instructions = [], [], {}
    running_index = total_frames = 0
    source_video_hw = {}
    video_hw = {cam: (VIDEO_HEIGHT, VIDEO_WIDTH) for cam in CAM_MAP}

    for new_idx, h5_path in enumerate(eps):
        raw_idx = int(h5_path.stem.replace("episode", ""))
        instruction = episode_instruction(src, raw_idx, args)
        if not instruction:
            raise ValueError(f"episode {raw_idx}: empty instruction")
        task_index = instructions.setdefault(instruction, len(instructions))

        with h5py.File(h5_path, "r") as f:
            lp = np.asarray(f["endpose/left_endpose"], dtype=np.float32)     # (T,7)
            lg = np.asarray(f["endpose/left_gripper"], dtype=np.float32)     # (T,)
            rp = np.asarray(f["endpose/right_endpose"], dtype=np.float32)
            rg = np.asarray(f["endpose/right_gripper"], dtype=np.float32)
            eef = np.concatenate([lp, lg[:, None], rp, rg[:, None]], axis=1)  # (raw_T,16)
            raw_T = eef.shape[0]
            lengths = {
                "left_endpose": len(lp), "left_gripper": len(lg),
                "right_endpose": len(rp), "right_gripper": len(rg),
            }
            if set(lengths.values()) != {raw_T}:
                raise ValueError(f"{h5_path}: endpose length mismatch: {lengths}")
            if raw_T < 2:
                raise ValueError(f"{h5_path}: need at least 2 raw frames, got {raw_T}")
            if not np.isfinite(eef).all():
                raise ValueError(f"{h5_path}: action contains NaN or infinity")

            # Match the released RoboTwin LeRobot alignment: row t observes
            # raw frame t and supervises the EEF target at raw frame t+1.
            state = eef[:-1]
            action = eef[1:]
            T = raw_T - 1

            ep_img_stats = {}
            for cam, h5cam in CAM_MAP.items():
                jpegs = f[f"observation/{h5cam}/rgb"]
                if len(jpegs) != raw_T:
                    raise ValueError(
                        f"{h5_path}: {h5cam} has {len(jpegs)} frames, expected {raw_T}"
                    )
                first = decode_rgb(jpegs[0], f"{h5_path}:{h5cam}:0")
                source_h, source_w = first.shape[:2]
                previous_hw = source_video_hw.setdefault(cam, (source_h, source_w))
                if previous_hw != (source_h, source_w):
                    raise ValueError(
                        f"{h5_path}: {cam} resolution {(source_h, source_w)} differs from {previous_hw}"
                    )
                vp = out / "videos" / "chunk-000" / cam / f"episode_{new_idx:06d}.mp4"
                sampled = []
                stat_ids = image_stat_indices(T)
                def resized_rgb_frames():
                    for i in range(T):
                        img = first if i == 0 else decode_rgb(
                            jpegs[i], f"{h5_path}:{h5cam}:{i}"
                        )
                        if img.shape[:2] != (source_h, source_w):
                            raise ValueError(
                                f"{h5_path}:{h5cam}:{i} has resolution {img.shape[:2]}, "
                                f"expected {(source_h, source_w)}"
                            )
                        # Match RoboTwin's conversion before LeRobot encoding.
                        # ``img`` is already legacy RGB (see decode_rgb), and
                        # PyAV receives it explicitly as rgb24: do not BGR-swap.
                        resized = cv2.resize(img, (VIDEO_WIDTH, VIDEO_HEIGHT))
                        if i in stat_ids:
                            sampled.append(downsample_stat_image(resized))
                        yield resized

                print(
                    f"  ep{raw_idx:>3} -> {new_idx:06d}  {cam}: "
                    f"encoding {T} AV1 frames",
                    flush=True,
                )
                encode_av1_video(vp, resized_rgb_frames(), args.fps)
                if not vp.is_file() or vp.stat().st_size == 0:
                    raise RuntimeError(f"AV1 encoder produced no data at {vp}")
                actual = inspect_video(vp)
                expected = {
                    "codec_id": AV1_CODEC_ID,
                    "pix_fmt": VIDEO_PIX_FMT,
                    "fps": float(args.fps),
                    "height": VIDEO_HEIGHT,
                    "width": VIDEO_WIDTH,
                    "frames": T,
                }
                comparable = {key: actual[key] for key in expected}
                if comparable != expected:
                    raise RuntimeError(
                        f"{vp}: encoded stream properties {actual}, expected {expected}"
                    )
                expected_keyframes = list(range(0, T, 2))
                if actual["keyframes"] != expected_keyframes:
                    raise RuntimeError(
                        f"{vp}: AV1 keyframes do not follow GOP 2; first actual="
                        f"{actual['keyframes'][:10]}, expected={expected_keyframes[:10]}"
                    )
                ep_img_stats[cam] = img_stats(sampled)

        timestamp = (np.arange(T) / args.fps).astype(np.float32)
        frame_index = np.arange(T, dtype=np.int64)

        table = pa.table({
            "observation.state": pa.FixedSizeListArray.from_arrays(
                pa.array(state.reshape(-1), type=pa.float32()), 16),
            "action": pa.FixedSizeListArray.from_arrays(
                pa.array(action.reshape(-1), type=pa.float32()), 16),
            "timestamp": pa.array(timestamp, type=pa.float32()),
            "frame_index": pa.array(frame_index),
            "episode_index": pa.array(np.full(T, new_idx, dtype=np.int64)),
            "index": pa.array(np.arange(running_index, running_index + T, dtype=np.int64)),
            "task_index": pa.array(np.full(T, task_index, dtype=np.int64)),
        })
        pq.write_table(table, out / "data" / "chunk-000" / f"episode_{new_idx:06d}.parquet")

        episodes.append({
            "episode_index": new_idx,
            "tasks": [instruction],
            "length": T,
            # required by parse_meta(); one segment per episode, no sub-task annotation
            "action_config": [
                {"start_frame": 0, "end_frame": T,
                 "action_text": instruction, "skill": ""}],
            "source_episode": h5_path.name,
        })
        st = {"observation.state": feat_stats(state), "action": feat_stats(action),
              "timestamp": feat_stats(timestamp[:, None]),
              "frame_index": feat_stats(frame_index[:, None]),
              "episode_index": feat_stats(np.full((T, 1), new_idx)),
              "index": feat_stats(np.arange(running_index, running_index + T)[:, None]),
              "task_index": feat_stats(np.full((T, 1), task_index))}
        st.update(ep_img_stats)
        stats_rows.append({"episode_index": new_idx, "stats": st})

        running_index += T
        total_frames += T
        print(f"  ep{raw_idx:>3} -> {new_idx:06d}  complete frames={T}", flush=True)

    vids = {cam: {
        "dtype": "video", "shape": [3, hw[0], hw[1]], "names": ["channels", "height", "width"],
        "info": {"video.fps": args.fps, "video.height": hw[0], "video.width": hw[1],
                 "video.channels": 3, "video.codec": "av1", "video.pix_fmt": VIDEO_PIX_FMT,
                 "video.is_depth_map": False, "has_audio": False},
    } for cam, hw in video_hw.items()}
    info = {
        "codebase_version": "v2.1", "robot_type": "aloha", "fps": args.fps,
        "total_episodes": len(episodes), "total_frames": total_frames,
        "total_tasks": len(instructions), "total_videos": len(episodes) * len(CAM_MAP),
        "total_chunks": 1, "chunks_size": 1000,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [16], "names": [ACTION_NAMES]},
            "action": {"dtype": "float32", "shape": [16], "names": [ACTION_NAMES]},
            **vids,
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    with open(out / "meta" / "episodes.jsonl", "w") as fh:
        for e in episodes:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(out / "meta" / "episodes_stats.jsonl", "w") as fh:
        for s in stats_rows:
            fh.write(json.dumps(s) + "\n")
    with open(out / "meta" / "tasks.jsonl", "w") as fh:
        for text, idx in sorted(instructions.items(), key=lambda kv: kv[1]):
            fh.write(json.dumps({"task_index": idx, "task": text}, ensure_ascii=False) + "\n")

    print(f"[raw2lerobot] wrote {len(episodes)} episodes / {total_frames} frames -> {out}")


if __name__ == "__main__":
    main()
