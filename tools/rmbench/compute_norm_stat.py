"""Compute loader-compatible action normalization statistics for RMBench.

LingBot-VA normalizes actions only after ``LatentLeRobotDataset`` converts the
two absolute end-effector poses to episode-relative poses and scatters the
dataset's 16 channels into its canonical 30-channel action space. This script
mirrors those operations exactly.

Only the six translation channels use empirical percentiles. Quaternion
channels remain pinned to [-1, 1] and grippers to [0, 1], matching the released
RoboTwin config. The stored quaternion four-vector is intentionally passed to
SciPy without reordering: that is what both LingBot-VA's training loader and its
RoboTwin evaluation client do.

For one dataset, ``norm_stat.json`` is written into the dataset root by default.
For joint statistics over several tasks, pass ``--dataset`` repeatedly and give
an explicit ``--out`` path.

Usage
-----
    python tools/rmbench/compute_norm_stat.py \
        --dataset /datasets/RMBench-data/lingbotva-rmbench/put_back_block

    python tools/rmbench/compute_norm_stat.py \
        --dataset /path/task_a --dataset /path/task_b \
        --out /path/joint_norm_stat.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R


ACTION_DIM = 30

# Must mirror va_robotwin_cfg.used_action_channel_ids. Source order is:
# left pose(7), left gripper, right pose(7), right gripper.
USED_ACTION_CHANNEL_IDS = (
    list(range(0, 7)) + [28] + list(range(7, 14)) + [29]
)
TRANSLATION_COLS = {0, 1, 2, 8, 9, 10}
QUATERNION_COLS = {3, 4, 5, 6, 11, 12, 13, 14}
GRIPPER_COLS = {7, 15}


def load_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def get_relative_pose(pose):
    """Mirror ``lerobot_latent_dataset.get_relative_pose`` without reordering."""
    rotations = R.from_quat(pose[:, 3:7])
    first_rotation = R.from_quat(np.tile(pose[:1, 3:7], (len(pose), 1)))
    relative_translation = pose[:, :3] - pose[:1, :3]
    relative_quaternion = (first_rotation.inv() * rotations).as_quat()
    return np.concatenate([relative_translation, relative_quaternion], axis=1)


def read_relative_actions(root):
    """Load exactly the episodes named by metadata and return 16-D actions."""
    episodes_path = root / "meta" / "episodes.jsonl"
    if not episodes_path.is_file():
        raise FileNotFoundError(f"missing {episodes_path}")

    episodes = load_jsonl(episodes_path)
    if not episodes:
        raise ValueError(f"{episodes_path} contains no episodes")

    chunks = []
    for episode in episodes:
        episode_index = episode["episode_index"]
        parquet_path = (
            root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        )
        if not parquet_path.is_file():
            raise FileNotFoundError(f"missing {parquet_path}")

        table = pq.read_table(parquet_path, columns=["action"])
        action = np.stack(
            table["action"].to_numpy(zero_copy_only=False)
        ).astype(np.float64)
        expected_length = episode.get("length")
        if action.shape != (expected_length, 16):
            raise ValueError(
                f"{parquet_path}: action shape {action.shape}, "
                f"expected ({expected_length}, 16)"
            )
        if not np.isfinite(action).all():
            raise ValueError(f"{parquet_path}: action contains NaN or infinity")

        relative = np.concatenate(
            [
                get_relative_pose(action[:, :7]),
                action[:, 7:8],
                get_relative_pose(action[:, 8:15]),
                action[:, 15:16],
            ],
            axis=1,
        )
        chunks.append(relative)

    return np.concatenate(chunks, axis=0), len(episodes)


def write_json_atomic(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        help="exported task root; repeat for joint statistics over several tasks",
    )
    parser.add_argument(
        "--out",
        help=(
            "output JSON (default: <dataset>/norm_stat.json for one dataset; "
            "required for several datasets)"
        ),
    )
    parser.add_argument("--q-low", type=float, default=1.0)
    parser.add_argument("--q-high", type=float, default=99.0)
    args = parser.parse_args()

    if not 0 <= args.q_low < args.q_high <= 100:
        raise SystemExit("require 0 <= --q-low < --q-high <= 100")

    roots = [Path(dataset) for dataset in args.dataset]
    chunks = []
    total_episodes = 0
    for root in roots:
        actions, episode_count = read_relative_actions(root)
        chunks.append(actions)
        total_episodes += episode_count
        print(
            f"[norm_stat] {root}: {episode_count} episodes / {len(actions)} frames",
            flush=True,
        )

    all_actions = np.concatenate(chunks, axis=0)
    empirical_low = np.percentile(all_actions, args.q_low, axis=0)
    empirical_high = np.percentile(all_actions, args.q_high, axis=0)
    degenerate_translation_columns = [
        column
        for column in sorted(TRANSLATION_COLS)
        if empirical_high[column] - empirical_low[column] <= 1e-6
    ]
    if degenerate_translation_columns:
        print(
            "[norm_stat] WARNING: near-zero percentile width on translation "
            f"columns {degenerate_translation_columns}. This can be valid for a "
            "stationary arm, but the loader will map that constant near -1; inspect "
            "the values before training.",
            flush=True,
        )

    print("\n[norm_stat] source-column percentiles:")
    for column in range(all_actions.shape[1]):
        if column in TRANSLATION_COLS:
            kind, used = "trans", "used"
        elif column in QUATERNION_COLS:
            kind, used = "quat", "pinned [-1,1]"
        else:
            kind, used = "grip", "pinned [0,1]"
        print(
            f"  col {column:2d} [{kind:5s}] "
            f"p{args.q_low:g}={empirical_low[column]: .6f} "
            f"p{args.q_high:g}={empirical_high[column]: .6f} "
            f"min={all_actions[:, column].min(): .6f} "
            f"max={all_actions[:, column].max(): .6f}  {used}"
        )

    q01 = np.zeros(ACTION_DIM, dtype=np.float64)
    q99 = np.zeros(ACTION_DIM, dtype=np.float64)
    for source_column, canonical_column in enumerate(USED_ACTION_CHANNEL_IDS):
        if source_column in TRANSLATION_COLS:
            q01[canonical_column] = empirical_low[source_column]
            q99[canonical_column] = empirical_high[source_column]
        elif source_column in QUATERNION_COLS:
            q01[canonical_column] = -1.0
            q99[canonical_column] = 1.0
        elif source_column in GRIPPER_COLS:
            q01[canonical_column] = 0.0
            q99[canonical_column] = 1.0
        else:
            raise AssertionError(f"unclassified source action column {source_column}")

    # Measure exactly the 16 channels consumed by the loader. Including the 14
    # unused, zero-filled canonical channels would dilute this diagnostic.
    used_low = q01[USED_ACTION_CHANNEL_IDS]
    used_high = q99[USED_ACTION_CHANNEL_IDS]
    normalized = (
        (all_actions - used_low) / (used_high - used_low + 1e-6) * 2.0 - 1.0
    )
    clipped_fraction = float(np.mean(np.abs(normalized) > 1.5))
    print(
        "\n[norm_stat] after loader normalization: "
        f"min={normalized.min():.3f} max={normalized.max():.3f}; "
        f"|x|>1.5={clipped_fraction:.3%}"
    )

    q01_list = q01.tolist()
    q99_list = q99.tolist()
    print("\n[norm_stat] paste into the task config:\n")
    print("va_rmbench_cfg.action_norm_method = 'quantiles'")
    print("va_rmbench_cfg.norm_stat = {")
    print(f'    "q01": {json.dumps(q01_list)},')
    print(f'    "q99": {json.dumps(q99_list)},')
    print("}")

    if args.out:
        output_path = Path(args.out)
    elif len(roots) == 1:
        output_path = roots[0] / "norm_stat.json"
    else:
        raise SystemExit(
            "--out is required for multiple datasets because joint statistics "
            "do not belong to one task directory"
        )

    write_json_atomic(
        {
            "q01": q01_list,
            "q99": q99_list,
            "datasets": [str(root) for root in roots],
            "episodes": total_episodes,
            "frames": int(len(all_actions)),
            "q_low_percentile": args.q_low,
            "q_high_percentile": args.q_high,
            "clipped_fraction": clipped_fraction,
        },
        output_path,
    )
    print(f"\n[norm_stat] wrote {output_path}")


if __name__ == "__main__":
    main()
