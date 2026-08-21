"""Compute the action normalization quantiles (norm_stat) for an exported RMBench task.

What must be replicated -- and why, from this repo's loader
(wan_va/dataset/lerobot_latent_dataset.py, _action_post_process,
env_type='robotwin_tshape'):

  1. The 16-dim parquet action is converted to EPISODE-RELATIVE pose first:
     get_relative_pose subtracts the first frame's translation and composes
     rotations with the inverse of the first frame's rotation
     (scipy R.from_quat, i.e. (x,y,z,w) ordering). Quantiles must therefore be
     computed on the post-conversion values, not the raw poses.
  2. The 16 columns are scattered into the canonical 30-dim action space via
     config.inverse_used_action_channel_ids. With va_robotwin_cfg's
     used_action_channel_ids = [0..6, 28, 7..13, 29]:
         left rel pose  -> canonical 0..6    left gripper  -> canonical 28
         right rel pose -> canonical 7..13   right gripper -> canonical 29
  3. Normalization: x -> 2*(x - q01)/(q99 - q01 + 1e-6) - 1, clipped to +-1.5.

Following the convention visible in va_robotwin_cfg.norm_stat: only translation
channels get empirical q01/q99; quaternion components are pinned to [-1, 1]
(mathematically bounded -- an empirical quantile would shrink the range and let
the model denormalise onto invalid rotations) and grippers to [0, 1]. The
empirical values for those channels are printed anyway so the assumption can be
checked instead of trusted.

Usage
-----
    python tools/rmbench/compute_norm_stat.py --dataset /datasets/lingbot-va-rmbench/put_back_block
    python tools/rmbench/compute_norm_stat.py --dataset A --dataset B   # joint stats for co-training
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

USED_IDS = list(range(0, 7)) + [28] + list(range(7, 14)) + [29]   # va_robotwin_cfg
TRANS = {0, 1, 2, 7, 8, 9}          # canonical translation channels (empirical quantiles)
QUAT = {3, 4, 5, 6, 10, 11, 12, 13}  # pinned [-1, 1]
GRIP = {28, 29}                      # pinned [0, 1]


def relative_pose(pose):
    """Mirror of the loader's get_relative_pose (scipy, (x,y,z,w) quats)."""
    rot = R.from_quat(pose[:, 3:7])
    first = R.from_quat(np.tile(pose[:1, 3:7], (len(pose), 1)))
    trans = pose[:, :3] - pose[:1, :3]
    quat = (first.inv() * rot).as_quat()
    return np.concatenate([trans, quat], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", required=True,
                    help="exported task root; repeat for joint stats over several tasks")
    ap.add_argument("--out", default=None, help="also write the dict as JSON here")
    args = ap.parse_args()

    rows, n_eps = [], 0
    for ds in args.dataset:
        ds = Path(ds)
        for p in sorted((ds / "data" / "chunk-000").glob("episode_*.parquet")):
            a = np.stack(pq.read_table(p, columns=["action"])["action"].to_numpy())
            rel = np.concatenate([relative_pose(a[:, :7]), a[:, 7:8],
                                  relative_pose(a[:, 8:15]), a[:, 15:16]], axis=1)
            rows.append(rel)
            n_eps += 1
    flat = np.concatenate(rows, 0)                                    # (N, 16)
    print(f"[norm_stat] {n_eps} episodes / {len(flat)} frames from {len(args.dataset)} dataset(s)")

    can = np.zeros((len(flat), 30), dtype=np.float64)
    for src, dst in enumerate(USED_IDS):
        can[:, dst] = flat[:, src]

    q01e = np.quantile(can, 0.01, axis=0)
    q99e = np.quantile(can, 0.99, axis=0)
    q01, q99 = np.zeros(30), np.zeros(30)
    for c in range(30):
        if c in TRANS:
            q01[c], q99[c] = q01e[c], q99e[c]
        elif c in QUAT:
            q01[c], q99[c] = -1.0, 1.0
        elif c in GRIP:
            q01[c], q99[c] = 0.0, 1.0

    print("\nempirical ranges on pinned channels (sanity check, not used):")
    for c in sorted(QUAT | GRIP):
        print(f"  ch{c:2d}: q01={q01e[c]: .4f}  q99={q99e[c]: .4f}")
    outside = np.mean((can < (q01 - 0.25 * (q99 - q01))) | (can > (q99 + 0.25 * (q99 - q01))))
    print(f"fraction of values landing outside the loader's +-1.5 clip: {outside:.5%}")

    fmt = lambda v: json.dumps([round(float(x), 12) for x in v])
    print("\npaste into the task config:\n")
    print("va_rmbench_cfg.action_norm_method = 'quantiles'")
    print("va_rmbench_cfg.norm_stat = {")
    print(f'    "q01": {fmt(q01)},')
    print(f'    "q99": {fmt(q99)},')
    print("}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"q01": q01.tolist(), "q99": q99.tolist(),
             "episodes": n_eps, "frames": len(flat)}, indent=2))
        print(f"\n[norm_stat] wrote {args.out}")


if __name__ == "__main__":
    main()
