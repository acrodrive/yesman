"""6단계: CF 판정을 그린다 (평가 세트 D2 등에서 고른 행). scripts/viz_l2.py와 같은 모양이지만 대표 후보만 그린다.

그림: 왼쪽 CAM_F0, 오른쪽 BEV. 회색 점선 = 다른 물체의 4초 움직임, 주황 굵은 선 = 대표 후보(불가능: 떨어진 후보 중 최고,
가능: 목표 경로), 검은 점선 = 사람의 실제 경로. 제목: CF 이름, 결정, 판정, 불가능 기준, 판단 근거, 시야.

실제 VLM 결정 표(D3x)이면 사람 결정과 VLM이 꼽은 핵심 요인도 제목에 쓴다.

사용법: python scripts/viz_l3.py --table data_lists/eval/D2.parquet --query "category == 'road'" --n 10 --out exp/step06/road
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nuplan.common.actor_state.state_representation import StateSE2

sys.path.insert(0, str(Path(__file__).parent))
from viz_l2 import future_tracks  # noqa: E402

from navsim.common.dataclasses import SensorConfig  # noqa: E402
from navsim.visualization.bev import add_annotations_to_bev_ax, add_map_to_bev_ax  # noqa: E402
from yesman.data import scene_loader  # noqa: E402

EXP = Path(os.environ["NAVSIM_EXP_ROOT"])


def draw(scene, row, out_dir: Path, tag: str, rng: float = 40.0) -> Path:
    cur = scene.scene_metadata.num_history_frames - 1
    frame = scene.frames[cur]
    fig = plt.figure(figsize=(17, 8.5))
    ax0 = fig.add_axes([0.01, 0.25, 0.45, 0.5])
    ax0.imshow(frame.cameras.cam_f0.image)
    ax0.set_title("CAM_F0")
    ax0.axis("off")
    ax = fig.add_axes([0.5, 0.05, 0.48, 0.82])
    add_map_to_bev_ax(ax, scene.map_api, StateSE2(*frame.ego_status.ego_pose))
    add_annotations_to_bev_ax(ax, frame.annotations)
    for tr in future_tracks(scene).values():
        ax.plot(tr[:, 1], tr[:, 0], ":", color="dimgray", lw=1, zorder=9)
    col = "best_failed_poses" if "best_failed_poses" in row else ("target_poses" if "target_poses" in row else
                                                                  "best_poses")
    if len(row.get(col, [])):
        p = np.asarray(row[col]).reshape(8, 3)
        ax.plot(np.r_[0, p[:, 1]], np.r_[0, p[:, 0]], "o-", color="orange", lw=2.5, ms=4, zorder=12,
                label="representative candidate")
    h = scene.get_future_trajectory(8).poses
    ax.plot(np.r_[0, h[:, 1]], np.r_[0, h[:, 0]], "k--", lw=1.5, zorder=11, label="human (actual)")
    ax.set_aspect("equal")
    ax.set_xlim(rng, -rng)
    ax.set_ylim(-rng / 2, rng * 1.5)
    ax.set_xlabel("y [m] (+ left)")
    ax.set_ylabel("x [m] (+ forward)")
    ax.legend(loc="lower right", fontsize=8)
    extra = "  ".join(f"{k}: {row[k]}" for k in ("category", "reason", "visibility", "weaker_status")
                      if k in row and isinstance(row[k], str) and row[k])
    fails = "  ".join(f"{k[5:]} {row[k]:.0%}" for k in ("fail_collision", "fail_off_road", "fail_wrong_way")
                      if k in row and pd.notna(row[k]))
    title = f"{tag}{row['token']} [{row['cf_name']}]  decision: {row['decision']}\n{extra}  |  fail rate: {fails}"
    if row.get("human_decision"):
        title += f"\nhuman: {row['human_decision']}  (v0 {np.linalg.norm(frame.ego_status.ego_velocity[:2]):.1f} m/s)"
    if row.get("key_issue"):
        title += f"\nVLM key issue: {row['key_issue'][:150]}"
    fig.suptitle(title, fontsize=10, family="monospace")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}{row['token']}_{row['cf_name']}.png"
    fig.savefig(path, dpi=70)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--query", default=None)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    df = pd.read_parquet(args.table)
    if args.query:
        df = df.query(args.query)
    df = df.sample(n=min(args.n, len(df)), random_state=args.seed)
    cur = [3]
    sc = SensorConfig(cam_f0=cur, cam_l0=False, cam_l1=False, cam_l2=False, cam_r0=False, cam_r1=False,
                      cam_r2=False, cam_b0=False, lidar_pc=False)
    loader = scene_loader(args.split, df.log_name, df.token, sensor_config=sc)
    for i, row in enumerate(df.to_dict("records")):
        print(draw(loader.get_scene_from_token(row["token"]), row, args.out, tag=f"{i:02d}_"))


if __name__ == "__main__":
    main()
