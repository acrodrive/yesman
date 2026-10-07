"""8단계 이후: planner가 그린 경로를 전방 카메라 영상과 BEV 위에 겹쳐 그린다 (Figure 3의 바탕).

그림: 왼쪽 CAM_F0에 경로를 투영, 오른쪽 BEV(지도, 물체, 다른 물체의 4초 움직임).
  검은 점선 = 사람 경로, 굵은 연두 = 목표 경로(CF⁺이면 L2 경로, 주황 아래에 깔림), 주황 = 모델이 그린 경로.
투영: 경로는 NAVSIM의 ego 좌표계(어노테이션과 같은 좌표계)이므로, NAVSIM의 어노테이션 투영과 같이
  sensor2lidar로 카메라 좌표계로 옮긴다. 높이는 주변 차량 박스 바닥 높이의 중앙값(없으면 -1.9 m)을 땅으로 본다.

사용법: python scripts/viz_pred.py --pred exp/train/step08_overfit_Bcomply/pred.parquet --n 12 --out exp/step08/viz
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
from navsim.visualization.camera import _transform_points_to_image  # noqa: E402
from yesman.data import scene_loader  # noqa: E402

LINES = {"human": dict(color="k", ls="--", lw=1.5), "target": dict(color="limegreen", ls="-", lw=7, alpha=0.5),
         "pred": dict(color="darkorange", ls="-", lw=2.5)}


def ground_z(annotations) -> float:
    b = annotations.boxes
    if len(b) == 0:
        return -1.9
    near = np.linalg.norm(b[:, :2], axis=1) < 40
    z = b[near, 2] - b[near, 5] / 2  # 중심 높이 - 높이/2 (BoundingBoxIndex: x y z l w h heading)
    return float(np.median(z)) if len(z) else -1.9


def project(cam, poses, z) -> np.ndarray:
    pts = np.c_[np.r_[0.0, poses[:, 0]], np.r_[0.0, poses[:, 1]], np.full(len(poses) + 1, z)]
    R, t = cam.sensor2lidar_rotation, cam.sensor2lidar_translation
    pts_cam = (pts - t) @ R  # lidar → camera
    uv, ok = _transform_points_to_image(pts_cam, cam.intrinsics, cam.image.shape[:2])
    uv[~ok] = np.nan
    return uv


def draw(scene, row, out: Path, tag: str, rng: float = 40.0):
    cur = scene.scene_metadata.num_history_frames - 1
    fr = scene.frames[cur]
    paths = {"human": scene.get_future_trajectory(8).poses,
             "target": np.asarray(row["target_poses"]).reshape(8, 3),
             "pred": np.asarray(row["pred_poses"]).reshape(8, 3)}
    if row["sample_type"] == "original":
        paths.pop("target")  # 원래 결정의 목표 = 사람 경로
    fig = plt.figure(figsize=(17, 7))
    ax0 = fig.add_axes([0.01, 0.1, 0.5, 0.8])
    ax0.imshow(fr.cameras.cam_f0.image)
    z = ground_z(fr.annotations)
    for k, p in paths.items():
        uv = project(fr.cameras.cam_f0, p, z)
        ax0.plot(uv[:, 0], uv[:, 1], marker="o", ms=3, **LINES[k])
    ax0.set_xlim(0, fr.cameras.cam_f0.image.shape[1])
    ax0.set_ylim(fr.cameras.cam_f0.image.shape[0], 0)
    ax0.axis("off")
    ax = fig.add_axes([0.53, 0.05, 0.45, 0.85])
    add_map_to_bev_ax(ax, scene.map_api, StateSE2(*fr.ego_status.ego_pose))
    add_annotations_to_bev_ax(ax, fr.annotations)
    for tr in future_tracks(scene).values():
        ax.plot(tr[:, 1], tr[:, 0], ":", color="dimgray", lw=1, zorder=9)
    for k, p in paths.items():
        ax.plot(np.r_[0, p[:, 1]], np.r_[0, p[:, 0]], marker="o", ms=3, zorder=12, label=k, **LINES[k])
    ax.set_aspect("equal")
    ax.set_xlim(rng / 2, -rng / 2)
    ax.set_ylim(-rng / 4, rng)
    ax.legend(loc="lower right", fontsize=8)
    err = np.linalg.norm(paths["pred"][:, :2] - np.asarray(row["target_poses"]).reshape(8, 3)[:, :2], axis=1)
    fig.suptitle(f"{tag}{row['token']} [{row['sample_type']}/{row['cf_name']}] decision: {row['decision']}\n"
                 f"pred vs target ADE {err.mean():.2f} m, FDE {err[-1]:.2f} m", fontsize=10, family="monospace")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{tag}{row['token']}_{row['cf_name']}.png"
    fig.savefig(path, dpi=70)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--split", default="navtrain")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--query", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    df = pd.read_parquet(args.pred)
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
