"""5단계: L2 판정을 그린다 (scripts/l2_run.py의 결과 표에서 고른 행).

그림: 왼쪽 CAM_F0, 오른쪽 ego 좌표계 BEV (가로축 y, 왼쪽 +).
- 지도와 현재 물체: 공식 navsim.visualization 함수. 다른 물체의 4초 동안 움직임: 회색 점선 (실제 미래 위치)
- 채점한 후보: 통과 = 초록 얇은 선, 실패 = 빨간 얇은 선
- 대표 후보(굵은 선): 가능이면 통과한 후보 중 점수가 가장 높은 것(파랑), 불가능이면 떨어진 후보 중 가장 좋은 것(주황)
- 사람의 실제 경로: 검은 점선
- 제목: 결정, 판정, 판단 근거, 후보 수, 항목별 실패 비율

사용법: python scripts/viz_l2.py --table exp/l2/navtest_probe.parquet --query "status == 'infeasible'" --n 30
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nuplan.common.actor_state.state_representation import StateSE2

from navsim.common.dataclasses import SensorConfig
from navsim.visualization.bev import add_annotations_to_bev_ax, add_map_to_bev_ax
from yesman.data import scene_loader
from yesman.l2 import BANK_PATH

EXP = Path(os.environ["NAVSIM_EXP_ROOT"])


def future_tracks(scene):
    """다른 물체의 현재~4초 위치 (현재 ego 좌표계). {track_token: (n, 2)}"""
    cur = scene.scene_metadata.num_history_frames - 1
    x0, y0, h0 = scene.frames[cur].ego_status.ego_pose
    tracks = {}
    for i in range(cur, min(cur + 9, len(scene.frames))):
        fr = scene.frames[i]
        # 각 프레임의 박스는 그 프레임 ego 좌표계이다 → 전역 → 현재 ego 좌표계
        xi, yi, hi = fr.ego_status.ego_pose
        b = fr.annotations.boxes[:, :2]
        gx = xi + np.cos(hi) * b[:, 0] - np.sin(hi) * b[:, 1]
        gy = yi + np.sin(hi) * b[:, 0] + np.cos(hi) * b[:, 1]
        dx, dy = gx - x0, gy - y0
        lx, ly = np.cos(h0) * dx + np.sin(h0) * dy, -np.sin(h0) * dx + np.cos(h0) * dy
        for t, a, c in zip(fr.annotations.track_tokens, lx, ly):
            tracks.setdefault(t, []).append((a, c))
    return {t: np.array(v) for t, v in tracks.items() if len(v) > 1}


def draw(scene, row, bank_poses, out_dir: Path, tag: str, rng: float = 40.0) -> Path:
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
    # 실제로 채점한 경로 (옮긴 뒤). 예전 결과 표에 없으면 경로 모음의 원래 경로를 그린다
    scored = (np.asarray(row["scored_poses"]).reshape(-1, 8, 3) if len(row.get("scored_poses", [])) else
              bank_poses[np.asarray(row["scored_bank_idx"], dtype=int)])
    for p, ok in zip(scored, row["scored_pass"]):
        ax.plot(np.r_[0, p[:, 1]], np.r_[0, p[:, 0]], "-", color="green" if ok else "red", lw=0.7, alpha=0.6,
                zorder=10)
    if row["best_bank_idx"] >= 0:
        p = scored[list(row["scored_bank_idx"]).index(int(row["best_bank_idx"]))]
        ax.plot(np.r_[0, p[:, 1]], np.r_[0, p[:, 0]], "o-", color="blue" if row["status"] == "feasible" else "orange",
                lw=2.5, ms=4, zorder=12, label="best candidate")
    h = scene.get_future_trajectory(8).poses
    ax.plot(np.r_[0, h[:, 1]], np.r_[0, h[:, 0]], "k--", lw=1.5, zorder=11, label="human (actual)")
    ax.set_aspect("equal")
    ax.set_xlim(rng, -rng)
    ax.set_ylim(-rng / 2, rng * 1.5)
    ax.set_xlabel("y [m] (+ left)")
    ax.set_ylabel("x [m] (+ forward)")
    ax.legend(loc="lower right", fontsize=8)
    fails = ", ".join(f"{k[5:]} {row[k]:.0%}" for k in ("fail_collision", "fail_off_road", "fail_wrong_way")
                      if k in row and pd.notna(row[k]))
    fig.suptitle(f"{tag}{row['token']} [{row['decision_type']}, {row.get('mode', 'ego')}]  decision: {row['decision']}\n"
                 f"L2: {row['status'].upper()}  reason: {row['reason'] or '-'}  |  candidates: prefilter "
                 f"{row['n_prefilter']}, match {row['n_match']}/{row['n_relabeled']}, pass {row['n_pass']}/"
                 f"{row['n_scored']}  |  fail rate: {fails}", fontsize=10, family="monospace")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}{row['token']}_{row['decision_type']}.png"
    fig.savefig(path, dpi=70)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--query", default="status == 'infeasible'")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--out", type=Path, default=EXP / "step05/viz")
    args = ap.parse_args()
    df = pd.read_parquet(args.table).query(args.query)
    df = df.sample(n=min(args.n, len(df)), random_state=args.seed)
    bank = np.stack(pd.read_parquet(BANK_PATH, columns=["poses"]).poses.to_numpy()).reshape(-1, 8, 3)
    cur = [3]
    sc = SensorConfig(cam_f0=cur, cam_l0=False, cam_l1=False, cam_l2=False, cam_r0=False, cam_r1=False,
                      cam_r2=False, cam_b0=False, lidar_pc=False)
    loader = scene_loader(args.split, df.log_name, df.token, sensor_config=sc)
    for i, row in enumerate(df.to_dict("records")):
        print(draw(loader.get_scene_from_token(row["token"]), row, bank, args.out, tag=f"{i:02d}_"))


if __name__ == "__main__":
    main()
