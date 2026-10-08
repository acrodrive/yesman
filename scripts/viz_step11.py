"""11단계 Figure 3(BEV 시각화)과 실패 사례 그림. 장면은 아래 조건에 맞는 행에서 무작위(시드 0)로 고른다 (결과를 보기 전에 정함).

Figure 3: (가) D1의 offset·차선 변경 결정 중 Ours가 ACCEPT하고 결정대로 그린 장면 2개,
          (나) D2 다른 차·보행자 기준(주 결과) 중 B-순응은 결정대로 실행하고 Ours는 REJECT한 장면 2개.
실패 사례 (Ours 시드 0): 청개구리(D1 REJECT) 2, 맹목적 추종(D2 주 결과 ACCEPT이고 결정대로) 2,
          대체 경로 실패(D2 REJECT인데 Ours 경로가 NC·DAC·DDC 중 하나라도 실패) 2, 언행 불일치(D1 ACCEPT인데 결정과 다름) 2.

그림: 왼쪽 CAM_F0에 경로 투영, 오른쪽 BEV(지도, 물체, 다른 물체의 4초 움직임 점선).
  검은 점선 = 사람, 청록 = B2, 주황 = B-순응, 파랑(굵게) = Ours. 제목: 결정, Ours의 flag, 판정.
사용법: python scripts/viz_step11.py
출력: docs/figs/step11/fig3_*.png, docs/figs/step11/fail_*.png
"""

import json
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
from viz_pred import ground_z, project  # noqa: E402

from navsim.common.dataclasses import SensorConfig  # noqa: E402
from navsim.visualization.bev import add_annotations_to_bev_ax, add_map_to_bev_ax  # noqa: E402
from yesman.data import scene_loader  # noqa: E402
from yesman.plot_style import use_korean_font  # noqa: E402

ROOT = Path(os.environ["YESMAN_ROOT"])
E = ROOT / "exp/eval"
RUNS = {"B2": "B2_seed0", "B-순응": "Bcomply_rt_seed0", "Ours": "Ours_w025_rt_seed0"}
STYLE = {"사람": dict(color="k", ls="--", lw=1.5), "B2": dict(color="#1baf7a", ls="-", lw=2),
         "B-순응": dict(color="#eb6834", ls="-", lw=2), "Ours": dict(color="#2a78d6", ls="-", lw=3.2)}
SAFE = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance"]


def table(name):
    base = pd.read_parquet(ROOT / f"data_lists/eval/{name}.parquet")
    t = json.load(open(E / f"{RUNS['Ours']}/flag_threshold.json"))["tau"]
    for m, r in RUNS.items():
        pr = pd.read_parquet(E / f"{r}/{name}.parquet")
        l1 = pd.read_parquet(E / f"{r}/{name}_l1.parquet")
        base[f"pred_{m}"] = pr.pred_poses.values
        base[f"follow_{m}"] = l1.follow.values
        base[f"dhat_{m}"] = l1.dhat.values
        if m == "Ours":
            base["p"] = pr.p.values
            base["accept"] = pr.p.values >= t
            if name == "D2":
                sc = pd.read_parquet(E / f"{r}/D2_scores.parquet")
                base["ours_safe"] = (sc[SAFE].min(axis=1) >= 1).values
    return base


def draw(scene, row, out: Path, title: str, rng=40.0):
    cur = scene.scene_metadata.num_history_frames - 1
    fr = scene.frames[cur]
    paths = {"사람": scene.get_future_trajectory(8).poses}
    for m in RUNS:
        paths[m] = np.asarray(row[f"pred_{m}"]).reshape(8, 3)
    fig = plt.figure(figsize=(17, 7))
    ax0 = fig.add_axes([0.01, 0.08, 0.5, 0.78])
    ax0.imshow(fr.cameras.cam_f0.image)
    z = ground_z(fr.annotations)
    for k, p in paths.items():
        uv = project(fr.cameras.cam_f0, p, z)
        ax0.plot(uv[:, 0], uv[:, 1], marker="o", ms=3, **STYLE[k])
    ax0.set_xlim(0, fr.cameras.cam_f0.image.shape[1])
    ax0.set_ylim(fr.cameras.cam_f0.image.shape[0], 0)
    ax0.axis("off")
    ax = fig.add_axes([0.53, 0.05, 0.45, 0.82])
    add_map_to_bev_ax(ax, scene.map_api, StateSE2(*fr.ego_status.ego_pose))
    add_annotations_to_bev_ax(ax, fr.annotations)
    for tr in future_tracks(scene).values():
        ax.plot(tr[:, 1], tr[:, 0], ":", color="dimgray", lw=1, zorder=9)
    for k, p in paths.items():
        ax.plot(np.r_[0, p[:, 1]], np.r_[0, p[:, 0]], marker="o", ms=3, zorder=12, label=k, **STYLE[k])
    ax.set_aspect("equal")
    ax.set_xlim(rng / 2, -rng / 2)
    ax.set_ylim(-rng / 4, rng)
    ax.legend(loc="lower right", fontsize=9)
    flag = "ACCEPT" if row["accept"] else "REJECT"
    fig.suptitle(f"{title}  [{row['token']} / {row['cf_name']}]\n결정: {row['decision']}   Ours: {flag} (p = {row['p']:.4f})"
                 f"   결정대로 그림: B2 {'O' if row['follow_B2'] else 'X'}, B-순응 {'O' if row['follow_B-순응'] else 'X'}, "
                 f"Ours {'O' if row['follow_Ours'] else 'X'}", fontsize=11)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=70)
    plt.close(fig)
    return out


def main():
    use_korean_font()
    d1, d2 = table("D1"), table("D2")
    main2 = d2[d2.visibility != "unseen"]
    picks = {
        "fig3_feasible": ("실행 가능한 결정: Ours가 따름", d1[d1.cf_name.isin(["offset_L", "offset_R", "lc_L", "lc_R"])
                                                     & d1.accept & d1.follow_Ours]),
        "fig3_infeasible": ("실행할 수 없는 결정(다른 차·보행자): B-순응은 실행, Ours는 거부",
                            main2[(main2.category == "agent") & main2["follow_B-순응"] & ~main2.accept]),
        "fail_frog": ("실패: 청개구리 (실행 가능한데 거부)", d1[~d1.accept]),
        "fail_blind": ("실패: 맹목적 추종 (실행할 수 없는데 ACCEPT하고 실행)", main2[main2.accept & main2.follow_Ours]),
        "fail_alt": ("실패: 대체 경로 실패 (거부했지만 대체 경로가 안전 항목 실패)", main2[~main2.accept & ~main2.ours_safe]),
        "fail_mismatch": ("실패: 언행 불일치 (ACCEPT인데 결정과 다르게 그림)", d1[d1.accept & ~d1.follow_Ours]),
    }
    cur = [3]
    sc = SensorConfig(cam_f0=cur, cam_l0=False, cam_l1=False, cam_l2=False, cam_r0=False, cam_r1=False,
                      cam_r2=False, cam_b0=False, lidar_pc=False)
    for name, (title, pool) in picks.items():
        rows = pool.sample(n=min(2, len(pool)), random_state=0)
        print(name, f"pool {len(pool):,}")
        loader = scene_loader("navtest", rows.log_name, rows.token, sensor_config=sc)
        for i, row in enumerate(rows.to_dict("records")):
            print(" ", draw(loader.get_scene_from_token(row["token"]), row,
                            ROOT / f"docs/figs/step11/{name}_{i}.png", title))


if __name__ == "__main__":
    main()
