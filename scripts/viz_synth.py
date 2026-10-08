"""12단계: 합성한 CF⁺ 목표 경로를 눈으로 확인한다. 결정 종류마다 합성된 행을 무작위(시드 0)로 2개씩 그린다.

그림: 왼쪽 CAM_F0 투영, 오른쪽 BEV. 검은 점선 = 사람, 회색 = 기존 목표(10단계 일관화), 파랑 = 합성 목표.
사용법: python scripts/viz_synth.py [--bundle val]
출력: docs/figs/step12/synth_<cf>.png (결정 종류마다 2장을 한 줄로)
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
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from viz_l2 import future_tracks  # noqa: E402
from viz_pred import ground_z, project  # noqa: E402

from navsim.common.dataclasses import SensorConfig  # noqa: E402
from navsim.visualization.bev import add_annotations_to_bev_ax, add_map_to_bev_ax  # noqa: E402
from yesman.data import scene_loader  # noqa: E402
from yesman.plot_style import use_korean_font  # noqa: E402

ROOT = Path(os.environ["YESMAN_ROOT"])
STYLE = {"사람": dict(color="k", ls="--", lw=1.5), "기존 목표": dict(color="#9a9990", ls="-", lw=4, alpha=0.7),
         "합성 목표": dict(color="#2a78d6", ls="-", lw=2.5)}


def draw(scene, row, old, out):
    fr = scene.frames[scene.scene_metadata.num_history_frames - 1]
    paths = {"사람": scene.get_future_trajectory(8).poses, "기존 목표": np.asarray(old).reshape(8, 3),
             "합성 목표": np.asarray(row["target_poses"]).reshape(8, 3)}
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
    ax.set_xlim(20, -20)
    ax.set_ylim(-10, 40)
    ax.legend(loc="lower right", fontsize=9)
    fig.suptitle(f"{row['cf_name']} [{row['token']}] {row['target_rule']}\n결정: {row['decision']}", fontsize=11)
    fig.savefig(out, dpi=60)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="val")
    args = ap.parse_args()
    use_korean_font()
    new = pd.read_parquet(ROOT / f"exp/l3/train_bundle_{args.bundle}_synth.parquet")
    old = pd.read_parquet(ROOT / f"exp/l3/train_bundle_{args.bundle}_retarget.parquet")
    assert (new.token.values == old.token.values).all()
    new["old_target"] = old.target_poses.values
    syn = new[new.target_rule.str.startswith("synth")]
    out_dir = ROOT / "docs/figs/step12"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = ROOT / "exp/step12/synth_viz"
    tmp.mkdir(parents=True, exist_ok=True)
    sc = SensorConfig(cam_f0=[3], cam_l0=False, cam_l1=False, cam_l2=False, cam_r0=False, cam_r1=False,
                      cam_r2=False, cam_b0=False, lidar_pc=False)
    for cf in ["offset_L", "lc_R", "faster", "stop", "turn_L", "keep"]:
        rows = syn[syn.cf_name == cf].sample(n=min(2, (syn.cf_name == cf).sum()), random_state=0)
        loader = scene_loader("navtrain", rows.log_name, rows.token, sensor_config=sc)
        files = []
        for i, row in enumerate(rows.to_dict("records")):
            f = tmp / f"{cf}_{i}.png"
            draw(loader.get_scene_from_token(row["token"]), row, row["old_target"], f)
            files.append(f)
        ims = [Image.open(f).convert("RGB") for f in files]
        g = Image.new("RGB", (sum(im.width for im in ims), ims[0].height), "white")
        x = 0
        for im in ims:
            g.paste(im, (x, 0))
            x += im.width
        g.save(out_dir / f"synth_{cf}.png")
        print(out_dir / f"synth_{cf}.png")


if __name__ == "__main__":
    main()
