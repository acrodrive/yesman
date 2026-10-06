"""4단계: L1 결과를 BEV로 그린다 (기준 차선열, 옆 거리, 구간별 결정).

그림: ego 좌표계 BEV (가로축 y, 왼쪽 +). 지도와 물체는 공식 navsim.visualization 함수로 그린다.
- 파란 굵은 선: L1이 고른 기준 차선열의 중심선 (주황 부분 = 교차로 연결 차선), 얇은 파란 선: 그 차선열의 경계
- 경로: 구간 1(0~2초) 초록, 구간 2(2~4초) 보라. 점 옆 숫자는 옆 거리 d [m] (왼쪽 +)
- 제목: L1 결정과 구간별 주요 값

사용법 (source scripts/setup/env.sh 후):
    python scripts/viz_l1.py --split navtest --logs <log> --tokens <token> ...
    python scripts/viz_l1.py --from_table exp/l1/navtrain_features.parquet --n 20 --seed 0 --out exp/step04/random20
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

from navsim.visualization.bev import add_annotations_to_bev_ax, add_map_to_bev_ax
from yesman.data import scene_loader
from yesman.l1 import SEGMENTS, classify, features_from_scene

OUT = Path(os.environ["NAVSIM_EXP_ROOT"]) / "step04" / "viz"


def to_local(xy: np.ndarray, ego_pose) -> np.ndarray:
    x0, y0, h0 = ego_pose
    c, s = np.cos(h0), np.sin(h0)
    dx, dy = xy[:, 0] - x0, xy[:, 1] - y0
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)


def draw(scene, out_dir: Path, tag: str = "", rng: float = 35.0) -> Path:
    cur = scene.scene_metadata.num_history_frames - 1
    frame = scene.frames[cur]
    ego = frame.ego_status.ego_pose
    dbg: dict = {}
    f = features_from_scene(scene, debug=dbg)
    dec = classify(f)

    fig, ax = plt.subplots(figsize=(9, 9))
    add_map_to_bev_ax(ax, scene.map_api, StateSE2(*ego))
    add_annotations_to_bev_ax(ax, frame.annotations)
    if dbg:
        for e in dbg["chain_edges"]:
            c = to_local(e.xy, ego)
            ax.plot(c[:, 1], c[:, 0], "-", color="orange" if e.is_connector else "royalblue", lw=3, alpha=0.8, zorder=8)
            for b in (e.left, e.right):
                bl = to_local(np.array(b.coords), ego)
                ax.plot(bl[:, 1], bl[:, 0], "-", color="royalblue", lw=0.8, zorder=8)
        loc = to_local(dbg["glob"][:, :2], ego)
        for k, (a, b) in enumerate(SEGMENTS):
            ax.plot(loc[a:b + 1, 1], loc[a:b + 1, 0], "o-", color=["green", "purple"][k], ms=5, lw=2, zorder=10,
                    label=f"segment {k + 1} ({a * 0.5:.0f}-{b * 0.5:.0f} s)")
        for i in range(9):
            ax.text(loc[i, 1] - 0.6, loc[i, 0], f"{dbg['d'][i]:+.1f}", fontsize=8, zorder=11)
    else:
        human = scene.get_future_trajectory(8).poses
        ax.plot(np.r_[0, human[:, 1]], np.r_[0, human[:, 0]], "o-", color="green", zorder=10)
    ax.set_aspect("equal")
    ax.set_xlim(rng, -rng)
    ax.set_ylim(-rng / 2, rng * 1.5)
    ax.set_xlabel("y [m] (+ left)")
    ax.set_ylabel("x [m] (+ forward)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    lines = [f"{tag} {scene.scene_metadata.initial_token}  v0 {f['v0']:.1f} m/s",
             f"L1: {dec}" + (f"  [{dec.reason}]" if dec.reason else "")]
    if f.get("valid"):
        for k in (1, 2):
            g = lambda n: f[f"seg{k}_{n}"]  # noqa: E731
            lines.append(f"seg{k}: v {g('v_start'):.1f}->{g('v_end'):.1f}  dh {np.degrees(g('dh')):+.0f}deg  "
                         f"conn {g('conn_overlap'):.0f}m  d {g('d_start'):+.2f}->{g('d_end'):+.2f}  "
                         f"w {g('wl_end'):.1f}/{g('wr_end'):.1f}  vlat {g('vlat_peak'):.2f}  "
                         f"hrel {np.degrees(g('hrel_max')):.0f}deg")
    ax.set_title("\n".join(lines), fontsize=8, loc="left", family="monospace")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}{scene.scene_metadata.initial_token}.png"
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--logs", nargs="*")
    ap.add_argument("--tokens", nargs="*")
    ap.add_argument("--from_table", type=Path, help="L1 특징 표(parquet, token과 log_name 열)에서 무작위로 고른다")
    ap.add_argument("--query", default=None, help="--from_table에서 고를 행 조건 (pandas query)")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if args.from_table:
        df = pd.read_parquet(args.from_table)
        if args.query:
            df = df.query(args.query)
        df = df.sample(n=min(args.n, len(df)), random_state=args.seed)
        tokens, logs, split = list(df.token), list(df.log_name), df.split.iloc[0]
    else:
        tokens, logs, split = args.tokens, args.logs, args.split
    loader = scene_loader(split, logs, tokens)
    for i, t in enumerate(tokens):
        print(draw(loader.get_scene_from_token(t), args.out, tag=f"{i:02d}_" if args.from_table else ""))


if __name__ == "__main__":
    main()
