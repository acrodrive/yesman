"""4단계: L1의 '애매함'이 어디서 나오는지 그림으로 보여 준다.

1. distributions.png: 애매함 규칙마다, 그 규칙이 보는 값의 navtrain 분포 위에 기준값(빨간 선)과 애매함 구간(주황)을 그린다.
   주황 구간에 들어간 구간(segment)이 애매함으로 표시된다. 분포가 기준값 근처에 몰려 있을수록 애매함이 많다.
2. examples/<이유>_NN.png: 이유마다 무작위 장면을 BEV로 그린다 (scripts/viz_l1.py의 그림).

사용법: python scripts/l1_ambiguity_viz.py [--n 4]   # → exp/step04/ambiguous/
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

sys.path.insert(0, str(Path(__file__).parent))
from l1_thresholds_analysis import per_segment, refresh_motion  # noqa: E402
from viz_l1 import draw  # noqa: E402

from yesman.data import scene_loader  # noqa: E402
from yesman.l1 import boundary_margin, load_thresholds  # noqa: E402

EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
REASONS = {
    "offset_margin": "offset 기준 근처: 같은 차선 안 옆 거리 변화 |dd|",
    "turn_margin": "turn 기준 근처: 연결 차선 위 방향 변화 |dh|",
    "stop_speed_margin": "stop 기준 근처: 구간 끝 속도",
    "near_boundary": "차선 경계 근처: 구간 시작/끝에서 경계선까지 거리",
}


def band_hist(ax, x, bins, center, margin, title, xlabel, extra_center=None):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    ax.hist(x, bins=bins, color="steelblue")
    in_band = np.zeros(len(x), dtype=bool)
    for c in [center] + ([extra_center] if extra_center is not None else []):
        if c is None:
            continue
        lo, hi = max(c - margin, bins[0]), c + margin
        ax.axvspan(lo, hi, color="orange", alpha=0.35)
        if c > 0:
            ax.axvline(c, color="red", ls="--")
        in_band |= (x >= c - margin) & (x <= c + margin)
    ax.set_title(f"{title}\norange band = ambiguous: {in_band.mean() * 100:.1f}% of these segments", fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("segments")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="이유마다 그릴 장면 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=EXP / "step04/ambiguous")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    th = load_thresholds()

    feats = pd.read_parquet(EXP / "l1/navtrain_features.parquet")
    feats = refresh_motion(feats[feats.valid == 1])
    seg = per_segment(feats)
    seg["lane_dh_deg"] = np.degrees(seg.lane_dh).abs()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    same = (seg.i0 == seg.i1) & ~(seg.on_conn & (seg.dh_deg.abs() >= th["turn_min_deg"]))
    straight = same & (seg.lane_dh_deg < th["curve_deg"])
    band_hist(axes[0, 0], seg.dd[straight].abs(), np.linspace(0, 2, 81), th["offset_min"], th["offset_margin"],
              "offset_margin: same-lane segments on straight roads (threshold 0.5 m)", "|dd| = change of lateral offset in the segment [m]")
    band_hist(axes[0, 1], seg.dh_deg[seg.on_conn].abs(), np.linspace(0, 60, 121), th["turn_min_deg"],
              th["turn_margin_deg"], "turn_margin: segments on intersection connectors (threshold 15 deg)", "|dh| = heading change in the segment [deg]")
    band_hist(axes[1, 0], seg.v_end, np.linspace(0, 3, 121), th["v_stop"], th["v_stop_margin"],
              "stop_speed_margin: all segments, 0-3 m/s shown (threshold 0.5 m/s)", "speed at segment end [m/s]")
    bm = np.minimum([boundary_margin(*a) for a in zip(seg.d_start, seg.wl_start, seg.wr_start)],
                    [boundary_margin(*a) for a in zip(seg.d_end, seg.wl_end, seg.wr_end)])
    band_hist(axes[1, 1], bm, np.linspace(0, 2.5, 101), 0.0, th["lat_boundary_margin"],
              "near_boundary: distance to nearest lane boundary at segment start/end", "distance to lane boundary [m]")
    fig.suptitle("Where L1 ambiguity comes from (navtrain). red line = threshold, orange = ambiguous band", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out / "distributions.png", dpi=80)
    plt.close(fig)
    print(f"saved {args.out / 'distributions.png'}")

    # 이유별 예시 장면 (그 이유 하나만 가진 장면에서 고른다)
    lab = pd.read_parquet(EXP / "l1/navtrain_labels.parquet")
    lab = lab[lab.valid & (lab.reason != "")]
    reasons = lab.reason.str.split(";").apply(lambda xs: {x.split(":")[1] for x in xs})
    only = lab[reasons.apply(len) == 1].assign(r=reasons[reasons.apply(len) == 1].apply(lambda s: next(iter(s))))
    picks = {r: only[only.r == r].sample(n=args.n, random_state=args.seed) for r in REASONS}
    allp = pd.concat(picks.values())
    loader = scene_loader("navtrain", allp.log_name, allp.token)
    ex = args.out / "examples"
    for r, df in picks.items():
        for i, t in enumerate(df.token):
            p = draw(loader.get_scene_from_token(t), ex, tag=f"{r}_{i:02d}_")
            print(p)


if __name__ == "__main__":
    main()
