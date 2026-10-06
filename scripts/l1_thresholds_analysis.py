"""4단계: navtrain 사람 경로의 L1 특징 분포를 보고 기준값(yesman/l1_thresholds.yaml)을 정하는 데 쓰는 표와 그림을 만든다.

- 앞뒤 행동 값(속도, 방향 변화)은 저장된 사람 경로(poses)와 현재 속도(v0)에서 다시 계산한다(motion_features).
  특징 추출 뒤에 속도 계산 방법을 바꾸어도 지도를 다시 읽지 않아도 된다.
- 출력: 분위수 표(stdout), 히스토그램 그림(exp/step04/thresholds/*.png)

사용법: python scripts/l1_thresholds_analysis.py [--table exp/l1/navtrain_features.parquet]
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from yesman.l1 import lane_index, motion_features

EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
Q = [0.5, 0.75, 0.9, 0.95, 0.99]


def refresh_motion(df: pd.DataFrame) -> pd.DataFrame:
    """poses와 v0로 앞뒤 행동 값을 다시 계산한다."""
    m = pd.DataFrame([motion_features(np.asarray(p), v0) for p, v0 in zip(df.poses, df.v0)], index=df.index)
    df = df.copy()
    df[m.columns] = m
    return df


def per_segment(df: pd.DataFrame) -> pd.DataFrame:
    """구간 단위 표로 펼친다 (장면 x 구간)."""
    rows = []
    for k in (1, 2):
        cols = [c for c in df.columns if c.startswith(f"seg{k}_")]
        s = df[cols + ["token", "v0"]].rename(columns={c: c[5:] for c in cols})
        s["seg"] = k
        rows.append(s)
    seg = pd.concat(rows, ignore_index=True)
    seg["dh_deg"] = np.degrees(seg.dh)
    seg["on_conn"] = (seg.conn_overlap > 0) | (seg.on_conn_start > 0)
    seg["dd"] = seg.d_end - seg.d_start
    seg["i0"] = [lane_index(*a) for a in zip(seg.d_start, seg.wl_start, seg.wr_start)]
    seg["i1"] = [lane_index(*a) for a in zip(seg.d_end, seg.wl_end, seg.wr_end)]
    seg["decel"] = (seg.v_start - seg.v_end).clip(lower=0) / 2.0
    return seg


def quant(name, x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    qs = np.quantile(x, Q) if len(x) else [np.nan] * len(Q)
    print(f"{name:<45} n={len(x):>7}  " + "  ".join(f"p{int(q * 100)}={v:7.3f}" for q, v in zip(Q, qs)))


def hist(ax, x, bins, title, logy=False, vlines=()):
    ax.hist(np.asarray(x)[np.isfinite(x)], bins=bins, color="steelblue")
    for v in vlines:
        ax.axvline(v, color="red", ls="--")
    ax.set_title(title, fontsize=9)
    if logy:
        ax.set_yscale("log")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=EXP / "l1/navtrain_features.parquet")
    ap.add_argument("--out", type=Path, default=EXP / "step04/thresholds")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.table)
    print(f"scenes {len(df)}, valid {int(df.valid.sum())} ({df.valid.mean() * 100:.2f}%)")
    print(df[df.valid == 0].reason.value_counts().head(10).to_string())
    df = refresh_motion(df[df.valid == 1])
    seg = per_segment(df)
    print(f"segments {len(seg)}")

    print("\n[앞뒤] 구간 끝 속도 v_end")
    quant("v_end (all)", seg.v_end)
    for c in (0.1, 0.3, 0.5, 1.0):
        print(f"  v_end < {c}: {(seg.v_end < c).mean() * 100:.1f}%")
    quant("v_end (v_end >= 0.5)", seg.v_end[seg.v_end >= 0.5])
    quant("decel (v_end < 0.5)", seg.decel[seg.v_end < 0.5])
    quant("decel (v_end < 0.5, v_start >= 0.5)", seg.decel[(seg.v_end < 0.5) & (seg.v_start >= 0.5)])

    print("\n[회전] 구간 방향 변화 |dh| [deg]")
    quant("|dh| on connector", seg.dh_deg[seg.on_conn].abs())
    quant("|dh| not on connector", seg.dh_deg[~seg.on_conn].abs())
    for c in (10, 15, 20, 25, 30):
        print(f"  on connector |dh| >= {c}: {(seg.dh_deg[seg.on_conn].abs() >= c).mean() * 100:.1f}%   "
              f"(not on connector: {(seg.dh_deg[~seg.on_conn].abs() >= c).mean() * 100:.1f}%)")
    quant("|dh| on connector, |dh| >= 15", seg.dh_deg[seg.on_conn & (seg.dh_deg.abs() >= 15)].abs())

    print("\n[옆] 같은 차선 안 옆 거리 변화 |dd| [m] (차선 번호가 같은 구간)")
    same = (seg.i0 == seg.i1) & ~(seg.on_conn & (seg.dh_deg.abs() >= 15))
    quant("|dd| same lane", seg.dd[same].abs())
    for c in (0.3, 0.5, 0.7, 1.0):
        print(f"  |dd| >= {c}: {(seg.dd[same].abs() >= c).mean() * 100:.2f}%")
    quant("|dd| same lane, |dd| >= 0.5", seg.dd[same & (seg.dd.abs() >= 0.5)].abs())
    quant("차선 폭 wl+wr (구간 끝)", seg.wl_end + seg.wr_end)
    lc = seg.i0 != seg.i1
    print(f"  lane index changed: {lc.mean() * 100:.2f}% (|Δidx|>=2: {((seg.i1 - seg.i0).abs() >= 2).mean() * 100:.3f}%)")
    quant("vlat_peak (lane change)", seg.vlat_peak[lc])
    quant("vlat_peak (same lane, not turning)", seg.vlat_peak[same])
    quant("|d_start| (구간 1 시작, 출발점)", seg.d_start[seg.seg == 1].abs())
    quant("hrel_max [deg] (not turning)", np.degrees(seg.hrel_max[~(seg.on_conn & (seg.dh_deg.abs() >= 15))]))

    print("\n[굽은 도로] 연결 차선이 아닌 같은 차선 구간: 차선 중심선 방향 변화 |lane_dh|별 |dd|")
    seg["lane_dh_deg"] = np.degrees(seg.lane_dh).abs()
    base = (seg.i0 == seg.i1) & ~seg.on_conn
    straight = base & (seg.lane_dh_deg < 2)
    q_off = (seg.dd[straight].abs() < 0.5).mean()  # 곧은 구간에서 offset_min=0.5 m가 몇 번째 백분위수인가
    print(f"  곧은 구간(|lane_dh| < 2deg)에서 |dd| < 0.5 m 비율 = {q_off * 100:.2f}% (이 백분위수를 굽은 구간 기준에 쓴다)")
    for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 20), (20, 90)]:
        m = base & (seg.lane_dh_deg >= lo) & (seg.lane_dh_deg < hi)
        x = seg.dd[m].abs()
        print(f"  |lane_dh| {lo:>2}-{hi:<2}deg n={m.sum():>6}  |dd| p50 {x.median():.2f}  p90 {x.quantile(.9):.2f}  "
              f"p{q_off * 100:.1f} {x.quantile(q_off):.2f}  (>=0.5: {(x >= 0.5).mean() * 100:.1f}%)")
    for c in (3, 5, 8):
        m = base & (seg.lane_dh_deg >= c)
        print(f"  curve_deg={c}: 굽은 구간 n={m.sum()}, 같은 백분위수의 |dd| = {seg.dd[m].abs().quantile(q_off):.2f} m")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    hist(axes[0, 0], seg.v_end, np.linspace(0, 25, 101), "v_end [m/s]", logy=True, vlines=[0.5])
    hist(axes[0, 1], seg.decel[seg.v_end < 0.5], np.linspace(0, 6, 61), "decel [m/s^2] (v_end < 0.5)", logy=True)
    hist(axes[0, 2], seg.dh_deg[seg.on_conn].abs(), np.linspace(0, 120, 121), "|dh| [deg] on connector", logy=True)
    hist(axes[1, 0], seg.dh_deg[~seg.on_conn].abs(), np.linspace(0, 120, 121), "|dh| [deg] not on connector", logy=True)
    hist(axes[1, 1], seg.dd[same].abs(), np.linspace(0, 3, 121), "|dd| [m] same lane", logy=True, vlines=[0.5])
    hist(axes[1, 2], seg.vlat_peak[lc], np.linspace(0, 4, 81), "vlat_peak [m/s] lane change")
    fig.tight_layout()
    fig.savefig(args.out / "distributions.png", dpi=80)
    print(f"\nsaved {args.out / 'distributions.png'}")


if __name__ == "__main__":
    main()
