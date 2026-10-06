"""4단계 끝났다는 기준: 굽은 도로에서 차선을 따라 주행한 사람 경로가 keep lane으로 판정되는가.

굽은 도로 장면 = 교차로 연결 차선을 지나지 않고, 기준 차선 중심선의 방향이 4초 동안 min_deg 이상 바뀐 장면.
- 사람이 실제로 차선을 바꾼 장면(구간 시작과 끝의 차선 번호가 다름)은 "차선을 따라 주행한" 장면이 아니므로 따로 센다.
- 결과: 두 구간 모두 keep lane인 비율, 구간별 좌우 행동 분포. keep이 아닌 장면 목록을 저장한다(그림 확인용).

사용법: python scripts/l1_curved_check.py [--split navtrain]
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from yesman.l1 import lane_index

EXP = Path(os.environ["NAVSIM_EXP_ROOT"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="navtrain")
    ap.add_argument("--min_deg", type=float, nargs="*", default=[20, 30, 45])
    args = ap.parse_args()
    f = pd.read_parquet(EXP / f"l1/{args.split}_features.parquet")
    lab = pd.read_parquet(EXP / f"l1/{args.split}_labels.parquet")
    f = f[f.valid == 1].merge(lab[lab.valid], on=["token", "log_name"])

    no_conn = np.ones(len(f), dtype=bool)
    lane_changed = np.zeros(len(f), dtype=bool)
    for k in (1, 2):
        no_conn &= (f[f"seg{k}_conn_overlap"] == 0).to_numpy() & (f[f"seg{k}_on_conn_start"] == 0).to_numpy()
        i0 = [lane_index(*a) for a in zip(f[f"seg{k}_d_start"], f[f"seg{k}_wl_start"], f[f"seg{k}_wr_start"])]
        i1 = [lane_index(*a) for a in zip(f[f"seg{k}_d_end"], f[f"seg{k}_wl_end"], f[f"seg{k}_wr_end"])]
        lane_changed |= np.array(i0) != np.array(i1)
    lane_turn = np.degrees(np.abs(f.seg1_lane_dh + f.seg2_lane_dh)).to_numpy()
    keep = ((f.seg1_lat == "keep_lane") & (f.seg2_lat == "keep_lane")).to_numpy()

    print(f"## 굽은 도로 확인 ({args.split})\n")
    print("| 4초 동안 차선 방향 변화 | 장면 | 차선을 바꾼 장면 | 차선을 따라간 장면 | 그중 두 구간 모두 keep lane | 애매함 표시 제외 시 |")
    print("| --- | --- | --- | --- | --- | --- |")
    for th in args.min_deg:
        m = no_conn & (lane_turn >= th)
        follow = m & ~lane_changed
        amb = f.ambiguous.to_numpy()
        print(f"| {th:.0f}° 이상 | {m.sum():,} | {(m & lane_changed).sum():,} | {follow.sum():,} | "
              f"{keep[follow].mean() * 100:.1f}% | {keep[follow & ~amb].mean() * 100:.1f}% |")
    th = args.min_deg[0]
    follow = no_conn & (lane_turn >= th) & ~lane_changed
    c = f[follow]
    dist = pd.concat([c.seg1_lat, c.seg2_lat]).value_counts(normalize=True) * 100
    print(f"\n{th:.0f}° 이상, 차선을 따라간 장면의 구간별 좌우 행동: " + ", ".join(f"{k} {v:.1f}%" for k, v in dist.items()))
    out = EXP / "step04/curved_not_keep.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    c[~keep[follow]][["token", "log_name"]].assign(split=args.split).to_parquet(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
