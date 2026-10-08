"""11단계: 표의 행마다 경로를 NAVSIM v2 채점 도구로 채점한다 (한 장면에 경로가 여러 개인 D2, D3용).

공식 스크립트는 장면마다 경로 하나만 채점하므로, L2와 같은 BatchScorer(yesman/scoring.py, human filter 적용,
3단계에서 공식 결과와 같음을 확인)를 쓰고, 경로마다 따로 채점한다(EP 정규화가 함께 채점한 경로에 따라 달라지므로). EC(two-frame extended comfort)는 이웃 프레임이 필요하므로 없다.
점수 = 곱하는 항목(NC, DAC, DDC, TLC) x 가중 평균(EP 5, TTC 5, LK 2, HC 2) (L2의 점수와 같다).

사용법:
  python scripts/score_rows.py --table exp/eval/Ours_w025_rt_seed0/D2.parquet           # pred_poses 채점
  python scripts/score_rows.py --table data_lists/eval/D3.parquet --col best_poses --out exp/eval/D3_best_scores.parquet
출력: 기본은 <table>_scores.parquet (원래 행 순서, 항목별 점수 + score)
"""

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from navsim.common.dataloader import MetricCacheLoader
from yesman.scoring import BatchScorer

EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
METRICS = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
           "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound", "lane_keeping",
           "history_comfort"]
W = {"ego_progress": 5, "time_to_collision_within_bound": 5, "lane_keeping": 2, "history_comfort": 2}
MULT = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance", "traffic_light_compliance"]
_S = None


def _init():
    global _S
    _S = BatchScorer(human_penalty_filter=True)


def score_group(args):
    token, items = args
    mcl = MetricCacheLoader(EXP / "metric_cache/navtest")
    mc = mcl.get_from_token(token)
    human = _S.human_metrics(mc)
    out = []
    for i, p in items:
        # 경로마다 따로 채점한다: EP(진행 거리)는 함께 채점한 경로 중 최대 진행으로 정규화되므로,
        # 공식 채점(장면마다 경로 하나)과 같게 하려면 한 번에 하나씩 채점해야 한다
        r = _S.score(mc, [p], human).iloc[0]
        row = {"i": i, **{m: float(r[m]) for m in METRICS}}
        row["score"] = float(np.prod([row[m] for m in MULT]) * sum(row[m] * w for m, w in W.items()) / sum(W.values()))
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--col", default="pred_poses")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=28)
    args = ap.parse_args()
    df = pd.read_parquet(args.table, columns=["token", args.col])
    has = df[args.col].map(lambda p: p is not None and len(p) == 24)
    groups = {}
    for i, (t, p) in enumerate(zip(df.token, df[args.col])):
        if has.iloc[i]:
            groups.setdefault(t, []).append((i, np.asarray(p, dtype=np.float64).reshape(8, 3)))
    t0, rows = time.time(), []
    with Pool(args.workers, initializer=_init) as pool:
        for r in pool.imap_unordered(score_group, list(groups.items()), chunksize=8):
            rows.extend(r)
    res = pd.DataFrame(rows).set_index("i").reindex(range(len(df)))
    res.insert(0, "token", df.token.values)
    out = args.out or args.table.with_name(args.table.stem + "_scores.parquet")
    res.to_parquet(out, index=False)
    print(f"{args.table} [{args.col}]: {int(has.sum()):,} paths in {len(groups):,} scenes, {time.time() - t0:.0f}s -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
