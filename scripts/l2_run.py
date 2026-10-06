"""5단계: L2로 navtest 장면의 결정을 판정한다 (병렬, 로그 단위).

--decisions original: 사람의 원래 결정 (L1 라벨). 구조적 애매함이 있는 장면은 뺀다.
--decisions probe: L2 확인용으로 일부러 바꾼 결정 4가지 (6단계 counterfactual의 맛보기)
    lc_left / lc_right: 구간 1에 차선 변경(세기 0.5), 구간 2는 keep lane. 앞뒤 행동은 원래대로
    turn_left: 두 구간 모두 좌회전(세기 0.6). 앞뒤 행동은 원래대로
    go_fast: 두 구간 모두 go 0.9. 좌우 행동은 원래대로

출력: exp/l2/<split>_<decisions>.parquet (장면 x 결정마다 한 줄: 판정, 판단 근거, 후보 수, 최고 후보 행 번호 등)

사용법: python scripts/l2_run.py --decisions original --workers 16
        python scripts/l2_run.py --decisions probe --max_scenes 2000 --workers 16
"""

import argparse
import os
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from navsim.common.dataloader import MetricCacheLoader
from yesman.data import scene_loader
from yesman.decision import Decision, Segment
from yesman.l2 import L2, L2Config, PathBank, row_to_decision

EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
_L2 = None


def probes(d: Decision):
    """원래 결정 d에서 시험용 결정들을 만든다."""
    s1, s2 = d.segments
    out = {}
    for side in ("left", "right"):
        out[f"lc_{side}"] = Decision([Segment(s1.lon, s1.lon_strength, f"lane_change_{side}", 0.5),
                                      Segment(s2.lon, s2.lon_strength, "keep_lane", 0.0)])
    out["turn_left"] = Decision([Segment(s.lon, s.lon_strength, "turn_left", 0.6) for s in (s1, s2)])
    out["go_fast"] = Decision([Segment("go", 0.9, s.lat, s.lat_strength) for s in (s1, s2)])
    # 원래 결정과 같은 시험 결정은 빼지 않는다 (예: 원래 좌회전이면 turn_left는 원래 결정과 같다). 표에 표시한다.
    return out


def init_worker(cfg_dict):
    global _L2
    _L2 = L2(PathBank(), L2Config(**cfg_dict))


def run_log(args):
    split, log, rows, mode = args
    mcl = MetricCacheLoader(EXP / f"metric_cache/{split}")
    loader = scene_loader(split, [log], [r["token"] for r in rows])
    out = []
    for r in rows:
        d = row_to_decision(r)
        try:
            scene = loader.get_scene_from_token(r["token"])
            ctx = _L2.context(scene, mcl.get_from_token(r["token"]))
            decs = {"original": d} if mode == "original" else probes(d)
            for name, dd in decs.items():
                t0 = time.time()
                seed = zlib.crc32(f"{r['token']}:{name}".encode())  # 장면과 결정마다 고정된 시드 (재현 가능)
                res = _L2.judge(ctx, dd, np.random.default_rng(seed))
                row = res.to_flat()
                row.update(token=r["token"], log_name=log, decision_type=name, decision=str(dd),
                           seconds=time.time() - t0, scored_bank_idx=res.scored_bank_idx, scored_pass=res.scored_pass,
                           same_as_original=str(dd) == str(d))
                out.append(row)
        except Exception as e:
            out.append({"token": r["token"], "log_name": log, "status": "error", "reason": f"{type(e).__name__}: {e}"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--decisions", default="original", choices=["original", "probe"])
    ap.add_argument("--max_scenes", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lab = pd.read_parquet(EXP / f"l1/{args.split}_labels.parquet")
    usable = lab.valid & ~lab.filter(like="structural").fillna(False).any(axis=1)
    print(f"{args.split}: scenes {len(lab):,}, usable (valid, no structural ambiguity) {usable.sum():,}")
    lab = lab[usable]
    if args.max_scenes:
        lab = lab.sample(n=min(args.max_scenes, len(lab)), random_state=args.seed)
    tasks = [(args.split, log, g.to_dict("records"), args.decisions) for log, g in lab.groupby("log_name")]
    cfg = L2Config().__dict__
    t0, rows = time.time(), []
    with Pool(args.workers, initializer=init_worker, initargs=(cfg,)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_log, tasks), 1):
            rows.extend(r)
            if i % 20 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} logs, {len(rows)} judgments, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    out = EXP / f"l2/{args.split}_{args.decisions}.parquet"
    df.to_parquet(out, index=False)
    print(f"saved {out} ({time.time() - t0:.0f}s)")
    print(df.groupby("decision_type").status.value_counts(normalize=True).unstack().round(3).to_string())


if __name__ == "__main__":
    main()
