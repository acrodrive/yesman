"""6단계: 장면마다 원래 결정과 CF 결정을 L2로 판정하고, CF⁻를 불가능 기준으로 분류한다 (병렬, 로그 단위).

입력: L1 라벨(exp/l1/<split>_labels.parquet), 경로 모음, metric cache(exp/metric_cache/<cache>)
출력: exp/l3/<name>.parquet — 장면 x 결정(원래 + CF)마다 한 줄
  열: token, log_name, cf_name(original / keep / lc_L ...), change_type, d_seg*_ (결정), status, reason,
      fail_*, category(strength/agent/road, CF⁻만), visibility(visible/unseen/unknown, agent 기준만),
      weaker_status, best_poses(가능: 목표 경로, 불가능: 떨어진 후보 중 최고), n_* 후보 수 등

사용법:
  python scripts/l3_run.py --split navtest --name navtest_cf                 # 평가 세트용: CF 메뉴 전부
  python scripts/l3_run.py --split navtrain --cache navtrain_sensor --tokens data_lists/navtrain_sensor_tokens.txt \
      --n_cf 4 --skip_original --name navtrain_sensor_cf                     # 학습용: 장면마다 CF 4개 무작위
"""

import argparse
import os
import time
import traceback
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from navsim.common.dataloader import MetricCacheLoader
from yesman.data import scene_loader
from yesman.decision import Decision
from yesman.l2 import L2, L2Config, PathBank, row_to_decision
from yesman.l3 import categorize, cf_menu, change_type, collision_cause_visibility, weaker

EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
_L2 = None


def init_worker(cfg_dict):
    global _L2
    _L2 = L2(PathBank(), L2Config(**cfg_dict))


def _row(token, log, name, d: Decision, res, d0: Decision):
    row = res.to_flat()
    row.update(token=token, log_name=log, cf_name=name, change_type="original" if name == "original" else
               change_type(d0, d), decision=str(d))
    for k, v in d.to_flat().items():
        if k.startswith("seg"):
            row[f"d_{k}"] = v
    row["best_poses"] = (res.best_poses.astype("float32").ravel().tolist() if res.best_poses is not None else [])
    return row


def run_log(args):
    split, cache, log, rows, n_cf, skip_original = args
    mcl = MetricCacheLoader(EXP / f"metric_cache/{cache}")
    loader = scene_loader(split, [log], [r["token"] for r in rows])
    out = []
    for r in rows:
        t0 = time.time()
        try:
            scene = loader.get_scene_from_token(r["token"])
            ctx = _L2.context(scene, mcl.get_from_token(r["token"]))
            d0 = row_to_decision(r)
            menu = cf_menu(d0, ctx.v0)
            names = sorted(menu)
            if n_cf and len(names) > n_cf:
                rng = np.random.default_rng(zlib.crc32(f"{r['token']}:menu".encode()))
                names = sorted(rng.choice(names, size=n_cf, replace=False))
            decs = {"original": d0, **{k: menu[k] for k in names}}
            for name, d in decs.items():
                seed = zlib.crc32(f"{r['token']}:{name}".encode())
                if name == "original" and skip_original:  # 학습 장면: 사람이 실제로 한 결정이므로 판정하지 않는다
                    out.append({"token": r["token"], "log_name": log, "cf_name": "original", "status": "human",
                                "change_type": "original", "decision": str(d),
                                **{f"d_{k}": v for k, v in d.to_flat().items() if k.startswith("seg")}})
                    continue
                res = _L2.judge(ctx, d, np.random.default_rng(seed))
                row = _row(r["token"], log, name, d, res, d0)
                if name != "original" and res.status == "infeasible":
                    w = weaker(d0, d)
                    wres = _L2.judge(ctx, w, np.random.default_rng(seed + 1)) if w is not None else None
                    row["weaker_decision"] = str(w) if w is not None else ""
                    row["weaker_status"] = wres.status if wres is not None else "none"
                    row["category"] = categorize(res.fail_frac, wres is not None and wres.status == "feasible")
                    row["visibility"] = (collision_cause_visibility(scene, res.best_poses)
                                         if row["category"] == "agent" else "")
                out.append(row)
        except Exception as e:
            out.append({"token": r["token"], "log_name": log, "cf_name": "error",
                        "status": "error", "reason": f"{type(e).__name__}: {e} | {traceback.format_exc(limit=2)}"})
        out[-1]["scene_seconds"] = time.time() - t0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--cache", default=None, help="metric cache 폴더 이름 (기본: split)")
    ap.add_argument("--tokens", type=Path, default=None, help="이 토큰들만 (예: 센서를 받은 navtrain 장면)")
    ap.add_argument("--logs", type=Path, default=None, help="이 로그들만 (줄마다 로그 이름)")
    ap.add_argument("--n_cf", type=int, default=0, help="장면마다 CF를 이만큼 무작위로 고른다 (0: 메뉴 전부)")
    ap.add_argument("--skip_original", action="store_true", help="원래 결정은 L2로 판정하지 않는다 (학습 장면)")
    ap.add_argument("--max_scenes", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    lab = pd.read_parquet(EXP / f"l1/{args.split}_labels.parquet")
    usable = lab.valid & ~lab.filter(like="structural").fillna(False).any(axis=1)
    lab = lab[usable]
    if args.tokens:
        lab = lab[lab.token.isin(set(args.tokens.read_text().split()))]
    if args.logs:
        lab = lab[lab.log_name.isin(set(args.logs.read_text().split()))]
    if args.max_scenes:
        lab = lab.sample(n=min(args.max_scenes, len(lab)), random_state=0)
    print(f"{args.split}: {len(lab):,} usable scenes in {lab.log_name.nunique()} logs", flush=True)
    tasks = [(args.split, args.cache or args.split, log, g.to_dict("records"), args.n_cf, args.skip_original)
             for log, g in lab.groupby("log_name")]
    tasks.sort(key=lambda t: -len(t[3]))  # 큰 로그부터 (병렬 끝부분의 빈 시간을 줄인다)
    t0, rows = time.time(), []
    with Pool(args.workers, initializer=init_worker, initargs=(L2Config().__dict__,)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_log, tasks), 1):
            rows.extend(r)
            if i % 10 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} logs, {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    out = EXP / f"l3/{args.name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"saved {out} ({time.time() - t0:.0f}s), errors {int((df.status == 'error').sum())}")
    print(df.groupby("cf_name").status.value_counts(normalize=True).unstack().round(3).to_string())
    if "category" in df:
        print(df.groupby("category").visibility.value_counts().to_string())


if __name__ == "__main__":
    main()
