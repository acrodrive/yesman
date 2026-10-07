"""따르기 판정 (yesman.md 6.2, 9.2): 모델이 그린 경로에 L1을 돌려 d̂을 구하고, 주어진 결정 d와 비교한다.

- 따르기 = d̂ ≈ d (6.2, 14절: 구간마다 행동 종류가 같고 세기 차이 ±0.2. 4단계 결정대로 이웃 라벨 허용이 주 지표,
  엄격 기준도 함께 낸다). 판단 head가 없는 모델(B1, B2, B-순응)은 항상 ACCEPT로 본다.
- d̂을 정할 수 없으면(invalid, 구조적 애매함) "따르지 않음"으로 센다(결과를 보기 전에 정함). 그 비율을 따로 낸다.
- 세기 오차 (9.2): 행동 종류가 정확히 같은 구간에서 |세기(d) − 세기(d̂)|. keep lane은 뺀다.
L1은 라벨을 만들 때와 같은 함수(yesman/l1.py classify + extract_features)이다(12절 "L1의 단일성").

사용법: python scripts/follow_eval.py --run B2_seed0 [--sets D0 D1 val]
출력: exp/eval/<run>/<set>_l1.parquet (행마다 d̂, follow, follow_strict, dhat_usable, 세기 오차),
     exp/eval/<run>/follow_summary.json
"""

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from yesman.data import scene_loader, scene_without_sensors
from yesman.decision import same_decision
from yesman.l1 import classify, extract_features, load_thresholds
from yesman.l2 import row_to_decision

EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
_TH = None


def _init():
    global _TH
    _TH = load_thresholds()


def judge_log(args):
    split, log, rows = args
    loader = scene_loader(split, [log], sorted({r["token"] for r in rows}))
    ctx = {}
    out = []
    for r in rows:
        tok = r["token"]
        if tok not in ctx:
            sc = scene_without_sensors(loader, tok)
            st = sc.frames[sc.scene_metadata.num_history_frames - 1].ego_status
            ctx[tok] = (float(np.linalg.norm(st.ego_velocity[:2])), np.asarray(st.ego_pose), sc.map_api)
        v0, pose, map_api = ctx[tok]
        d = row_to_decision({k[2:]: v for k, v in r.items() if k.startswith("d_seg")})
        try:
            dh = classify(extract_features(np.asarray(r["pred_poses"]).reshape(8, 3), v0, pose, map_api), _TH)
        except Exception as e:  # noqa: BLE001
            out.append({"i": r["i"], "dhat": f"error {type(e).__name__}", "dhat_usable": False, "follow": False,
                        "follow_strict": False})
            continue
        usable = dh.usable
        row = {"i": r["i"], "dhat": str(dh), "dhat_usable": usable,
               "follow": bool(usable and same_decision(d, dh)),
               "follow_strict": bool(usable and same_decision(d, dh, allow_neighbor=False))}
        if usable:
            for k, (a, b) in enumerate(zip(d.segments, dh.segments), 1):
                row[f"seg{k}_lon_err"] = abs(a.lon_strength - b.lon_strength) if a.lon == b.lon else np.nan
                row[f"seg{k}_lat_err"] = (abs(a.lat_strength - b.lat_strength)
                                          if a.lat == b.lat and a.lat != "keep_lane" else np.nan)
            for k, s in enumerate(dh.segments, 1):
                row.update({f"dhat_seg{k}_lon": s.lon, f"dhat_seg{k}_lat": s.lat})
        out.append(row)
    return out


def run_set(run, name, workers):
    df = pd.read_parquet(EXP / f"eval/{run}/{name}.parquet")
    df["i"] = np.arange(len(df))
    split = "navtrain" if name in ("val", "train5k") else "navtest"
    cols = ["i", "token", "pred_poses"] + [c for c in df.columns if c.startswith("d_seg")]
    tasks = [(split, log, g[cols].to_dict("records")) for log, g in df.groupby("log_name")]
    tasks.sort(key=lambda t: -len(t[2]))
    t0, rows = time.time(), []
    with Pool(workers, initializer=_init) as pool:
        for r in pool.imap_unordered(judge_log, tasks):
            rows.extend(r)
    res = pd.DataFrame(rows).set_index("i").sort_index()
    keep = [c for c in ("token", "log_name", "cf_name", "sample_type", "change_type", "flag", "category",
                        "visibility", "status", "p") if c in df.columns]
    out = pd.concat([df[keep], res], axis=1)
    out.to_parquet(EXP / f"eval/{run}/{name}_l1.parquet", index=False)
    print(f"[{run}] {name}: {len(out):,} rows, {time.time() - t0:.0f}s", flush=True)
    return out


def summarize(out: pd.DataFrame, by=None):
    def s(g):
        errs = g.filter(regex=r"seg\d_(lon|lat)_err")
        return {"n": int(len(g)), "follow": float(g.follow.mean()), "follow_strict": float(g.follow_strict.mean()),
                "dhat_unusable": float(1 - g.dhat_usable.mean()),
                "lon_strength_err": float(np.nanmean(errs.filter(like="lon").to_numpy())),
                "lat_strength_err": float(np.nanmean(errs.filter(like="lat").to_numpy()))
                if errs.filter(like="lat").notna().any().any() else None}
    res = {"all": s(out)}
    if by and by in out:
        res.update({f"{by}={k}": s(g) for k, g in out.groupby(by)})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--sets", nargs="+", default=["D0", "D1", "val"])
    ap.add_argument("--workers", type=int, default=28)
    args = ap.parse_args()
    summ_path = EXP / f"eval/{args.run}/follow_summary.json"
    summ = json.load(open(summ_path)) if summ_path.exists() else {}
    for name in args.sets:
        out = run_set(args.run, name, args.workers)
        by = {"D1": "cf_name", "val": "sample_type", "train5k": "sample_type", "D2": "category"}.get(name)
        summ[name] = summarize(out, by)
        print(json.dumps(summ[name]["all"]))
    json.dump(summ, open(summ_path, "w"), indent=1)


if __name__ == "__main__":
    main()
