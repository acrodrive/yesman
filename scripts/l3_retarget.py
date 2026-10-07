"""10단계 시험 (가): 학습 묶음 CF⁺의 목표 경로를 일관된 규칙으로 다시 고른다.

6단계는 L2를 통과한 후보 중 "점수가 가장 높은 경로"를 CF⁺ 목표로 썼다. 이 경로는 다른 로그의 사람 경로라 같은 결정에도
장면마다 모양(시작 시점, 옆 이동 속도, 속도 변화)이 제각각이다. 여기서는 같은 판정을 같은 시드로 다시 돌려 같은 후보를
얻고(scripts/l3_run.py의 judge), 목표만 다음 규칙으로 다시 고른다 (결과를 보기 전에 정함):
  1. 통과한 후보 중 세기가 결정과 ±0.1 안인 것 (구간마다 앞뒤 세기, keep lane이 아닌 좌우 세기)
  2. 그런 후보가 없으면 통과한 후보 전체
  3. 그중 사람의 실제 경로와 평균 거리가 가장 가까운 경로 ("결정을 만족하는 가장 작은 변경")
판정 결과(가능/불가능)는 바꾸지 않는다. 다시 돌린 판정이 "가능"이 아니면(재현 실패) 원래 목표를 그대로 두고 센다.
평가 세트(D0~D3)는 건드리지 않는다.

사용법: python scripts/l3_retarget.py --bundle train   (그리고 --bundle val)
출력: exp/l3/train_bundle_<bundle>_retarget.parquet (CF⁺ 행의 target_poses만 바뀜, target_rule 열 추가)
"""

import argparse
import os
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from navsim.common.dataloader import MetricCacheLoader
from yesman.data import scene_loader
from yesman.l2 import L2Config, row_to_decision

sys.path.insert(0, str(Path(__file__).parent))
import l3_run  # noqa: E402

EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
TOL = 0.1


def strength_err(d, dh) -> float:
    e = 0.0
    for a, b in zip(d.segments, dh.segments):
        if a.lon == b.lon:
            e = max(e, abs(a.lon_strength - b.lon_strength))
        if a.lat == b.lat and a.lat != "keep_lane":
            e = max(e, abs(a.lat_strength - b.lat_strength))
    return e


def run_log(args):
    log, rows = args
    mcl = MetricCacheLoader(EXP / "metric_cache/navtrain_sensor")
    loader = scene_loader("navtrain", [log], sorted({r["token"] for r in rows}))
    out, ctxs = [], {}
    for r in rows:
        tok = r["token"]
        try:
            if tok not in ctxs:
                scene = loader.get_scene_from_token(tok)
                ctxs = {tok: (l3_run._L2.context(scene, mcl.get_from_token(tok)),
                              scene.get_future_trajectory(8).poses)}
            ctx, human = ctxs[tok]
            d = row_to_decision({k[2:]: v for k, v in r.items() if k.startswith("d_seg")})
            res = l3_run.judge(ctx, d, zlib.crc32(f"{tok}:{r['cf_name']}".encode()), is_cf=True)
            if res.status != "feasible":
                out.append({"i": r["i"], "rule": f"rejudge_{res.status}", "poses": None})
                continue
            cands = []
            for idx, ok, poses in zip(res.scored_bank_idx, res.scored_pass, res.scored_poses):
                if not ok:
                    continue
                dh = ctx.relabel_cache[(idx, res.mode)][1]
                ade = float(np.linalg.norm(poses[:, :2] - human[:, :2], axis=1).mean())
                cands.append((strength_err(d, dh), ade, poses))
            tight = [c for c in cands if c[0] <= TOL]
            pool = tight or cands
            best = min(pool, key=lambda c: c[1])
            old = np.asarray(r["target_poses"]).reshape(8, 3)
            out.append({"i": r["i"], "rule": "tight" if tight else "all_pass", "poses": best[2].astype(np.float32),
                        "n_pass": len(cands), "n_tight": len(tight),
                        "moved": float(np.linalg.norm(best[2][:, :2] - old[:, :2], axis=1).mean()),
                        "old_close": bool(np.allclose(old, res.best_poses, atol=1e-3))})
        except Exception as e:  # noqa: BLE001
            out.append({"i": r["i"], "rule": f"error {type(e).__name__}: {e}", "poses": None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", choices=["train", "val"], required=True)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--max_logs", type=int, default=0, help="시험용: 작은 로그 N개만 (저장하지 않는다)")
    args = ap.parse_args()
    df = pd.read_parquet(EXP / f"l3/train_bundle_{args.bundle}.parquet")
    df["i"] = np.arange(len(df))
    pos = df[df.sample_type == "cf_pos"]
    cols = ["i", "token", "cf_name", "target_poses"] + [c for c in df.columns if c.startswith("d_seg")]
    tasks = sorted(((log, g.sort_values("token")[cols].to_dict("records")) for log, g in pos.groupby("log_name")),
                   key=lambda t: -len(t[1]))  # 장면 순서로 (장면 정보를 한 번만 만든다), 큰 로그부터
    if args.max_logs:
        tasks = tasks[-args.max_logs:]
    t0, rows = time.time(), []
    with Pool(args.workers, initializer=l3_run.init_worker, initargs=(L2Config().__dict__,)) as pool:
        for k, r in enumerate(pool.imap_unordered(run_log, tasks), 1):
            rows.extend(r)
            if k % 20 == 0 or k == len(tasks):
                print(f"{k}/{len(tasks)} logs, {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    res = pd.DataFrame(rows).set_index("i")
    df["target_rule"] = ""
    df.loc[df.sample_type != "cf_pos", "target_rule"] = "unchanged"
    df.loc[res.index, "target_rule"] = res.rule
    ok = res.poses.notna()
    new = df.target_poses.copy()
    for i, p in res.poses[ok].items():
        new.at[i] = p.ravel().tolist()
    df["target_poses"] = new
    df = df.drop(columns="i")
    out = EXP / f"l3/train_bundle_{args.bundle}_retarget.parquet"
    if not args.max_logs:
        df.to_parquet(out, index=False)
        print(f"saved {out} ({time.time() - t0:.0f}s)")
    print("rule:", res.rule.str.split(" ").str[0].value_counts().to_dict())
    if "old_close" in res:
        print(f"re-judge reproduced old target: {res.old_close.dropna().mean():.3f}; "
              f"target moved mean {res.moved.mean():.2f} m, median {res.moved.median():.2f} m; "
              f"passing candidates median {res.n_pass.median():.0f}, tight median {res.n_tight.median():.0f}")


if __name__ == "__main__":
    main()
