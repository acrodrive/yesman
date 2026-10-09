"""12단계: 학습 묶음 CF⁺의 목표 경로를 합성 경로로 바꾼다 (yesman/synth.py).

순서 (결과를 보기 전에 정함): 합성 후보 → L1이 결정과 맞다고 판정(세기 ±0.1, 이웃 라벨 허용 없음, 구조적 애매함 없음)
→ 사람 경로에 가까운 순서로 최대 3개를 NAVSIM v2로 따로 채점(human filter) → NC, DAC, DDC를 모두 통과한 첫 경로를 목표로 쓴다.
통과한 합성 경로가 없으면 10단계의 일관화된 목표(train_bundle_*_retarget.parquet)를 그대로 둔다.
판정 결과(CF⁺/CF⁻)와 평가 세트는 바꾸지 않는다.

사용법: python scripts/l3_synth.py --bundle train   (그리고 --bundle val)   [--max_logs 2 : 시험]
출력: exp/l3/train_bundle_<bundle>_synth.parquet (CF⁺ 행의 target_poses만 바뀜, target_rule = synth_* / fallback_*)
"""

import argparse
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from navsim.common.dataloader import MetricCacheLoader
from yesman.data import scene_loader
from yesman.l1 import load_thresholds
from yesman.l2 import row_to_decision
from yesman.scoring import BatchScorer
from yesman.synth import candidates, matching

EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
SAFE = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance"]
_S, _TH = None, None


def _init():
    global _S, _TH
    _S, _TH = BatchScorer(human_penalty_filter=True), load_thresholds()


def dec(r):
    return row_to_decision({k[2:]: v for k, v in r.items() if k.startswith("d_seg")})


def run_log(args):
    log, rows, originals = args
    mcl = MetricCacheLoader(EXP / "metric_cache/navtrain_sensor")
    loader = scene_loader("navtrain", [log], sorted({r["token"] for r in rows}))
    out, cur = [], {}
    for r in rows:
        tok = r["token"]
        try:
            if tok not in cur:
                sc = loader.get_scene_from_token(tok)
                st = sc.frames[sc.scene_metadata.num_history_frames - 1].ego_status
                mc = mcl.get_from_token(tok)
                cur = {tok: (sc.get_future_trajectory(8).poses, float(np.linalg.norm(st.ego_velocity[:2])),
                             np.asarray(st.ego_pose), sc.map_api, mc, _S.human_metrics(mc))}
            human, v0, pose, map_api, mc, hm = cur[tok]
            d, d0 = dec(r), dec(originals[tok])
            cands = candidates(d, d0, human, v0, pose, map_api, _TH)
            ok = matching(d, cands, v0, pose, map_api, human, _TH)
            chosen, n_scored = None, 0
            for ade, name, p in ok[:3]:
                n_scored += 1
                res = _S.score(mc, [p], hm).iloc[0]
                if all(res[m] >= 1 for m in SAFE):
                    chosen = (name, p, ade)
                    break
            if chosen:
                out.append({"i": r["i"], "rule": f"synth_{chosen[0].split('_')[0]}", "poses": chosen[1].astype(np.float32),
                            "ade_to_human": chosen[2], "n_cand": len(cands), "n_match": len(ok), "n_scored": n_scored})
            else:
                why = "no_candidate" if not cands else "no_l1_match" if not ok else "unsafe"
                out.append({"i": r["i"], "rule": f"fallback_{why}", "poses": None, "n_cand": len(cands),
                            "n_match": len(ok), "n_scored": n_scored})
        except Exception as e:  # noqa: BLE001
            out.append({"i": r["i"], "rule": f"fallback_error {type(e).__name__}: {e}", "poses": None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", choices=["train", "val"], required=True)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--max_logs", type=int, default=0, help="시험용: 작은 로그 N개만 (저장하지 않는다)")
    ap.add_argument("--input", type=Path, default=None, help="12단계 A: 이 묶음의 target_rule == new_l2best인 CF⁺만 합성한다")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    df = pd.read_parquet(args.input or EXP / f"l3/train_bundle_{args.bundle}_retarget.parquet")
    df["i"] = np.arange(len(df))
    cols = ["i", "token", "cf_name"] + [c for c in df.columns if c.startswith("d_seg")]
    orig = {r["token"]: r for r in df[df.sample_type == "original"][cols].to_dict("records")}
    pos = df[df.sample_type == "cf_pos"]
    if args.input:
        pos = pos[pos.target_rule == "new_l2best"]
    tasks = sorted(((log, g.sort_values("token")[cols].to_dict("records"), {t: orig[t] for t in g.token.unique()})
                    for log, g in pos.groupby("log_name")), key=lambda t: -len(t[1]))
    if args.max_logs:
        tasks = tasks[-args.max_logs:]
    t0, rows = time.time(), []
    with Pool(args.workers, initializer=_init) as pool:
        for k, r in enumerate(pool.imap_unordered(run_log, tasks), 1):
            rows.extend(r)
            if k % 20 == 0 or k == len(tasks):
                print(f"{k}/{len(tasks)} logs, {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    res = pd.DataFrame(rows).set_index("i")
    print("rule:", res.rule.str.split(" ").str[0].value_counts().to_dict())
    if "ade_to_human" in res:
        print(f"synth target ADE to human: mean {res.ade_to_human.mean():.2f} m, median {res.ade_to_human.median():.2f} m")
    by = pd.concat([res, df.loc[res.index, "cf_name"]], axis=1).groupby("cf_name").rule.apply(
        lambda s: s.str.startswith("synth").mean()).round(3)
    print("synth rate by CF:", by.to_dict())
    if args.max_logs:
        return
    if not args.input:
        df["target_rule"] = df.target_rule.where(df.sample_type != "cf_pos", "")
    df.loc[res.index, "target_rule"] = res.rule
    new = df.target_poses.copy()
    for i, p in res.poses.dropna().items():
        new.at[i] = p.ravel().tolist()
    df["target_poses"] = new
    out = args.out or EXP / f"l3/train_bundle_{args.bundle}_synth.parquet"
    df.drop(columns="i").to_parquet(out, index=False)
    print(f"saved {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
