"""7단계: VLM 원문 출력(scripts/vlm_d3.py)을 6.1의 결정 형식으로 파싱하고, L2로 판정하여 D3로 저장한다.

판정은 6단계 CF와 같은 함수(scripts/l3_run.py의 judge)를 쓴다. VLM 결정은 사람이 실제로 한 결정이 아니므로
CF와 같이 "교차로 밖 turn의 가능 → 판정 불가" 규칙을 적용한다. 불가능이면 CF⁻와 같은 방법으로 불가능 기준
(strength / agent / road)과 보이지 않는 원인을 붙인다(세기를 낮추는 기준은 사람의 원래 결정 d0이다).

사용법: python scripts/d3_judge.py --name d3 [--out data_lists/eval/D3.parquet]
출력 열: token, log_name, scene_description, key_issue, raw_output, parse_status(ok / json_error / truncated / ...),
        decision(VLM), d_seg*(VLM 결정), human_decision, same_as_human(6.2의 ≈, 이웃 라벨 허용), same_as_human_strict,
        status(L2: feasible / infeasible / undetermined), reason, category, visibility, best_score, best_poses, n_* 등
"""

import argparse
import json
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
from yesman.decision import LAT_ACTIONS, LON_ACTIONS, Decision, Segment, same_decision
from yesman.l2 import L2Config, row_to_decision
from yesman.l3 import categorize, collision_cause_visibility, weaker

sys.path.insert(0, str(Path(__file__).parent))
import l3_run  # noqa: E402

ROOT = Path(os.environ["YESMAN_ROOT"])
EXP = Path(os.environ["NAVSIM_EXP_ROOT"])


def parse(raw: str, finish_reason: str):
    """(parse_status, scene_description, key_issue, Decision 또는 None)."""
    if finish_reason == "length":
        return "truncated", "", "", None
    try:
        o = json.loads(raw)
        segs = []
        for k in ("segment_1", "segment_2"):
            s = o["decision"][k]
            lon, lat = s["longitudinal"], s["lateral"]
            if lon not in LON_ACTIONS or lat not in LAT_ACTIONS:
                return "bad_action", o.get("scene_description", ""), o.get("key_issue", ""), None
            ls, rs = float(s["longitudinal_strength"]), float(s["lateral_strength"])
            if not (0 <= ls <= 1 and 0 <= rs <= 1):
                return "bad_strength", o.get("scene_description", ""), o.get("key_issue", ""), None
            segs.append(Segment(lon, ls, lat, 0.0 if lat == "keep_lane" else rs))
        return "ok", o["scene_description"], o["key_issue"], Decision(segs)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return "json_error", "", "", None


def judge_log(args):
    log, rows = args
    mcl = MetricCacheLoader(EXP / "metric_cache/navtest")
    loader = scene_loader("navtest", [log], [r["token"] for r in rows])
    out = []
    for r in rows:
        d0 = row_to_decision(r["human"])
        row = {"token": r["token"], "log_name": log, "human_decision": str(d0)}
        d = r["decision"]
        if d is None:
            out.append(row)
            continue
        row.update(same_as_human=bool(same_decision(d0, d)),
                   same_as_human_strict=bool(same_decision(d0, d, allow_neighbor=False)))
        scene = loader.get_scene_from_token(r["token"])
        ctx = l3_run._L2.context(scene, mcl.get_from_token(r["token"]))
        seed = zlib.crc32(f"{r['token']}:vlm".encode())
        res = l3_run.judge(ctx, d, seed, is_cf=True)
        row.update(l3_run._row(r["token"], log, "vlm", d, res, d0))
        if res.status == "infeasible":
            w = weaker(d0, d)
            wres = l3_run.judge(ctx, w, seed + 1, is_cf=True) if w is not None else None
            row["weaker_decision"] = str(w) if w is not None else ""
            row["weaker_status"] = wres.status if wres is not None else "none"
            row["category"] = categorize(res.fail_frac, wres is not None and wres.status == "feasible")
            row["visibility"] = collision_cause_visibility(scene, res.best_poses) if row["category"] == "agent" else ""
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="d3")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    raw = pd.read_parquet(EXP / f"d3/{args.name}_raw.parquet")
    scenes = pd.read_parquet(EXP / "d3/scenes.parquet").set_index("token")
    human = pd.read_parquet(EXP / "l1/navtest_labels.parquet").drop_duplicates("token").set_index("token")
    parsed = [parse(r.raw_output, r.finish_reason) for r in raw.itertuples()]
    raw["parse_status"] = [p[0] for p in parsed]
    raw["scene_description"] = [p[1] for p in parsed]
    raw["key_issue"] = [p[2] for p in parsed]
    print("parse:", raw.parse_status.value_counts().to_dict(), flush=True)

    tasks = {}
    for r, p in zip(raw.itertuples(), parsed):
        log = scenes.loc[r.token, "log_name"]
        tasks.setdefault(log, []).append({"token": r.token, "decision": p[3],
                                          "human": human.loc[r.token].to_dict()})
    tasks = sorted(tasks.items(), key=lambda t: -len(t[1]))
    t0, rows = time.time(), []
    with Pool(args.workers, initializer=l3_run.init_worker, initargs=(L2Config().__dict__,)) as pool:
        for i, r in enumerate(pool.imap_unordered(judge_log, tasks), 1):
            rows.extend(r)
            if i % 20 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} logs, {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    judged = pd.DataFrame(rows)
    keep = ["token", "parse_status", "scene_description", "key_issue", "raw_output", "finish_reason",
            "n_prompt_tokens", "n_output_tokens"]
    df = raw[keep].merge(judged, on="token", how="left")
    df.loc[df.parse_status != "ok", "status"] = "parse_failed"
    df["cf_name"] = "vlm"  # D0~D2의 cf_name(original / CF 이름)과 맞춘다
    df = df.sort_values("token").reset_index(drop=True)
    out = args.out or EXP / f"d3/{args.name}.parquet"
    df.to_parquet(out, index=False)
    print(f"saved {out} ({time.time() - t0:.0f}s)")
    print(df.status.value_counts().to_string())


if __name__ == "__main__":
    main()
