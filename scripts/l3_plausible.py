"""12단계 C: 그럴싸한 틀린 결정(CF⁻)을 만든다. VLM이 틀리는 방식(겉보기로 보통 할 법한 결정을 내지만 움직임이
달라 틀림)을 흉내 낸다. VLM은 쓰지 않는다. 규칙은 결과를 보기 전에 정했다(2026-10-09).

학습 장면(navtrain 센서 장면, LTF 특징이 있는 23,388개)마다 후보 결정:
- lookalike_1..10: 다른 로그의 장면 중 내비게이션 명령이 같고 ego 속도 차이가 2 m/s 이내인 것에서, LTF 장면 토큰
  평균(keyval 65개의 평균, 길이 1로 정규화)의 cosine 유사도가 가장 높은 10개 장면의 사람 결정(L1 라벨)
- past: 같은 로그에서 1.0~2.5초 전 장면(1.5초 전에 가장 가까운 것)의 사람 결정
L1 라벨이 invalid이거나 지금 장면의 사람 결정과 같으면(6.2의 ≈, 이웃 라벨 허용) 뺀다. 같은 장면에서 겹치는 후보도 뺀다.
판정은 6단계 CF와 같다(scripts/l3_run.py judge, 불가능이면 불가능 기준과 시야).

사용법: python scripts/l3_plausible.py [--max_logs 2 : 시험]
출력: exp/l3/navtrain_sensor_cf_plausible.parquet (6단계 CF 판정 표와 같은 열, cf_name = lookalike_k / past)
"""

import argparse
import os
import sys
import time
import traceback
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from navsim.common.dataloader import MetricCacheLoader
from yesman.data import scene_loader
from yesman.decision import same_decision
from yesman.l2 import L2Config, row_to_decision
from yesman.l3 import categorize, collision_cause_visibility, weaker

sys.path.insert(0, str(Path(__file__).parent))
import l3_run  # noqa: E402

ROOT = Path(os.environ["YESMAN_ROOT"])
EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
FEAT = ROOT / "exp/features/ltf/navtrain"
K, DV, PAST = 10, 2.0, (1.0, 2.5, 1.5)  # K: 처음 4로 정했으나 시험(로그 24개)에서 불가능이 2.7%라 10으로 늘림 (모델 결과 보기 전)


def lookalikes(tokens, logs):
    """장면마다 비슷해 보이는 다른 로그 장면 K개의 행 번호."""
    kv = torch.from_numpy(np.load(FEAT / "keyval.npy")).cuda().float().mean(1)
    emb = torch.nn.functional.normalize(kv, dim=1)
    st = torch.from_numpy(np.load(FEAT / "status.npy")).cuda().float()
    cmd, v = st[:, :4].argmax(1), st[:, 4:6].norm(dim=1)
    log_id = torch.as_tensor(pd.factorize(pd.Series(logs))[0]).cuda()
    out = []
    for lo in range(0, len(tokens), 2048):
        sl = slice(lo, lo + 2048)
        sim = emb[sl] @ emb.T
        bad = (log_id[sl, None] == log_id[None]) | (cmd[sl, None] != cmd[None]) | ((v[sl, None] - v[None]).abs() > DV)
        sim[bad] = -2
        top = sim.topk(K, dim=1)
        out.append(torch.where(top.values > -2, top.indices, -1).cpu().numpy())
    return np.concatenate(out)


def judge_log(args):
    log, rows, log_tokens = args
    mcl = MetricCacheLoader(EXP / "metric_cache/navtrain_sensor")
    loader = scene_loader("navtrain", [log], sorted(set(log_tokens) | {r["token"] for r in rows}))
    nh = loader._scene_filter.num_history_frames
    ts = {t: loader.scene_frames_dicts[t][nh - 1]["timestamp"] / 1e6 for t in loader.tokens}
    out = []
    for r in rows:
        try:
            d0 = row_to_decision(r["human"])
            cands = dict(r["lookalike"])
            prev = [(abs((ts[r["token"]] - ts[t]) - PAST[2]), t) for t in ts
                    if PAST[0] <= ts[r["token"]] - ts[t] <= PAST[1] and t in log_tokens]
            if prev:
                cands["past"] = log_tokens[min(prev)[1]]
            decs = {}
            for name, lab in cands.items():
                if not lab.get("valid", False):
                    continue
                d = row_to_decision(lab)
                if same_decision(d0, d) or any(same_decision(x, d, allow_neighbor=False) for x in decs.values()):
                    continue
                decs[name] = d
            if not decs:
                continue
            scene = loader.get_scene_from_token(r["token"])
            ctx = l3_run._L2.context(scene, mcl.get_from_token(r["token"]))
            for name, d in decs.items():
                seed = zlib.crc32(f"{r['token']}:{name}".encode())
                res = l3_run.judge(ctx, d, seed, is_cf=True)
                row = l3_run._row(r["token"], log, name, d, res, d0)
                if res.status == "infeasible":
                    w = weaker(d0, d)
                    wres = l3_run.judge(ctx, w, seed + 1, is_cf=True) if w is not None else None
                    row["weaker_decision"] = str(w) if w is not None else ""
                    row["weaker_status"] = wres.status if wres is not None else "none"
                    row["category"] = categorize(res.fail_frac, wres is not None and wres.status == "feasible")
                    row["visibility"] = (collision_cause_visibility(scene, res.best_poses)
                                         if row["category"] == "agent" else "")
                out.append(row)
        except Exception as e:  # noqa: BLE001
            out.append({"token": r["token"], "log_name": log, "cf_name": "error", "status": "error",
                        "reason": f"{type(e).__name__}: {e} | {traceback.format_exc(limit=2)}"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--max_logs", type=int, default=0, help="시험: 무작위 로그 N개만 (저장하지 않는다)")
    args = ap.parse_args()
    tokens = (FEAT / "tokens.txt").read_text().split()
    labels = pd.read_parquet(EXP / "l1/navtrain_labels.parquet").drop_duplicates("token").set_index("token")
    logs = labels.log_name.reindex(tokens).to_numpy()
    nn = lookalikes(tokens, logs)
    lab = lambda t: labels.loc[t].to_dict()  # noqa: E731
    by_log = {}
    for i, t in enumerate(tokens):
        la = {f"lookalike_{k + 1}": lab(tokens[j]) for k, j in enumerate(nn[i]) if j >= 0}
        by_log.setdefault(logs[i], []).append({"token": t, "human": lab(t), "lookalike": la})
    log_tok = labels.reset_index().groupby("log_name").token.apply(list).to_dict()
    tasks = sorted(((log, rows, {t: labels.loc[t].to_dict() for t in log_tok[log]}) for log, rows in by_log.items()),
                   key=lambda x: -len(x[1]))
    if args.max_logs:
        tasks = [tasks[i] for i in np.random.default_rng(0).choice(len(tasks), args.max_logs, replace=False)]
    print(f"{len(tokens):,} scenes, {len(tasks)} logs", flush=True)
    t0, rows = time.time(), []
    with Pool(args.workers, initializer=l3_run.init_worker, initargs=(L2Config().__dict__,)) as pool:
        for i, r in enumerate(pool.imap_unordered(judge_log, tasks), 1):
            rows.extend(r)
            if i % 20 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} logs, {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    print(df.groupby(df.cf_name.str.split("_").str[0]).status.value_counts().unstack(fill_value=0).to_string())
    if "category" in df:
        print("infeasible category:", df[df.status == "infeasible"].category.value_counts().to_dict())
    if args.max_logs:
        return
    out = EXP / "l3/navtrain_sensor_cf_plausible.parquet"
    df.to_parquet(out, index=False)
    print(f"saved {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
