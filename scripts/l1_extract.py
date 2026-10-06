"""4단계: split의 모든 장면에서 사람 경로의 L1 특징을 계산하여 표(parquet)로 저장한다.

사람 경로(8x3, ego 좌표계)도 poses 열에 함께 저장한다(5단계 경로 모음용). 특징만 저장하고 결정은 저장하지 않는다. 결정은 기준값(yesman/l1_thresholds.yaml)을 정한 뒤 classify()로 다시 계산한다.
그래야 기준값을 바꿀 때마다 지도를 다시 읽지 않아도 된다.

사용법 (source scripts/setup/env.sh 후):
    python scripts/l1_extract.py --split navtrain --workers 16      # → exp/l1/navtrain_features.parquet
    python scripts/l1_extract.py --split navtest --workers 16
"""

import argparse
import os
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from yesman.data import DATA_ROOT, SPLITS, scene_filter, scene_loader
from yesman.l1 import features_from_scene

OUT = Path(os.environ["NAVSIM_EXP_ROOT"]) / "l1"


def run_log(args):
    split, log = args
    rows = []
    try:
        loader = scene_loader(split, [log])
    except Exception:
        return [{"log_name": log, "valid": 0, "reason": "load_error: " + traceback.format_exc(limit=1)}]
    for token in loader.tokens:
        try:
            scene = loader.get_scene_from_token(token)
            f = features_from_scene(scene)
            f["map_name"] = scene.scene_metadata.map_name
            f["poses"] = scene.get_future_trajectory(8).poses.astype("float32").ravel().tolist()  # 5단계 경로 모음용 (8x3)
        except Exception as e:  # 한 장면의 오류로 전체가 멈추지 않게 한다
            f = {"valid": 0, "reason": f"error: {type(e).__name__}: {e}"}
        f.update(token=token, log_name=log, split=split)
        rows.append(f)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="navtrain", choices=list(SPLITS))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max_logs", type=int, default=None, help="시험용: 앞의 로그 몇 개만")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or OUT / f"{args.split}_features.parquet"

    logs = scene_filter(args.split).log_names
    if not logs:  # log_names가 없는 filter는 폴더의 모든 로그
        logs = sorted(p.stem for p in (DATA_ROOT / "navsim_logs" / SPLITS[args.split]).glob("*.pkl"))
    logs = logs[: args.max_logs] if args.max_logs else logs
    print(f"{args.split}: {len(logs)} logs, {args.workers} workers", flush=True)

    t0, rows = time.time(), []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(run_log, [(args.split, l) for l in logs]), 1):
            rows.extend(r)
            if i % 50 == 0 or i == len(logs):
                print(f"{i}/{len(logs)} logs, {len(rows)} scenes, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    n_valid = int(df.valid.sum())
    print(f"saved {out}: {len(df)} scenes, valid {n_valid} ({n_valid / len(df) * 100:.1f}%), {time.time() - t0:.0f}s")
    print(df[df.valid == 0].reason.str.split(":").str[0].value_counts().to_string())


if __name__ == "__main__":
    main()
