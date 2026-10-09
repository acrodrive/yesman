"""12단계 B (진단): 장면마다 현재 시점의 정답 주변 차량을 LTF 특징과 같은 순서로 저장한다.

뽑는 기준은 LTF가 물체 검출을 학습할 때의 정답과 같다(transfuser_features._compute_agent_targets):
ego 좌표계 |x|, |y| <= 32 m 안의 vehicle, 가까운 순 최대 30개. 여기에 정답 속도(velocity_3d의 x, y)를 더한다.
열: x, y, heading, length, width, vx, vy, valid (valid = 1이면 실제 물체, 0이면 빈 자리)

사용법: python scripts/gt_objects.py --split navtrain   (navtest도)
출력: exp/features/ltf/<split>/obj_gt.npy (N, 30, 8) float32, 행 순서 = tokens.txt
"""

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from yesman.data import scene_loader, scene_without_sensors

ROOT = Path(os.environ["YESMAN_ROOT"])
FEAT = ROOT / "exp/features/ltf"
MAX_OBJ, RANGE = 30, 32.0


def objects(annotations) -> np.ndarray:
    out = np.zeros((MAX_OBJ, 8), np.float32)
    b, v = annotations.boxes, annotations.velocity_3d
    keep = [i for i, n in enumerate(annotations.names)
            if n == "vehicle" and abs(b[i][0]) <= RANGE and abs(b[i][1]) <= RANGE]
    if keep:
        keep = np.array(keep)[np.argsort(np.linalg.norm(b[keep, :2], axis=1))[:MAX_OBJ]]
        n = len(keep)
        out[:n, :5] = b[keep][:, [0, 1, 6, 3, 4]]  # BoundingBoxIndex: x y z l w h heading
        out[:n, 5:7] = v[keep, :2]
        out[:n, 7] = 1
    return out


def run_log(args):
    split, log, tokens = args
    loader = scene_loader(split, [log], tokens)
    res = {}
    for t in tokens:
        sc = scene_without_sensors(loader, t)
        res[t] = objects(sc.frames[sc.scene_metadata.num_history_frames - 1].annotations)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["navtrain", "navtest"], required=True)
    ap.add_argument("--workers", type=int, default=28)
    args = ap.parse_args()
    tokens = (FEAT / args.split / "tokens.txt").read_text().split()
    logs = pd.read_parquet(ROOT / f"exp/l1/{args.split}_features.parquet", columns=["token", "log_name"])
    logs = logs[logs.token.isin(set(tokens))].drop_duplicates("token")
    assert len(logs) == len(tokens), f"log를 모르는 토큰 {len(tokens) - len(logs)}개"
    tasks = [(args.split, log, sorted(g.token)) for log, g in logs.groupby("log_name")]
    res = {}
    with Pool(args.workers) as pool:
        for k, r in enumerate(pool.imap_unordered(run_log, tasks), 1):
            res.update(r)
            if k % 100 == 0 or k == len(tasks):
                print(f"{k}/{len(tasks)} logs", flush=True)
    arr = np.stack([res[t] for t in tokens])
    np.save(FEAT / args.split / "obj_gt.npy", arr)
    print(f"saved {arr.shape}, 장면당 차량 평균 {arr[..., 7].sum(1).mean():.1f}개")


if __name__ == "__main__":
    main()
