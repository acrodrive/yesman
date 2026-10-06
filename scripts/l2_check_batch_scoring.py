"""5단계: 여러 경로를 한 번에 채점하는 BatchScorer가 공식 pdm_score()와 같은 값을 내는지 확인한다.

navtest 장면마다 경로 6개(사람, 등속, 다른 장면의 사람 경로 4개)를 BatchScorer로 한 번에 채점하고,
같은 경로를 하나씩 공식 pdm_score()로 채점하여 항목별로 비교한다. EP는 함께 채점한 경로에 따라 정규화가 바뀌므로 비교하지 않는다.

사용법: python scripts/l2_check_batch_scoring.py --n 50
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from navsim.agents.constant_velocity_agent import ConstantVelocityAgent
from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score
from yesman.data import scene_loader
from yesman.scoring import METRICS, BatchScorer

EXP = Path(os.environ["NAVSIM_EXP_ROOT"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    bs = BatchScorer()
    mcl = MetricCacheLoader(EXP / "metric_cache/navtest")
    feats = pd.read_parquet(EXP / "l1/navtest_features.parquet", columns=["token", "log_name", "valid", "poses"])
    feats = feats[feats.valid == 1]
    pick = feats.sample(n=args.n, random_state=0)
    loader = scene_loader("navtest", pick.log_name, pick.token)
    other = np.stack(feats.poses.sample(n=4 * args.n, random_state=1).to_numpy()).reshape(-1, 8, 3)

    diffs, t_batch, t_single = [], 0.0, 0.0
    for i, t in enumerate(pick.token):
        mc = mcl.get_from_token(t)
        scene = loader.get_scene_from_token(t)
        paths = [scene.get_future_trajectory(8).poses,
                 ConstantVelocityAgent().compute_trajectory(scene.get_agent_input()).poses] + list(other[4 * i:4 * i + 4])
        t0 = time.time()
        batch = bs.score(mc, paths)
        t_batch += time.time() - t0
        for j, p in enumerate(paths):
            t0 = time.time()
            res, _ = pdm_score(mc, Trajectory(np.asarray(p, dtype=np.float32)), bs.sampling, bs.simulator,
                               bs.scorer, bs.traffic)
            t_single += time.time() - t0
            diffs.append({m: abs(float(res[m].iloc[0]) - batch[m].iloc[j]) for m in METRICS if m != "ego_progress"})
    d = pd.DataFrame(diffs)
    print(f"scenes {args.n}, paths {len(d)}")
    print("max |batch - official| per metric:", d.max().to_dict())
    print(f"time: batch {t_batch / args.n:.2f} s/scene (6 paths), official one-by-one {t_single / args.n:.2f} s/scene")


if __name__ == "__main__":
    main()
