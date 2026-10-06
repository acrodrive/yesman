"""6단계: 장면 목록(토큰 파일)에 있는 장면만 metric cache를 만든다 (공식 run_metric_caching.py와 같은 설정).

공식 스크립트는 split(scene filter) 단위로 돈다. 학습 장면은 navtrain 중 센서를 받은 장면(data_lists/navtrain_sensor_tokens.txt)만
쓰므로, 공식 설정을 hydra compose로 읽은 뒤 scene filter의 토큰과 로그를 그 목록으로 바꿔서 같은 cache_data()를 부른다.

사용법: python scripts/run_metric_caching_tokens.py --split navtrain --tokens data_lists/navtrain_sensor_tokens.txt \
            --out exp/metric_cache/navtrain_sensor --workers 16
"""

import argparse
import logging
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import open_dict

from navsim.planning.metric_caching.caching import cache_data
from navsim.planning.script.builders.worker_pool_builder import build_worker

DEVKIT = Path(os.environ["NAVSIM_DEVKIT_ROOT"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="navtrain")
    ap.add_argument("--tokens", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    tokens = [t.strip() for t in args.tokens.read_text().split() if t.strip()]
    with initialize_config_dir(config_dir=str(DEVKIT / "navsim/planning/script/config/metric_caching"),
                               version_base=None):
        cfg = compose(config_name="default_metric_caching", overrides=[
            f"train_test_split={args.split}", f"metric_cache_path={args.out.resolve()}",
            f"worker.threads_per_node={args.workers}"])
    with open_dict(cfg):
        # 토큰만 바꾸면 SceneLoader가 모든 로그를 읽으므로 로그도 줄인다: 토큰 → 로그 대응은 L1 특징 표에 있다
        import pandas as pd
        f = pd.read_parquet(Path(os.environ["NAVSIM_EXP_ROOT"]) / f"l1/{args.split}_features.parquet",
                            columns=["token", "log_name"])
        f = f[f.token.isin(set(tokens))]
        cfg.train_test_split.scene_filter.tokens = sorted(f.token)
        cfg.train_test_split.scene_filter.log_names = sorted(f.log_name.unique())
    print(f"{len(f)} tokens in {f.log_name.nunique()} logs -> {args.out}", flush=True)
    cache_data(cfg=cfg, worker=build_worker(cfg))


if __name__ == "__main__":
    main()
