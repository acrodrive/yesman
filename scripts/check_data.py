"""2단계 확인: 받은 데이터의 개수를 확인하고, 무작위 장면을 불러 보고, 센서를 받은 navtrain 로그 목록을 저장한다.

사용법 (source scripts/setup/env.sh 후):
    python scripts/check_data.py [--num-random 10] [--seed 0]

결과물
  data_lists/navtrain_sensor_logs.txt    센서(전방 3캠)를 받은 navtrain 로그. 학습 장면은 이 로그로 제한한다
  data_lists/navtrain_sensor_tokens.txt  그 로그 안의 navtrain 장면 중 현재 프레임 3캠이 모두 있는 장면
"""

import argparse
import os
import pickle
import random
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader

DATA_ROOT = Path(os.environ["OPENSCENE_DATA_ROOT"])
FILTER_DIR = Path(os.environ["NAVSIM_DEVKIT_ROOT"]) / "navsim/planning/script/config/common/train_test_split/scene_filter"
OUT_DIR = Path(os.environ["YESMAN_ROOT"]) / "data_lists"
CAMS = ("CAM_L0", "CAM_F0", "CAM_R0")


def load_filter(name: str) -> SceneFilter:
    return instantiate(OmegaConf.load(FILTER_DIR / f"{name}.yaml"))


def scan(split: str, scene_filter: SceneFilter, log_names):
    """로그를 읽어 split 장면 중 현재 프레임 3캠 이미지가 모두 있는 장면과 없는 장면을 센다."""
    tokens = set(scene_filter.tokens)
    ok, missing = [], []
    for log in sorted(log_names):
        for frame in pickle.load(open(DATA_ROOT / f"navsim_logs/{split}/{log}.pkl", "rb")):
            if frame["token"] not in tokens:
                continue
            paths = [DATA_ROOT / f"sensor_blobs/{split}" / frame["cams"][c]["data_path"] for c in CAMS]
            (ok if all(p.exists() for p in paths) else missing).append(frame["token"])
    return ok, missing


def check_random_scenes(split: str, base: SceneFilter, tokens, n: int, rng: random.Random):
    """무작위 장면을 SceneLoader로 불러 카메라 이미지, 차량 상태, 미래 경로가 모두 읽히는지 본다."""
    sample = rng.sample(sorted(tokens), min(n, len(tokens)))
    current = [base.num_history_frames - 1]
    scene_filter = SceneFilter(
        num_history_frames=base.num_history_frames, num_future_frames=base.num_future_frames,
        frame_interval=base.frame_interval, has_route=base.has_route, tokens=sample,
    )
    loader = SceneLoader(
        DATA_ROOT / f"navsim_logs/{split}", DATA_ROOT / f"sensor_blobs/{split}", scene_filter,
        sensor_config=SensorConfig(cam_f0=current, cam_l0=current, cam_l1=False, cam_l2=False, cam_r0=current,
                                   cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False),
    )
    assert len(loader.tokens) == len(sample), (len(loader.tokens), len(sample))
    for token in loader.tokens:
        scene = loader.get_scene_from_token(token)
        frame = scene.frames[base.num_history_frames - 1]
        for name in ("cam_l0", "cam_f0", "cam_r0"):
            img = getattr(frame.cameras, name).image
            assert img is not None and img.shape == (1080, 1920, 3), (token, name, None if img is None else img.shape)
        status = frame.ego_status
        assert np.isfinite(status.ego_velocity).all() and np.isfinite(status.ego_acceleration).all()
        future = scene.get_future_trajectory(num_trajectory_frames=8).poses
        assert future.shape == (8, 3) and np.isfinite(future).all(), (token, future.shape)
    print(f"[{split}] random {len(sample)} scenes: cameras (1080x1920x3 x3), ego status, future (8x3) all readable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-random", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    OUT_DIR.mkdir(exist_ok=True)

    # navtest: 모든 장면의 현재 프레임 3캠이 있어야 한다
    navtest = load_filter("navtest")
    ok, missing = scan("test", navtest, navtest.log_names)
    print(f"[test] navtest scenes {len(navtest.tokens)}: with 3 cams {len(ok)}, missing {len(missing)}")
    check_random_scenes("test", navtest, ok, args.num_random, rng)

    # navtrain: 받은 조각의 로그만 센서가 있다
    navtrain = load_filter("navtrain")
    done = sorted((DATA_ROOT / "_done").glob("navtrain_cur_*"))
    sensor_logs = sorted({l for f in done for l in f.read_text().split()})
    all_logs = {p.stem for p in (DATA_ROOT / "navsim_logs/trainval").glob("*.pkl")}
    print(f"[trainval] shards {[f.name.split('_')[-1] for f in done]}, logs with sensors {len(sensor_logs)}")
    print(f"[trainval] navtrain logs {len(navtrain.log_names)}, present in trainval logs "
          f"{len(set(navtrain.log_names) & all_logs)}")
    assert set(sensor_logs) <= set(navtrain.log_names), "navtrain이 아닌 로그가 섞였다"
    ok, missing = scan("trainval", navtrain, sensor_logs)
    print(f"[trainval] navtrain scenes {len(navtrain.tokens)}: in sensor logs {len(ok) + len(missing)} "
          f"({(len(ok) + len(missing)) / len(navtrain.tokens):.1%}), with 3 cams {len(ok)}, missing {len(missing)}")
    check_random_scenes("trainval", navtrain, ok, args.num_random, rng)

    (OUT_DIR / "navtrain_sensor_logs.txt").write_text("\n".join(sensor_logs) + "\n")
    (OUT_DIR / "navtrain_sensor_tokens.txt").write_text("\n".join(sorted(ok)) + "\n")
    print(f"saved {OUT_DIR}/navtrain_sensor_logs.txt, navtrain_sensor_tokens.txt")
    print("OK")


if __name__ == "__main__":
    main()
