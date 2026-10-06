"""1단계 확인: NAVSIM 공식 튜토리얼(tutorial/tutorial_visualization.ipynb)을 이 저장소의 데이터로 실행한다.

튜토리얼은 mini split과 카메라 8대·LiDAR를 쓰지만, 이 연구는 test/trainval의 전방 3캠만 받는다.
그래서 같은 API로 BEV(지도 + 물체), 등속 agent 경로, 전방 3캠 그림만 그려 exp/step1/ 에 저장한다.

사용법 (source scripts/setup/env.sh 후):
    python scripts/check_navsim_example.py
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.agents.constant_velocity_agent import ConstantVelocityAgent
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.visualization.plots import plot_bev_frame, plot_bev_with_agent

DATA_ROOT = Path(os.environ["OPENSCENE_DATA_ROOT"])
DEVKIT = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
OUT = Path(os.environ["NAVSIM_EXP_ROOT"]) / "step1"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(DEVKIT / "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml")
    scene_filter: SceneFilter = instantiate(cfg)
    # 1단계에서는 test 카메라 첫 조각만 받았으므로, 이미지가 있는 로그로 제한한다
    scene_filter.log_names = sorted(p.name for p in (DATA_ROOT / "sensor_blobs/test").iterdir())
    current = [scene_filter.num_history_frames - 1]
    sensor_config = SensorConfig(
        cam_f0=current, cam_l0=current, cam_l1=False, cam_l2=False,
        cam_r0=current, cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False,
    )
    loader = SceneLoader(DATA_ROOT / "navsim_logs/test", DATA_ROOT / "sensor_blobs/test", scene_filter,
                         sensor_config=sensor_config)
    token = loader.tokens[0]
    scene = loader.get_scene_from_token(token)
    frame_idx = scene.scene_metadata.num_history_frames - 1
    print(f"scenes {len(loader.tokens)}, token {token}, map {scene.scene_metadata.map_name}")

    fig, ax = plot_bev_frame(scene, frame_idx)
    fig.savefig(OUT / f"{token}_bev.png", dpi=100)

    fig, ax = plot_bev_with_agent(scene, ConstantVelocityAgent())  # 초록: 사람 경로, 빨강: 등속 agent
    fig.savefig(OUT / f"{token}_bev_cv_agent.png", dpi=100)

    cams = scene.frames[frame_idx].cameras
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    for ax, (name, cam) in zip(axes, [("CAM_L0", cams.cam_l0), ("CAM_F0", cams.cam_f0), ("CAM_R0", cams.cam_r0)]):
        ax.imshow(cam.image), ax.set_title(name), ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / f"{token}_cams.png", dpi=80)
    print(f"saved figures to {OUT}")
    print("OK")


if __name__ == "__main__":
    main()
