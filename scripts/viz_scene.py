"""3단계: 장면 하나를 전방 3캠과 BEV로 그려 좌표계(원점, 축 방향, 각도 부호)를 확인한다.

그림 한 장 = 위: CAM_L0, CAM_F0, CAM_R0 (사람 미래 경로와 가까운 차량 번호를 영상에 투영)
             아래: ego 좌표계 BEV (공식 navsim.visualization의 지도·박스 + 축 눈금, heading 화살표, 차량 번호)
BEV와 영상에서 같은 번호의 차량이 같은 쪽에 있고, 투영한 경로가 도로를 따라가면 좌표계를 맞게 이해한 것이다.

좌표계 (NAVSIM, ego 좌표계)
- 원점: 현재 프레임(history 4프레임 중 마지막)의 ego 뒤차축. 로그의 lidar2ego가 단위 행렬이라 lidar 좌표계 = ego 좌표계이다.
- 축: x 앞, y 왼쪽, z 위 (오른손 좌표계).
- 각도: heading은 x축에서 y축 쪽으로(위에서 보아 반시계) 잰 값이다. 좌회전하면 커진다.

수치 확인 (--n_check 장면에서)
- 현재 프레임을 ego 좌표계로 바꾸면 (0, 0, 0)이다.
- 0.5초 간격 미래 경로에서 움직인 방향 atan2(dy, dx)가 그 구간의 heading 평균과 맞다 (heading 부호 확인).
- 첫 0.5초 동안 움직인 거리 dx/0.5가 현재 속도 vx와 맞다 (속도 축 확인).

사용법 (source scripts/setup/env.sh 후):
    python scripts/viz_scene.py                    # navtest에서 좌회전, 우회전, 직진 장면을 하나씩 골라 그린다
    python scripts/viz_scene.py --tokens <token> ...
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from hydra.utils import instantiate
from nuplan.common.actor_state.state_representation import StateSE2
from omegaconf import OmegaConf

from navsim.agents.constant_velocity_agent import ConstantVelocityAgent
from navsim.common.dataclasses import Scene, SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.common.enums import BoundingBoxIndex
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from navsim.visualization.bev import add_annotations_to_bev_ax, add_map_to_bev_ax

DATA_ROOT = Path(os.environ["OPENSCENE_DATA_ROOT"])
DEVKIT = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
OUT = Path(os.environ["NAVSIM_EXP_ROOT"]) / "step03" / "viz"
CAMS = ["cam_l0", "cam_f0", "cam_r0"]
BEV_RANGE = 40.0  # BEV에 그리는 범위 (앞뒤, 좌우 ±m)


def load_navtest_loader() -> SceneLoader:
    """navtest 장면 로더. 현재 프레임의 전방 3캠만 읽는다."""
    cfg = OmegaConf.load(DEVKIT / "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml")
    scene_filter: SceneFilter = instantiate(cfg)
    cur = [scene_filter.num_history_frames - 1]
    sensor_config = SensorConfig(cam_f0=cur, cam_l0=cur, cam_l1=False, cam_l2=False, cam_r0=cur,
                                 cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False)
    return SceneLoader(DATA_ROOT / "navsim_logs/test", DATA_ROOT / "sensor_blobs/test", scene_filter,
                       sensor_config=sensor_config)


def scene_without_sensors(loader: SceneLoader, token: str) -> Scene:
    """이미지를 읽지 않고 장면(로그 정보, 지도)만 만든다."""
    f = loader._scene_filter
    return Scene.from_scene_dict_list(loader.scene_frames_dicts[token], None, f.num_history_frames,
                                      f.num_future_frames, SensorConfig.build_no_sensors())


def check_frames(scene: Scene) -> dict:
    """원점, heading 부호, 속도 축을 수치로 확인한다."""
    cur = scene.scene_metadata.num_history_frames - 1
    poses = np.array([f.ego_status.ego_pose for f in scene.frames], dtype=np.float64)  # 전역 좌표 (x, y, heading)
    local = convert_absolute_to_relative_se2_array(StateSE2(*poses[cur]), poses)
    fut = local[cur:cur + 9]  # 현재 + 4초 (0.5초 간격)
    d = np.diff(fut[:, :2], axis=0)
    step = np.linalg.norm(d, axis=1)
    moving = step > 1.0  # 0.5초에 1m 이상 움직인 구간만 (2 m/s 이상)
    motion_dir = np.arctan2(d[:, 1], d[:, 0])
    mid_heading = (fut[:-1, 2] + fut[1:, 2]) / 2
    err = np.abs(np.angle(np.exp(1j * (motion_dir - mid_heading))))
    vx = scene.frames[cur].ego_status.ego_velocity[0]  # 현재 프레임의 속도는 ego 좌표계 값이다
    return dict(
        origin_err=float(np.abs(local[cur]).max()),
        heading_err_deg=np.degrees(err[moving]),
        vx=float(vx),
        vx_from_pose=float(fut[1, 0] / 0.5),
    )


def ground_z(boxes: np.ndarray) -> float:
    """가까운 물체 박스 바닥 높이의 중앙값을 지면 높이로 쓴다 (ego 원점은 뒤차축이라 지면보다 약간 위다)."""
    near = np.linalg.norm(boxes[:, :2], axis=1) < 30
    if near.sum() == 0:
        return 0.0
    return float(np.median(boxes[near, BoundingBoxIndex.Z] - boxes[near, BoundingBoxIndex.HEIGHT] / 2))


def project(points_ego: np.ndarray, cam) -> tuple:
    """ego(=lidar) 좌표 점들을 영상 픽셀로 투영한다. NAVSIM camera.py와 같은 핀홀 모델 (왜곡 보정 없음)."""
    r, t = cam.sensor2lidar_rotation, cam.sensor2lidar_translation  # p_lidar = R p_cam + t
    p_cam = (points_ego - t) @ r  # = R^T (p - t)
    uvw = p_cam @ cam.intrinsics.T
    z = uvw[:, 2]
    uv = uvw[:, :2] / np.maximum(z[:, None], 1e-3)
    h, w = cam.image.shape[:2]
    ok = (z > 0.5) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv, ok


def nearest_vehicles(annotations, k: int = 5) -> list:
    idx = [i for i, n in enumerate(annotations.names) if n == "vehicle"]
    boxes = annotations.boxes
    idx = [i for i in idx if abs(boxes[i, 0]) < BEV_RANGE and abs(boxes[i, 1]) < BEV_RANGE]
    idx.sort(key=lambda i: np.linalg.norm(boxes[i, :2]))
    return idx[:k]


def draw_scene(scene: Scene, tag: str, out_dir: Path) -> Path:
    cur = scene.scene_metadata.num_history_frames - 1
    frame = scene.frames[cur]
    human = scene.get_future_trajectory(8).poses  # (8, 3) 0.5초 간격 4초
    hist = scene.get_history_trajectory().poses  # (4, 3) 마지막이 현재 = (0, 0, 0)
    cv = ConstantVelocityAgent().compute_trajectory(scene.get_agent_input()).poses
    ann = frame.annotations
    vehicles = nearest_vehicles(ann)
    gz = ground_z(ann.boxes)
    chk = check_frames(scene)

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 2.1])

    # 위: 카메라 3장에 사람 미래 경로(초록), 등속 경로(빨강), 차량 번호를 투영
    traj_pts = np.concatenate([[[0, 0]], human[:, :2]])
    traj_dense = np.concatenate([np.linspace(traj_pts[i], traj_pts[i + 1], 10) for i in range(len(traj_pts) - 1)])
    cv_dense = np.linspace([0, 0], cv[-1, :2], 80)
    for j, name in enumerate(CAMS):
        cam = getattr(frame.cameras, name)
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(cam.image)
        for pts, color in [(traj_dense, "lime"), (cv_dense, "red")]:
            p3 = np.c_[pts, np.full(len(pts), gz)]
            uv, ok = project(p3, cam)
            ax.scatter(uv[ok, 0], uv[ok, 1], s=4, c=color)
        for n, i in enumerate(vehicles, 1):
            uv, ok = project(ann.boxes[i:i + 1, :3], cam)
            if ok[0]:
                ax.text(uv[0, 0], uv[0, 1], str(n), color="yellow", fontsize=16, weight="bold",
                        ha="center", va="center", bbox=dict(fc="black", alpha=0.6, pad=1))
        ax.set_title(name.upper())
        ax.axis("off")

    # 아래 가운데: BEV (공식 함수는 가로축 = y, 세로축 = x로 그린다. 가로축을 뒤집어 왼쪽(+y)이 화면 왼쪽)
    ax = fig.add_subplot(gs[1, :])
    add_map_to_bev_ax(ax, scene.map_api, StateSE2(*frame.ego_status.ego_pose))
    add_annotations_to_bev_ax(ax, ann)
    ax.plot(hist[:, 1], hist[:, 0], "o-", color="grey", ms=4, label="history (past 1.5 s)", zorder=10)
    ax.plot(np.r_[0, human[:, 1]], np.r_[0, human[:, 0]], "o-", color="green", ms=4, label="human future (4 s)",
            zorder=10)
    ax.plot(np.r_[0, cv[:, 1]], np.r_[0, cv[:, 0]], "o-", color="red", ms=3, label="constant velocity", zorder=10)
    # heading 화살표: 방향 (cos h, sin h)를 (x, y)로 두고, 그림에서는 (dy, dx)로 그린다
    for x, y, h in np.r_[[[0, 0, 0]], human]:
        ax.annotate("", xy=(y + 2.5 * np.sin(h), x + 2.5 * np.cos(h)), xytext=(y, x),
                    arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.5), zorder=11)
    for n, i in enumerate(vehicles, 1):
        bx, by = ann.boxes[i, 0], ann.boxes[i, 1]
        ax.text(by, bx, str(n), color="yellow", fontsize=13, weight="bold", ha="center", va="center",
                bbox=dict(fc="black", alpha=0.6, pad=1), zorder=12, clip_on=True)
    ax.set_aspect("equal")
    ax.set_xlim(BEV_RANGE, -BEV_RANGE)  # 가로축 = y, 뒤집어서 +y(왼쪽)가 화면 왼쪽
    ax.set_ylim(-BEV_RANGE / 2, BEV_RANGE * 1.5)
    ax.set_xlabel("y [m]  (+ = left)")
    ax.set_ylabel("x [m]  (+ = forward)")
    ax.axhline(0, color="k", lw=0.5, zorder=0)
    ax.axvline(0, color="k", lw=0.5, zorder=0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fx, fy, fh = human[-1]
    txt = (f"origin = rear axle of current frame (local pose of current frame max |err| = {chk['origin_err']:.1e})\n"
           f"human end @4s: x = {fx:.1f} m, y = {fy:+.1f} m, heading = {np.degrees(fh):+.1f} deg\n"
           f"motion dir vs heading: max |err| = "
           f"{(chk['heading_err_deg'].max() if len(chk['heading_err_deg']) else float('nan')):.1f} deg\n"
           f"vx (ego status) = {chk['vx']:.2f} m/s, dx/0.5s = {chk['vx_from_pose']:.2f} m/s")
    ax.text(0.01, 0.99, txt, transform=ax.transAxes, va="top", ha="left", fontsize=10, family="monospace",
            bbox=dict(fc="white", alpha=0.85))
    fig.suptitle(f"[{tag}] token {scene.scene_metadata.initial_token}  log {scene.scene_metadata.log_name}  "
                 f"map {scene.scene_metadata.map_name}  ground z = {gz:+.2f} m")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}_{scene.scene_metadata.initial_token}.png"
    fig.savefig(path, dpi=70)
    plt.close(fig)
    return path


def pick_tokens(loader: SceneLoader, n_check: int, seed: int = 0) -> dict:
    """무작위로 장면을 보면서 좌회전, 우회전, 직진 장면을 하나씩 고르고, 좌표계 수치 확인을 모은다."""
    rng = np.random.default_rng(seed)
    tokens = list(rng.permutation(loader.tokens))
    picked, errs, origin, vx = {}, [], [], []
    for t in tokens[:n_check]:
        sc = scene_without_sensors(loader, t)
        chk = check_frames(sc)
        errs.append(chk["heading_err_deg"])
        origin.append(chk["origin_err"])
        if abs(chk["vx"]) > 2:
            vx.append((chk["vx"], chk["vx_from_pose"]))
        x, y, h = sc.get_future_trajectory(8).poses[-1]
        if "left_turn" not in picked and np.degrees(h) > 60 and y > 5:
            picked["left_turn"] = t
        elif "right_turn" not in picked and np.degrees(h) < -60 and y < -5:
            picked["right_turn"] = t
        elif "straight" not in picked and abs(np.degrees(h)) < 3 and abs(y) < 0.5 and x > 30 and len(
                nearest_vehicles(sc.frames[sc.scene_metadata.num_history_frames - 1].annotations)) >= 3:
            picked["straight"] = t
    errs = np.concatenate(errs)
    vx = np.array(vx)
    print(f"[check] scenes {n_check}: local pose of current frame max |err| = {max(origin):.2e}")
    print(f"[check] motion direction vs heading (moving 0.5 s steps {len(errs)}): "
          f"median {np.median(errs):.2f} deg, p99 {np.percentile(errs, 99):.2f} deg, max {errs.max():.2f} deg")
    print(f"[check] vx (ego status) vs dx/0.5s (scenes with |vx| > 2 m/s: {len(vx)}): "
          f"median |diff| {np.median(np.abs(vx[:, 0] - vx[:, 1])):.2f} m/s, "
          f"sign agreement {np.mean(np.sign(vx[:, 0]) == np.sign(vx[:, 1])) * 100:.1f}%")
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", nargs="*", default=None)
    ap.add_argument("--n_check", type=int, default=1000, help="좌표계 수치 확인과 장면 고르기에 쓸 장면 수")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    loader = load_navtest_loader()
    print(f"navtest scenes: {len(loader.tokens)}")
    if args.tokens:
        picked = {f"scene{i}": t for i, t in enumerate(args.tokens)}
    else:
        picked = pick_tokens(loader, args.n_check)
    for tag, t in picked.items():
        path = draw_scene(loader.get_scene_from_token(t), tag, args.out)
        print(f"{tag}: {t} -> {path}")


if __name__ == "__main__":
    main()
