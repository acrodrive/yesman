"""L1 단위 테스트 (4단계 끝났다는 기준: 직접 만든 경로에서 기대한 결정이 나온다).

실제 navtest 장면의 지도 위에 경로를 직접 만든다. 경로는 그 장면의 차선 중심선을 따라 진행 거리 s(t)와
옆 거리 d(t)(왼쪽 +)로 정의하고 ego 좌표계로 바꾼다. 기준값은 yesman/l1_thresholds.yaml을 쓴다.

실행 (source scripts/setup/env.sh 후): pytest -q tests/test_l1.py
"""

import numpy as np
import pytest

from yesman.data import scene_loader
from yesman.decision import Decision, Segment, same_decision
from yesman.l1 import boundary_margin, classify, extract_features, features_from_scene, lane_index, load_thresholds

# 직선 다차선 도로 (Las Vegas strip). 3단계 시각화의 직진 장면이다.
STRAIGHT = ("2021.06.28.16.57.59_veh-26_00016_00484", "d32336b185505124")
# 교차로 좌회전 연결 차선 위의 장면 (navtest에서 사람 경로가 연결 차선을 따라 좌회전한 장면)
LEFT_TURN = ("2021.06.03.18.47.39_veh-35_00503_00777", "39fd7cb73aa259db")
# 연결 차선이 없는 굽은 도로 (4초 동안 차선 방향이 약 36도 바뀐다)
CURVED = ("2021.06.28.13.53.26_veh-26_00492_00696", "0b4c7130090c5e85")
T = np.arange(1, 9) * 0.5  # 경로 시각 0.5~4.0초


def _load(log, token):
    return scene_loader("navtest", [log], [token]).get_scene_from_token(token)


@pytest.fixture(scope="module")
def straight():
    scene = _load(*STRAIGHT)
    dbg = {}
    features_from_scene(scene, debug=dbg)
    return scene, dbg


def make_traj(scene, dbg, s_t, d_t):
    """기준 차선열 중심선을 따라 (s(t), d(t))인 경로를 ego 좌표계 (8, 3)로 만든다. s는 출발점의 투영 위치부터 잰다."""
    xy, cs = dbg["chain_xy"], dbg["chain_s"]
    s_abs = dbg["s"][0] + np.asarray(s_t)
    px, py = np.interp(s_abs, cs, xy[:, 0]), np.interp(s_abs, cs, xy[:, 1])
    ds = 0.5  # 중심선 방향을 앞뒤 0.5 m 차이로 구한다
    hx = np.interp(s_abs + ds, cs, xy[:, 0]) - np.interp(s_abs - ds, cs, xy[:, 0])
    hy = np.interp(s_abs + ds, cs, xy[:, 1]) - np.interp(s_abs - ds, cs, xy[:, 1])
    tang = np.arctan2(hy, hx)
    d_t = np.asarray(d_t, dtype=float)
    gx, gy = px - np.sin(tang) * d_t, py + np.cos(tang) * d_t
    # heading: 실제로 움직인 방향 (옆으로 움직이면 중심선 방향에서 기운다)
    d_all = np.r_[dbg["d"][0], d_t]
    s_all = np.r_[0.0, np.asarray(s_t)]
    slope = np.gradient(d_all, s_all)[1:] if np.all(np.diff(s_all) > 1e-6) else np.zeros(8)
    gh = tang + np.arctan(slope)
    x0, y0, h0 = scene.frames[scene.scene_metadata.num_history_frames - 1].ego_status.ego_pose
    c, s = np.cos(h0), np.sin(h0)
    dx, dy = gx - x0, gy - y0
    return np.stack([c * dx + s * dy, -s * dx + c * dy, (gh - h0 + np.pi) % (2 * np.pi) - np.pi], axis=1)


def run_l1(scene, poses, v0):
    status = scene.frames[scene.scene_metadata.num_history_frames - 1].ego_status
    return classify(extract_features(poses, v0, status.ego_pose, scene.map_api))


def lat(dec):
    return [s.lat for s in dec.segments]


def lon(dec):
    return [s.lon for s in dec.segments]


def smooth_step(t, t0, t1):
    """t0~t1 동안 0에서 1로 부드럽게 바뀌는 함수 (cubic)."""
    x = np.clip((t - t0) / (t1 - t0), 0, 1)
    return 3 * x ** 2 - 2 * x ** 3


# ---------------------------------------------------------------------------------------------------------------

def test_straight_constant_speed(straight):
    scene, dbg = straight
    v = 10.0
    dec = run_l1(scene, make_traj(scene, dbg, v * T, np.full(8, dbg["d"][0])), v0=v)
    assert lon(dec) == ["go", "go"] and lat(dec) == ["keep_lane", "keep_lane"], dec
    th = load_thresholds()
    assert dec.segments[0].lon_strength == pytest.approx(min(1, v / th["v_max"]), abs=0.02)
    assert not dec.ambiguous, dec


def test_stop(straight):
    """10 m/s에서 2초 동안 5 m/s^2로 멈추고 그대로 선다."""
    scene, dbg = straight
    v0, a = 10.0, 5.0
    tt = np.minimum(T, v0 / a)
    s = v0 * tt - 0.5 * a * tt ** 2
    dec = run_l1(scene, make_traj(scene, dbg, s, np.full(8, dbg["d"][0])), v0=v0)
    assert lon(dec) == ["stop", "stop"], dec
    th = load_thresholds()
    assert dec.segments[0].lon_strength == pytest.approx(min(1, (v0 / 2.0) / th["a_max"]), abs=0.05)
    assert dec.segments[1].lon_strength == pytest.approx(0.0, abs=0.01)  # 이미 서 있음
    assert lat(dec) == ["keep_lane", "keep_lane"], dec


def test_offset_left(straight):
    """구간 1에서 같은 차선 안에서 왼쪽으로 0.8 m 옮긴다."""
    scene, dbg = straight
    v, d0 = 10.0, dbg["d"][0]
    d = d0 + 0.8 * smooth_step(T, 0.0, 2.0)
    dec = run_l1(scene, make_traj(scene, dbg, v * T, d), v0=v)
    assert lat(dec) == ["offset_left", "keep_lane"], dec


def test_offset_right(straight):
    scene, dbg = straight
    v, d0 = 10.0, dbg["d"][0]
    d = d0 - 0.8 * smooth_step(T, 2.0, 4.0)
    dec = run_l1(scene, make_traj(scene, dbg, v * T, d), v0=v)
    assert lat(dec) == ["keep_lane", "offset_right"], dec


@pytest.mark.parametrize("side,sign", [("left", 1), ("right", -1)])
def test_lane_change(straight, side, sign):
    """구간 1에서 한 차선 폭만큼 옆 차선으로 옮기고, 구간 2에서는 새 차선을 따라간다."""
    scene, dbg = straight
    v = 10.0
    w = dbg["wl"][0] + dbg["wr"][0]
    d = dbg["d"][0] + sign * w * smooth_step(T, 0.0, 2.0)
    dec = run_l1(scene, make_traj(scene, dbg, v * T, d), v0=v)
    assert lat(dec) == [f"lane_change_{side}", "keep_lane"], dec
    assert dec.segments[0].lat_strength > 0


def test_left_turn():
    """교차로에서 좌회전 연결 차선을 따라 6 m/s로 달린다."""
    if LEFT_TURN[1] == "PLACEHOLDER":
        pytest.skip("좌회전 장면 미정")
    scene = _load(*LEFT_TURN)
    dbg = {}
    features_from_scene(scene, debug=dbg)  # 사람 경로가 고른 기준 차선열 = 좌회전 연결 차선을 지나는 차선열
    v = 6.0
    dec = run_l1(scene, make_traj(scene, dbg, v * T, np.full(8, dbg["d"][0])), v0=v)
    assert "turn_left" in lat(dec) and "turn_right" not in lat(dec), dec
    assert all(l in ("turn_left", "keep_lane") for l in lat(dec)), dec


@pytest.mark.parametrize("offset", [0.0, 0.4, -0.4])
def test_curved_road_keep_lane(offset):
    """굽은 도로에서 차선 중심선(또는 중심선에서 0.4 m 옆)을 따라 달리면 keep lane이다."""
    scene = _load(*CURVED)
    dbg = {}
    features_from_scene(scene, debug=dbg)
    v = 7.0
    dec = run_l1(scene, make_traj(scene, dbg, v * T, np.full(8, offset)), v0=v)
    assert lat(dec) == ["keep_lane", "keep_lane"], dec
    h = make_traj(scene, dbg, v * T, np.full(8, offset))[-1, 2]
    assert abs(np.degrees(h)) > 20  # 실제로 굽은 길이다


def test_right_turn_is_not_left():
    """좌회전 장면의 경로를 좌우로 뒤집으면(y, heading 부호 반전) 좌회전으로 판정되지 않는다."""
    scene = _load(*LEFT_TURN)
    poses = scene.get_future_trajectory(8).poses.copy()
    poses[:, 1] *= -1
    poses[:, 2] *= -1
    v0 = float(np.linalg.norm(scene.frames[3].ego_status.ego_velocity[:2]))
    dec = run_l1(scene, poses, v0)
    assert "turn_left" not in lat(dec), dec


# ---------------------------------------------------------------------------------------------------------------

def test_lane_index_and_margin():
    assert lane_index(0.0, 1.8, 1.8) == 0
    assert lane_index(1.9, 1.8, 1.8) == 1
    assert lane_index(1.8 + 3.6 + 0.1, 1.8, 1.8) == 2
    assert lane_index(-1.9, 1.8, 1.8) == -1
    assert boundary_margin(1.7, 1.8, 1.8) == pytest.approx(0.1)
    assert boundary_margin(0.0, 1.8, 1.8) == pytest.approx(1.8)
    assert boundary_margin(-3.6, 1.8, 1.8) == pytest.approx(1.8)


def test_same_decision():
    a = Decision([Segment("go", 0.5, "keep_lane", 0.0), Segment("go", 0.5, "lane_change_left", 0.6)])
    b = Decision([Segment("go", 0.6, "keep_lane", 0.0), Segment("go", 0.55, "lane_change_left", 0.75)])
    c = Decision([Segment("go", 0.5, "keep_lane", 0.0), Segment("go", 0.5, "offset_left", 0.6)])
    assert same_decision(a, b) is True
    assert same_decision(a, c) is False
    assert same_decision(a, Decision([], valid=False)) is None
