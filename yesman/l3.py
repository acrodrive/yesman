"""L3 counterfactual decision 생성과 불가능 기준 분류 (yesman.md 7.3, 6단계).

CF 메뉴 (장면마다, 원래 결정 d0에서 만든다): 5단계의 교훈대로 결정의 모양은 사람 경로를 L1이 실제로 읽는 모양을 따른다.
차선 중앙에서 출발한 navtrain 사람 경로에서 가장 흔한 (구간 1, 구간 2) 조합과 세기 중앙값이다 (docs/step06_cf.md).
- 좌우 바꾸기 (앞뒤는 원래대로)
    keep:          [keep, keep]
    offset_L/R:    [offset 0.4, keep]
    lc_L/R:        [offset 0.6, lane change 0.5]
    turn_L/R:      [turn 0.6, turn 0.6]      (교차로에서 가장 흔한 회전)
    turn_late_L/R: [keep, turn 0.6]          (조금 가다가 회전. 교차로 밖 좌회전에서도 가장 흔하다)
- 세기 바꾸기 (좌우는 원래대로)
    faster: 구간마다 go 세기 +0.3 (stop이면 go 0.3)
    slower: 구간마다 go 세기 -0.3 (원래 세기가 0.15 미만이거나 stop인 구간은 그대로. 바뀌는 구간이 없으면 만들지 않는다)
- 앞뒤 바꾸기 (좌우는 원래대로)
    stop: [stop s, stop 0] — s = 지금 속도에서 2초 안에 멈추는 평균 감속도 / a_max (최대 1)
원래 결정과 같은 CF(d̂ ≈ d)는 만들지 않는다.

불가능 기준 분류 (CF⁻, 계획서 7.3의 순서):
1. 세기 기준: CF에서 바뀐 세기를 0.3 낮춘 결정이 "가능"이면 (그 행동 자체는 할 수 있지만 그 세기로는 못 한다)
2. 다른 차·보행자 기준: 아니면, 채점한 후보 중 충돌로 떨어진 비율이 절반 이상이고 도로 이탈, 역주행 비율보다 크면
3. 도로 모양 기준: 그 외 (주로 도로 이탈, 역주행)
보이지 않는 원인: 다른 차·보행자 기준에서, 대표 후보가 처음 부딪히는 물체가 현재 시점에 전방 3캠 시야(카메라 기준 방위각
±87도) 밖에 있거나 아직 없으면 표시한다. 대표 후보는 계획 경로(시뮬레이션 전)로 부딪힘을 다시 찾으므로, 찾지 못하면 unknown이다.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from shapely.geometry import Polygon

from yesman.decision import Decision, Segment, same_decision
from yesman.l1 import SEG_DURATION, load_thresholds

STRENGTH_DROP = 0.3  # 세기 기준 확인: 바뀐 세기를 이만큼 낮춘다
CAM_X = 1.65  # 전방 카메라 위치 (뒤차축 앞) [m]
CAM_HALF_FOV = np.deg2rad(87.0)  # L0, F0, R0가 함께 덮는 방위각 (카메라 기준 ±87도)


def _seg(s: Segment, **kw) -> Segment:
    d = dict(lon=s.lon, lon_strength=s.lon_strength, lat=s.lat, lat_strength=s.lat_strength)
    d.update(kw)
    return Segment(**d)


def cf_menu(d0: Decision, v0: float) -> Dict[str, Decision]:
    """원래 결정 d0에서 CF 후보를 만든다. 이름 → 결정."""
    th = load_thresholds()
    s1, s2 = d0.segments
    lat = {"keep": [("keep_lane", 0.0), ("keep_lane", 0.0)]}
    for side, short in (("left", "L"), ("right", "R")):
        lat[f"offset_{short}"] = [(f"offset_{side}", 0.4), ("keep_lane", 0.0)]
        lat[f"lc_{short}"] = [(f"offset_{side}", 0.6), (f"lane_change_{side}", 0.5)]
        lat[f"turn_{short}"] = [(f"turn_{side}", 0.6), (f"turn_{side}", 0.6)]
        lat[f"turn_late_{short}"] = [("keep_lane", 0.0), (f"turn_{side}", 0.6)]
    out = {}
    for name, ((a1, x1), (a2, x2)) in lat.items():
        out[name] = Decision([_seg(s1, lat=a1, lat_strength=x1), _seg(s2, lat=a2, lat_strength=x2)])
    out["faster"] = Decision([_seg(s, lon="go", lon_strength=min(1.0, s.lon_strength + 0.3) if s.lon == "go" else 0.3)
                              for s in (s1, s2)])
    slow = [_seg(s, lon_strength=s.lon_strength - 0.3) if (s.lon == "go" and s.lon_strength >= 0.45) else _seg(s)
            for s in (s1, s2)]
    if any(a.lon_strength != b.lon_strength for a, b in zip(slow, (s1, s2))):
        out["slower"] = Decision(slow)
    stop1 = float(np.clip(v0 / SEG_DURATION / th["a_max"], 0.0, 1.0))
    out["stop"] = Decision([_seg(s1, lon="stop", lon_strength=stop1), _seg(s2, lon="stop", lon_strength=0.0)])
    return {k: v for k, v in out.items() if not same_decision(d0, v, allow_neighbor=False)}


def cf_menu_wide(d0: Decision, v0: float, rng: np.random.Generator) -> Dict[str, Decision]:
    """12단계 A: 넓힌 CF 메뉴 (결과를 보기 전에 정함). 실제 VLM의 틀린 결정은 앞뒤와 좌우를 동시에 바꾸거나 메뉴에 없는
    세기를 쓴다(12단계 분석). 장면마다 4개: 동시 변경 2개, 좌우 세기 무작위 1개, 속도 무작위 1개.
    - 동시 변경: cf_menu의 좌우 CF 하나 + 앞뒤 CF(faster / slower / stop) 하나를 함께 적용한다.
    - 좌우 세기 무작위: cf_menu의 좌우 CF 하나의 좌우 세기를 U(0.2, 1.0)으로 바꾼다.
    - 속도 무작위: 구간마다 go 세기를 U(0.05, 1.0)으로 바꾼다(좌우는 원래대로).
    원래 결정과 같은 것(이웃 라벨 허용 없이)은 뺀다."""
    base = cf_menu(d0, v0)
    lat_names = [k for k in base if k in ("keep", "offset_L", "offset_R", "lc_L", "lc_R", "turn_L", "turn_R",
                                          "turn_late_L", "turn_late_R")]
    lon_names = [k for k in base if k in ("faster", "slower", "stop")]
    out = {}
    for j in range(2):
        if not (lat_names and lon_names):
            break
        a, b = base[rng.choice(lat_names)], base[rng.choice(lon_names)]
        segs = [_seg(sa, lon=sb.lon, lon_strength=sb.lon_strength) for sa, sb in zip(a.segments, b.segments)]
        out[f"combo{j}"] = Decision(segs)
    if lat_names:
        a = base[rng.choice(lat_names)]
        segs = [_seg(sa, lat_strength=float(rng.uniform(0.2, 1.0))) if sa.lat != "keep_lane" else _seg(sa)
                for sa in a.segments]
        out["latstr"] = Decision(segs)
    out["lonrand"] = Decision([_seg(s, lon="go", lon_strength=float(rng.uniform(0.05, 1.0))) for s in d0.segments])
    return {k: v for k, v in out.items() if not same_decision(d0, v, allow_neighbor=False)}


def change_type(d0: Decision, d: Decision) -> str:
    """원래 결정과 무엇이 다른가: lat_type / lon_type / strength, 그리고 바뀐 구간 수."""
    kinds, segs = set(), 0
    for a, b in zip(d0.segments, d.segments):
        k = set()
        if a.lat != b.lat:
            k.add("lat_type")
        elif a.lat != "keep_lane" and abs(a.lat_strength - b.lat_strength) > 1e-6:
            k.add("strength")
        if a.lon != b.lon:
            k.add("lon_type")
        elif abs(a.lon_strength - b.lon_strength) > 1e-6:
            k.add("strength")
        kinds |= k
        segs += bool(k)
    order = ["lat_type", "lon_type", "strength"]
    return "+".join(x for x in order if x in kinds) + f"/{segs}seg"


def weaker(d0: Decision, d: Decision) -> Optional[Decision]:
    """세기 기준 확인용: CF에서 원래 결정과 다른 행동의 세기를 STRENGTH_DROP만큼 낮춘 결정. 낮출 것이 없으면 None."""
    segs, changed = [], False
    for a, b in zip(d0.segments, d.segments):
        kw = {}
        if (b.lat != a.lat or abs(b.lat_strength - a.lat_strength) > 1e-6) and b.lat != "keep_lane":
            kw["lat_strength"] = max(0.05, b.lat_strength - STRENGTH_DROP)
        if b.lon != a.lon or abs(b.lon_strength - a.lon_strength) > 1e-6:
            if b.lon == "go" and a.lon == "go":
                # 빠르게/느리게: 원래 세기 쪽으로 STRENGTH_DROP만큼 되돌린다
                step = np.sign(b.lon_strength - a.lon_strength) * STRENGTH_DROP
                kw["lon_strength"] = float(np.clip(b.lon_strength - step, 0.0, 1.0))
            elif b.lon_strength > 0:
                kw["lon_strength"] = max(0.0, b.lon_strength - STRENGTH_DROP)
        if kw:
            changed = True
            kw = {k: v for k, v in kw.items() if abs(v - getattr(b, k)) > 1e-6}
        segs.append(_seg(b, **kw))
    out = Decision(segs)
    return out if changed and not same_decision(d, out, allow_neighbor=False) else None


def categorize(fail_frac: Dict[str, float], weaker_feasible: Optional[bool]) -> str:
    """불가능 기준: strength / agent / road."""
    if weaker_feasible:
        return "strength"
    c = fail_frac.get("no_at_fault_collisions", 0.0)
    r = max(fail_frac.get("drivable_area_compliance", 0.0), fail_frac.get("driving_direction_compliance", 0.0))
    if c >= 0.5 and c >= r:
        return "agent"
    return "road"


# ----------------------------------------------------------------------------------------------------------------
# 보이지 않는 원인: 대표 후보가 처음 부딪히는 물체가 현재 시점에 전방 3캠 시야 밖인가
# ----------------------------------------------------------------------------------------------------------------

def _ego_polygon(x: float, y: float, h: float) -> Polygon:
    vp = get_pacifica_parameters()
    car = OrientedBox(StateSE2(x + vp.rear_axle_to_center * np.cos(h), y + vp.rear_axle_to_center * np.sin(h), h),
                      vp.length, vp.width, vp.height)
    return car.geometry


def _boxes_in_current_frame(scene, frame_idx: int) -> Tuple[List[str], List[Polygon], np.ndarray]:
    """프레임 frame_idx의 물체 박스를 현재 ego 좌표계로 바꾼다. (track 토큰, 다각형, 중심 (n, 2))"""
    cur = scene.scene_metadata.num_history_frames - 1
    x0, y0, h0 = scene.frames[cur].ego_status.ego_pose
    fr = scene.frames[frame_idx]
    xi, yi, hi = fr.ego_status.ego_pose
    toks, polys, centers = [], [], []
    for t, b in zip(fr.annotations.track_tokens, fr.annotations.boxes):
        gx = xi + np.cos(hi) * b[0] - np.sin(hi) * b[1]
        gy = yi + np.sin(hi) * b[0] + np.cos(hi) * b[1]
        gh = hi + b[6]
        dx, dy = gx - x0, gy - y0
        lx, ly = np.cos(h0) * dx + np.sin(h0) * dy, -np.sin(h0) * dx + np.cos(h0) * dy
        polys.append(OrientedBox(StateSE2(lx, ly, gh - h0), b[3], b[4], b[5]).geometry)
        toks.append(t)
        centers.append((lx, ly))
    return toks, polys, np.array(centers).reshape(-1, 2)


def visible_now(poly: Polygon) -> bool:
    """다각형의 꼭짓점 하나라도 전방 3캠 시야(카메라 기준 방위각 ±87도) 안이면 보인다고 본다 (가림은 무시)."""
    xy = np.asarray(poly.exterior.coords)
    az = np.arctan2(xy[:, 1], xy[:, 0] - CAM_X)
    return bool(np.any(np.abs(az) <= CAM_HALF_FOV))


def collision_cause_visibility(scene, poses: np.ndarray) -> str:
    """대표 후보(계획 경로 (8, 3))가 처음 부딪히는 물체를 찾아 visible / unseen / unknown을 낸다."""
    cur = scene.scene_metadata.num_history_frames - 1
    now_toks, now_polys, _ = _boxes_in_current_frame(scene, cur)
    now = dict(zip(now_toks, now_polys))
    for i in range(8):
        fi = cur + 1 + i
        if fi >= len(scene.frames):
            break
        ego = _ego_polygon(*poses[i])
        toks, polys, _ = _boxes_in_current_frame(scene, fi)
        for t, p in zip(toks, polys):
            if ego.intersects(p):
                if t not in now:
                    return "unseen"  # 현재 시점에는 아직 없던 물체
                return "visible" if visible_now(now[t]) else "unseen"
    return "unknown"
