"""12단계: CF⁺ 학습 목표 경로 합성 — 다른 장면의 경로를 옮겨 오는 대신, 이 장면 사람의 실제 경로를 결정에 맞게 고친다.

원칙: 결정이 바꾸는 부분만 고치고 나머지는 사람 경로 그대로 둔다("결정을 만족하는 가장 작은 변경").
- 앞뒤 CF(faster, slower, stop): 사람 경로의 모양(지나간 길)은 그대로 두고 속도만 바꾼다. 구간마다 끝 속도를
  결정의 세기에 맞추고(go: 세기 x v_max, stop: 0) 구간 안에서는 등가속으로 잇는다. 사람 경로보다 길어지면
  마지막 방향으로 곧게 늘인다. 사람이 거의 서 있었으면(4초 동안 2 m 미만) 기준 차선을 따라 그린다.
- 좌우 CF(keep, offset, lane change): 사람의 속도(점 사이 이동 거리)는 그대로 두고, 이 장면의 기준 차선
  (L1 lane_reference) 위에 옆 거리 곡선을 새로 그린다. offset: 그 구간 동안 세기 x offset_max만큼 부드럽게
  (smoothstep) 옮긴다. lane change: 그 구간 동안 최대 옆 속도가 세기 x vlat_max가 되게 옮긴다.
- 회전 CF: 사람의 속도는 그대로 두고, 그 방향의 회전 차선열(L1 turn_reference) 위에 그린다.
- 경계에 걸리지 않도록 크기와 시점을 조금씩 바꾼 변형을 만들고, L1이 결정과 맞다고 판정(세기 ±0.1, 이웃
  라벨 허용 없음)한 것 중 사람 경로에 가장 가까운 것을 앞에 둔다. 안전 확인(NAVSIM)은 scripts/l3_synth.py가 한다.
"""

from typing import List, Optional, Tuple

import numpy as np

from yesman.decision import Decision, same_decision
from yesman.l1 import (DT, LaneReference, classify, embed_on_lane, extract_features, lane_reference,
                       load_thresholds, turn_reference)

T9 = np.arange(9) * DT  # 0, 0.5, ..., 4.0초


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return 3 * u ** 2 - 2 * u ** 3


def _with_origin(poses: np.ndarray) -> np.ndarray:
    return np.vstack([[0.0, 0.0, 0.0], np.asarray(poses, dtype=np.float64).reshape(8, 3)])


def human_step(poses: np.ndarray) -> np.ndarray:
    """사람 경로의 0.5초마다 이동 거리 (8,)."""
    p = _with_origin(poses)
    return np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)


def speed_profile(v0: float, v_end: Tuple[float, float]) -> np.ndarray:
    """구간 끝 속도로 9점 속도를 만든다 (구간 안 등가속, 0 아래로는 자른다)."""
    v = np.empty(9)
    v[0] = v0
    start = v0
    for k, (a, b) in enumerate(((0, 4), (4, 8))):
        for i in range(a + 1, b + 1):
            v[i] = start + (v_end[k] - start) * (i - a) / (b - a)
        start = v_end[k]
    return np.maximum(v, 0.0)


def retime_along(poses: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """사람 경로(원점 포함 9점)를 따라 누적 거리 dist(9,)인 점을 고른다. 끝을 넘으면 마지막 방향으로 늘인다."""
    p = _with_origin(poses)
    seg = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    if cum[-1] < 0.5:  # 사람이 거의 서 있었다: 현재 방향(0)으로 곧게
        heading = 0.0
        xy = np.stack([dist * np.cos(heading), dist * np.sin(heading)], 1)
        return np.c_[xy[1:], np.zeros(8)]
    keep = np.r_[True, seg > 1e-3]
    cum_k, p_k = cum[keep], p[keep]
    out = []
    for s in dist:
        if s <= cum_k[-1]:
            j = min(int(np.searchsorted(cum_k, s, side="right")) - 1, len(cum_k) - 2)
            u = (s - cum_k[j]) / max(cum_k[j + 1] - cum_k[j], 1e-6)
            xy = p_k[j, :2] + u * (p_k[j + 1, :2] - p_k[j, :2])
        else:
            last = p_k[-1, :2] - p_k[-2, :2]
            last = last / max(np.linalg.norm(last), 1e-6)
            xy = p_k[-1, :2] + (s - cum_k[-1]) * last
        out.append(xy)
    xy = np.array(out)
    d = np.diff(xy, axis=0)
    h = np.arctan2(d[:, 1], d[:, 0])
    # 거의 움직이지 않는 점은 앞의 방향을 쓴다
    moving = np.linalg.norm(d, axis=1) > 0.05
    for i in range(len(h)):
        if not moving[i]:
            h[i] = h[i - 1] if i > 0 else 0.0
    return np.c_[xy[1:], h]


def lateral_curve(d: Decision, scale: float, frac: float, th) -> Optional[np.ndarray]:
    """좌우 결정(keep/offset/lane change)의 옆 거리 변화 (9,) (왼쪽 +). turn이 있으면 None."""
    out = np.zeros(9)
    base = 0.0
    for k, s in enumerate(d.segments):
        a, b = 4 * k, 4 * (k + 1)
        u = (T9 - T9[a]) / (2.0 * frac)
        if s.lat.startswith("turn"):
            return None
        sign = 1.0 if s.lat.endswith("left") else -1.0
        if s.lat.startswith("offset"):
            amp = sign * s.lat_strength * th["offset_max"] * scale
        elif s.lat.startswith("lane_change"):
            # smoothstep의 최대 기울기 = 1.5 x 크기 / 시간 → 최대 옆 속도가 세기 x vlat_max가 되는 크기
            amp = sign * s.lat_strength * th["vlat_max"] * (2.0 * frac) / 1.5 * scale
        else:
            amp = 0.0
        out[a:] += amp * smoothstep(u[a:]) if amp else 0.0
        base += amp
    return out


def _hrel(s_rel: np.ndarray, d_rel: np.ndarray) -> np.ndarray:
    ds = np.gradient(s_rel, DT)
    dd = np.gradient(d_rel, DT)
    return np.where(ds > 0.3, np.arctan2(dd, np.maximum(ds, 1e-3)), 0.0)


def candidates(d: Decision, d0: Decision, human: np.ndarray, v0: float, ego_pose, map_api,
               th=None) -> List[Tuple[str, np.ndarray]]:
    """결정 d를 만족하도록 고친 경로 후보들 (이름, (8, 3))."""
    th = th or load_thresholds()
    out = []
    lon_changed = any(a.lon != b.lon or abs(a.lon_strength - b.lon_strength) > 1e-6
                      for a, b in zip(d.segments, d0.segments))
    lat_changed = any(a.lat != b.lat or abs(a.lat_strength - b.lat_strength) > 1e-6
                      for a, b in zip(d.segments, d0.segments))
    if lon_changed and not lat_changed:
        # 사람이 거의 서 있었으면(4초 동안 2 m 미만) 경로 모양을 알 수 없으므로 기준 차선을 따라 늘인다
        still = human_step(human).sum() < 2.0
        ref = lane_reference(ego_pose, map_api) if still else None
        for sc in (1.0, 0.95, 1.05, 0.9, 1.1):
            v_end = tuple(0.0 if s.lon == "stop" else s.lon_strength * th["v_max"] * sc for s in d.segments)
            v = speed_profile(v0, v_end)
            dist = np.r_[0.0, np.cumsum((v[:-1] + v[1:]) / 2 * DT)]
            if still:
                p = embed_on_lane(ref, dist, np.zeros(9), np.zeros(9)) if ref is not None else None
                if p is not None:
                    out.append((f"retime-lane_{sc}", p))
            else:
                out.append((f"retime_{sc}", retime_along(human, dist)))
        return out
    step = human_step(human)
    s_rel = np.r_[0.0, np.cumsum(step)]
    turns = {s.lat[5:] for s in d.segments if s.lat.startswith("turn")}
    if turns:
        if len(turns) > 1:
            return out
        ref = turn_reference(ego_pose, map_api, next(iter(turns)))
        if ref is None:
            return out
        for sc in (1.0, 0.9, 1.1, 0.8, 1.2):
            p = embed_on_lane(ref, s_rel * sc, np.zeros(9), np.zeros(9))
            if p is not None:
                out.append((f"turnref_{sc}", p))
        return out
    ref = lane_reference(ego_pose, map_api)
    if ref is None:
        return out
    for scale in (1.0, 1.1, 0.9, 1.2):
        for frac in (1.0, 0.75):
            dc = lateral_curve(d, scale, frac, th)
            if dc is None:
                continue
            p = embed_on_lane(ref, s_rel, dc, _hrel(s_rel, dc))
            if p is not None:
                out.append((f"lat_{scale}_{frac}", p))
    return out


def matching(d: Decision, cands, v0: float, ego_pose, map_api, human: np.ndarray, th=None):
    """L1이 d와 맞다고 판정한 후보만, 사람 경로와의 평균 거리 순으로."""
    th = th or load_thresholds()
    ok = []
    for name, p in cands:
        dh = classify(extract_features(p, v0, ego_pose, map_api), th)
        if dh.usable and same_decision(d, dh, strength_tol=0.1, allow_neighbor=False):
            ade = float(np.linalg.norm(p[:, :2] - np.asarray(human)[:, :2], axis=1).mean())
            ok.append((ade, name, p))
    ok.sort(key=lambda x: x[0])
    return ok
