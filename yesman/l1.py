"""L1 결정 라벨러 (yesman.md 6.1, 7.1): 경로 하나와 지도의 차선 정보로 결정 d를 계산한다.

라벨을 만들 때(사람 경로)와 평가할 때(모델 경로) 모두 이 함수 하나를 쓴다.

두 단계로 나뉜다.
1. extract_features(): 지도를 써서 경로를 차선 중심선 좌표계로 바꾸고, 구간마다 필요한 값을 계산한다 (느림).
2. classify(): 기준값(thresholds)으로 행동과 세기를 정한다 (빠름). 기준값은 navtrain 분포로 정한다(l1_thresholds.yaml).

좌표와 시간
- 경로는 NAVSIM 형식이다. ego 좌표계(현재 뒤차축 원점, x 앞, y 왼쪽, heading 반시계 +)에서 0.5초 간격 8점(4초)이다.
- 현재 위치 (0, 0, 0)을 앞에 붙여 9점(t = 0, 0.5, ..., 4.0초)으로 쓴다. 구간 k는 점 SEGMENTS[k][0] ~ SEGMENTS[k][1]이다.

기준 차선열 (reference chain)
- 출발점을 포함하고 방향이 ego와 맞는 차선(또는 교차로 연결 차선)에서 시작하여, 이어지는 차선(outgoing edges)을
  경로가 지나간 거리만큼 따라간 차선열들을 만든다. 그중 경로와의 옆 거리 평균이 가장 작은 차선열을 기준으로 쓴다.
  따라서 교차로에서 좌회전한 경로는 좌회전 연결 차선을 따라가는 차선열이 기준이 된다.
  고를 때의 비용 = 옆 거리 평균 + CHAIN_HEADING_COST x 방향 차이 평균 + 차선열 끝을 넘은 점의 벌점.
- 경로의 각 점을 기준 중심선에 투영하여 진행 거리 s와 옆 거리 d(왼쪽 +)를 구한다. 그 점에서의 차선 반폭(왼쪽, 오른쪽)은
  차선 경계선까지의 거리로 구한다.

좌우 행동 판정 (구간마다, 위에서부터 먼저 해당하는 것)
1. turn: (a) 구간이 교차로 연결 차선(lane connector)을 지나고, 경로 방향이 turn_min_deg 이상 바뀌었다.
   세기 = |방향 변화| / turn_max_deg. 또는 (b) 구간 끝에서 경로 방향이 기준 차선 방향과 turn_offlane_deg 이상
   틀어졌다(교차로 밖, 즉 연결 차선을 지나지 않는 구간에서 차선을 벗어나 꺾음. 5단계에서 추가).
   세기 = |차선 대비 방향| / turn_max_deg. 교차로 안에서는 기준 차선열 선택이 흔들려 (b)를 쓰지 않는다.
   굽은 길을 따라가면 차선도 같이 굽으므로 (b)에 해당하지 않는다.
2. lane change: 구간 시작과 끝의 차선 번호가 다르다. 차선 번호는 기준 차선을 0으로, 왼쪽으로 한 차선 폭마다 +1이다.
   옆 차선이 실제로 있는지는 보지 않는다(기하만으로 판정). 그래야 도로가 없는 쪽으로 차선 변경을 따라간 경로도 lane change로 잡힌다.
   세기 = 구간 안에서 0.5초마다 잰 옆 이동 속도의 최대값 / vlat_max.
3. offset: 같은 차선 안에서 옆 거리가 offset_min 이상 바뀌었다. 세기 = |옆 거리 변화| / offset_max.
   차선 중심선의 방향이 구간 동안 curve_deg 이상 바뀌면(굽은 도로) 기준을 offset_min_curve로 키운다.
   사람은 굽은 길에서 코너 안쪽으로 가로질러 달리므로, 그만큼의 옆 거리 변화는 차선을 따라간 것으로 본다.
4. 나머지는 keep lane (세기 0). 굽은 도로에서 차선을 따라 달린 경로는 d가 거의 변하지 않으므로 keep lane이 된다.

앞뒤 행동 판정 (구간마다)
- 구간 끝 속도가 v_stop 미만이면 stop, 아니면 go이다. 구간 끝 속도는 마지막 두 0.5초 평균 속도에서 선형 외삽한 값이다.
- go 세기 = 구간 끝 속도 / v_max. stop 세기 = 구간 평균 감속도 (구간 시작 속도 - 끝 속도) / 2초 / a_max.
  구간 1의 시작 속도는 현재 차량 속도(ego status)이고, 구간 2의 시작 속도는 구간 1의 끝 속도이다.

애매함 (ambiguous): 값이 기준값 경계의 margin 안에 있으면 그 구간의 라벨을 애매함으로 표시한다. 출발 차선을 찾지 못하면
결정 전체가 invalid이다(주차장 등). 경로가 기준 차선열 밖으로 크게 벗어나면(차선 방향과 heading_rel_max_deg 이상 어긋남,
차선열 끝을 넘어감) 좌우 행동을 애매함으로 표시한다.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer
from nuplan.common.maps.abstract_map_objects import LaneConnector, LaneGraphEdgeMapObject
from shapely.geometry import Point

from yesman.decision import Decision, Segment

DT = 0.5  # 경로 점 간격 [s]
SEGMENTS: Tuple[Tuple[int, int], ...] = ((0, 4), (4, 8))  # 9점(t=0..4초) 기준 구간 경계: 0~2초, 2~4초
SEG_DURATION = 2.0  # [s]
THRESHOLDS_PATH = Path(__file__).with_name("l1_thresholds.yaml")

# 지도 탐색 설정 (기준값이 아니라 탐색 범위이다)
START_SEARCH_RADIUS = 3.0  # 출발 차선 후보를 찾는 반경 [m]
START_MAX_DIST = 1.0  # 출발점이 차선 다각형 밖이어도 이 거리 안이면 후보로 본다 [m]
START_HEADING_TOL = np.deg2rad(60)  # 출발 차선 방향과 ego 방향의 최대 차이
CHAIN_EXTRA = 15.0  # 경로가 지나간 거리보다 이만큼 더 길게 차선열을 만든다 [m]
MAX_CHAINS = 64
CHAIN_BEYOND_PENALTY = 2.0  # 차선열을 고를 때, 차선열 끝을 넘은 경로 점 하나마다 더하는 벌점 [m]
CHAIN_HEADING_COST = 0.05  # 차선열을 고를 때, 경로와 중심선의 방향 차이 1도마다 더하는 비용 [m/deg] (20도 = 옆 거리 1 m)


@lru_cache(maxsize=1)
def load_thresholds(path: Optional[str] = None) -> Dict[str, float]:
    with open(path or THRESHOLDS_PATH) as f:
        return {k: float(v) for k, v in yaml.safe_load(f).items()}


# ----------------------------------------------------------------------------------------------------------------
# 지도: 차선 하나의 중심선과 경계선 (프로세스 안에서 캐시)
# ----------------------------------------------------------------------------------------------------------------

@dataclass
class _Edge:
    id: str
    xy: np.ndarray  # (n, 2) 중심선
    left: object  # shapely LineString (왼쪽 경계)
    right: object
    is_connector: bool
    obj: LaneGraphEdgeMapObject

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.xy, axis=0), axis=1).sum())


_EDGE_CACHE: Dict[Tuple[int, str], _Edge] = {}


def _edge(map_api: AbstractMap, obj: LaneGraphEdgeMapObject) -> _Edge:
    key = (id(map_api), obj.id)
    e = _EDGE_CACHE.get(key)
    if e is None:
        xy = np.array([[p.x, p.y] for p in obj.baseline_path.discrete_path], dtype=np.float64)
        keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6]  # 겹친 점 제거
        e = _Edge(obj.id, xy[keep], obj.left_boundary.linestring, obj.right_boundary.linestring,
                  isinstance(obj, LaneConnector), obj)
        _EDGE_CACHE[key] = e
    return e


def _chains(map_api: AbstractMap, start: _Edge, need: float) -> List[List[_Edge]]:
    """start에서 이어지는 차선을 따라 길이 need 이상이 되는 차선열을 모두 만든다 (갈래마다 하나)."""
    out: List[List[_Edge]] = []

    def rec(path: List[_Edge], length: float):
        if len(out) >= MAX_CHAINS:
            return
        nxt = path[-1].obj.outgoing_edges if length < need else []
        nxt = [_edge(map_api, o) for o in nxt if o.id not in {p.id for p in path}]
        if not nxt:
            out.append(path)
            return
        for e in nxt:
            rec(path + [e], length + e.length)

    rec([start], start.length)
    return out


# ----------------------------------------------------------------------------------------------------------------
# 폴리라인 투영
# ----------------------------------------------------------------------------------------------------------------

@dataclass
class _Polyline:
    xy: np.ndarray  # (n, 2)
    s: np.ndarray  # (n,) 누적 거리
    edge_idx: np.ndarray  # (n-1,) 각 선분이 속한 차선열 안의 차선 번호
    edges: List[_Edge]

    @classmethod
    def from_chain(cls, chain: List[_Edge]) -> "_Polyline":
        xys, idx = [], []
        for k, e in enumerate(chain):
            xy = e.xy if not xys else e.xy[1:] if np.linalg.norm(e.xy[0] - xys[-1][-1]) < 0.5 else e.xy
            if len(xy) == 0:
                continue
            if xys:  # 앞 차선의 끝점과 이어지는 선분은 다음 차선에 속한다
                idx.append(k)
            idx.extend([k] * (len(xy) - 1))
            xys.append(xy)
        xy = np.concatenate(xys)
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
        return cls(xy, s, np.array(idx, dtype=int), chain)

    def project(self, q: np.ndarray, s_lo: float = -np.inf, s_hi: float = np.inf) -> Tuple[float, float, int, float, bool]:
        """점 q를 투영한다. (s, d(왼쪽 +), 선분 번호, 중심선 방향, 끝을 넘었는지)를 낸다. s_lo~s_hi 안의 선분만 본다."""
        a, b = self.xy[:-1], self.xy[1:]
        seg = b - a
        L2 = (seg ** 2).sum(1)
        t = np.clip(((q - a) * seg).sum(1) / L2, 0.0, 1.0)
        foot = a + t[:, None] * seg
        dist = np.linalg.norm(q - foot, axis=1)
        s_foot = self.s[:-1] + t * np.sqrt(L2)
        dist = np.where((s_foot >= s_lo) & (s_foot <= s_hi), dist, np.inf)
        i = int(np.argmin(dist))
        if not np.isfinite(dist[i]):  # 창 안에 선분이 없으면 전체에서 찾는다
            return self.project(q)
        tan = seg[i] / np.sqrt(L2[i])
        rel = q - foot[i]
        d = float(np.sign(tan[0] * rel[1] - tan[1] * rel[0]) * dist[i])
        # 끝을 넘었는지: 마지막 선분에서 t가 1로 잘렸고 실제로 더 나갔다
        raw_t = ((q - a[i]) * seg[i]).sum() / L2[i]
        beyond = (i == len(seg) - 1 and raw_t > 1.0 + 1e-6) or (i == 0 and raw_t < -1e-6)
        return float(s_foot[i]), d, i, float(np.arctan2(tan[1], tan[0])), bool(beyond)


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


# ----------------------------------------------------------------------------------------------------------------
# 1단계: 특징 추출
# ----------------------------------------------------------------------------------------------------------------

def local_to_global(poses: np.ndarray, ego_pose: Sequence[float]) -> np.ndarray:
    """ego 좌표계 (x, y, heading) 배열을 전역 좌표로 바꾼다."""
    x0, y0, h0 = ego_pose
    c, s = np.cos(h0), np.sin(h0)
    out = np.empty_like(poses, dtype=np.float64)
    out[:, 0] = x0 + c * poses[:, 0] - s * poses[:, 1]
    out[:, 1] = y0 + s * poses[:, 0] + c * poses[:, 1]
    out[:, 2] = _wrap(h0 + poses[:, 2])
    return out


def motion_features(poses: np.ndarray, v0: float) -> Dict[str, float]:
    """지도가 필요 없는 구간별 값: 시작 속도, 끝 속도, 방향 변화."""
    loc = np.vstack([[0.0, 0.0, 0.0], np.asarray(poses, dtype=np.float64).reshape(8, 3)])
    step = np.linalg.norm(np.diff(loc[:, :2], axis=0), axis=1)  # (8,)
    avg = np.r_[np.nan, step / DT]  # avg[i]: 점 i 직전 0.5초 평균 속도 (시각 t_i - 0.25초의 속도로 본다)
    # 점 i 시각의 속도: 앞의 두 평균 속도로 선형 외삽한다 (등가속이면 정확하다). 0.5초 평균을 그대로 쓰면
    # 구간 끝에 막 멈춘 경로도 끝 속도가 0보다 크게 나온다.
    speed = np.r_[v0, avg[1], np.maximum(0.0, 1.5 * avg[2:] - 0.5 * avg[1:-1])]
    f = {}
    for k, (a, b) in enumerate(SEGMENTS, 1):
        f[f"seg{k}_v_start"] = float(speed[a])
        f[f"seg{k}_v_end"] = float(speed[b])
        f[f"seg{k}_dh"] = float(_wrap(loc[b, 2] - loc[a, 2]))
    return f


def _start_candidates(map_api: AbstractMap, x: float, y: float, h: float) -> List[Tuple[_Edge, float]]:
    """출발점을 포함하고 방향이 맞는 차선과 연결 차선. (차선, 출발점의 |d|) 목록."""
    objs = map_api.get_proximal_map_objects(Point2D(x, y), START_SEARCH_RADIUS,
                                            [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR])
    p = Point(x, y)
    out = []
    for layer in (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR):
        for o in objs[layer]:
            if o.polygon.distance(p) > START_MAX_DIST:
                continue
            e = _edge(map_api, o)
            if len(e.xy) < 2:
                continue
            _, d, _, th, _ = _Polyline.from_chain([e]).project(np.array([x, y]))
            if abs(_wrap(th - h)) <= START_HEADING_TOL:
                out.append((e, abs(d)))
    return out


def extract_features(poses: np.ndarray, v0: float, ego_pose: Sequence[float], map_api: AbstractMap,
                     debug: Optional[dict] = None) -> Dict[str, float]:
    """경로 하나의 L1 특징을 계산한다.

    :param poses: (8, 3) ego 좌표계 경로 (0.5초 간격, 4초)
    :param v0: 현재 속도 [m/s] (ego status 속도 벡터의 크기)
    :param ego_pose: 현재 ego의 전역 pose (x, y, heading)
    :param map_api: nuPlan 지도
    :param debug: dict를 주면 그림용 값(기준 중심선, 투영 결과 등)을 채운다
    :return: 특징 dict. "valid"가 0이면 "reason"에 이유가 있다.
    """
    poses = np.asarray(poses, dtype=np.float64)
    assert poses.shape == (8, 3), poses.shape
    loc = np.vstack([[0.0, 0.0, 0.0], poses])  # (9, 3) t = 0..4초
    glob = local_to_global(loc, ego_pose)
    f: Dict[str, float] = {"valid": 0, "reason": "", "v0": float(v0)}

    # 앞뒤 행동용 값 (지도 불필요)
    f.update(motion_features(poses, v0))
    step = np.linalg.norm(np.diff(loc[:, :2], axis=0), axis=1)  # (8,)

    # 출발 차선과 기준 차선열
    cands = _start_candidates(map_api, glob[0, 0], glob[0, 1], glob[0, 2])
    f["n_start_cands"] = len(cands)
    if not cands:
        f["reason"] = "no_start_lane"
        return f
    traveled = float(step.sum())
    starts = {e.id: e for e, _ in cands}
    for e, _ in cands:  # 출발점이 차선 시작점보다 뒤에 있으면 앞 차선(incoming)에서 시작하는 차선열도 본다
        if _Polyline.from_chain([e]).project(glob[0, :2])[4]:
            for o in e.obj.incoming_edges:
                starts.setdefault(o.id, _edge(map_api, o))
    best = None
    for e in starts.values():
        s0, _, _, _, _ = _Polyline.from_chain([e]).project(glob[0, :2])
        for chain in _chains(map_api, e, need=s0 + traveled + CHAIN_EXTRA):
            pl = _Polyline.from_chain(chain)
            proj = [pl.project(glob[0, :2], -np.inf, s0 + 1.0)]
            for i in range(1, 9):  # 앞 점보다 조금 뒤 ~ 지나간 거리보다 조금 앞 범위에서 찾는다 (U자 차선에서 건너뛰지 않도록)
                proj.append(pl.project(glob[i, :2], proj[-1][0] - 2.0, proj[-1][0] + step[i - 1] + 2.0))
            # 옆 거리 평균 + 방향 차이 평균 + 차선열 밖(앞뒤 끝을 넘은) 점마다 벌점
            hdiff = np.degrees(np.abs(_wrap(glob[:, 2] - np.array([p[3] for p in proj]))))
            cost = float(np.mean([abs(p[1]) for p in proj]) + CHAIN_HEADING_COST * hdiff.mean()
                         + CHAIN_BEYOND_PENALTY * sum(p[4] for p in proj))
            if best is None or cost < best[0]:
                best = (cost, pl, proj)
    cost, pl, proj = best
    f["chain_cost"] = cost
    f["chain_ids"] = ",".join(e.id for e in pl.edges)

    s = np.array([p[0] for p in proj])
    d = np.array([p[1] for p in proj])
    th = np.array([p[3] for p in proj])
    beyond = np.array([p[4] for p in proj])
    wl, wr = np.zeros(9), np.zeros(9)
    for i, p in enumerate(proj):  # 투영점에서 그 차선의 왼쪽, 오른쪽 경계까지 거리 = 반폭
        e = pl.edges[pl.edge_idx[p[2]]]
        foot = Point(*(glob[i, :2] - np.array([-np.sin(th[i]), np.cos(th[i])]) * d[i]))
        wl[i], wr[i] = e.left.distance(foot), e.right.distance(foot)
    hrel = _wrap(glob[:, 2] - th)  # 경로 방향 - 중심선 방향

    # 연결 차선 구간 [s_start, s_end]
    conn = []
    for k, e in enumerate(pl.edges):
        seg_ids = np.where(pl.edge_idx == k)[0]
        if e.is_connector and len(seg_ids):
            conn.append((pl.s[seg_ids[0]], pl.s[seg_ids[-1] + 1]))

    if debug is not None:
        debug.update(glob=glob, chain_xy=pl.xy, chain_s=pl.s, s=s, d=d, wl=wl, wr=wr, conn=conn,
                     chain_edges=pl.edges, start_cands=[e.id for e, _ in cands])
    f["start_d"] = float(d[0])
    for name, arr in (("s", s), ("d", d), ("wl", wl), ("wr", wr), ("lane_h", th), ("hrel", hrel)):
        f[f"pt_{name}"] = arr.astype(np.float32).tolist()  # 점 9개(t = 0..4초)의 값. 분석용
    for k, (a, b) in enumerate(SEGMENTS, 1):
        lo, hi = s[a], s[b]
        f[f"seg{k}_conn_overlap"] = float(sum(max(0.0, min(hi, c1) - max(lo, c0)) for c0, c1 in conn))
        f[f"seg{k}_on_conn_start"] = float(any(c0 <= lo <= c1 for c0, c1 in conn))
        for tag, i in (("start", a), ("end", b)):
            f[f"seg{k}_d_{tag}"] = float(d[i])
            f[f"seg{k}_wl_{tag}"] = float(wl[i])
            f[f"seg{k}_wr_{tag}"] = float(wr[i])
        f[f"seg{k}_ds"] = float(hi - lo)
        f[f"seg{k}_vlat_peak"] = float(np.max(np.abs(np.diff(d[a:b + 1]))) / DT)
        f[f"seg{k}_hrel_end"] = float(hrel[b])
        f[f"seg{k}_lane_dh"] = float(_wrap(th[b] - th[a]))  # 차선 중심선의 방향 변화 (도로가 굽은 정도)
        f[f"seg{k}_hrel_max"] = float(np.max(np.abs(hrel[a:b + 1])))
        f[f"seg{k}_beyond"] = float(beyond[a:b + 1].any())
    f["valid"] = 1
    return f


# ----------------------------------------------------------------------------------------------------------------
# 2단계: 분류
# ----------------------------------------------------------------------------------------------------------------

def lane_index(d: float, wl: float, wr: float) -> int:
    """기준 차선에서 몇 차선 옆인가 (왼쪽 +). 한 차선 폭 = wl + wr."""
    w = max(wl + wr, 1e-3)
    if d > wl:
        return 1 + int((d - wl) // w)
    if d < -wr:
        return -1 - int((-d - wr) // w)
    return 0


def boundary_margin(d: float, wl: float, wr: float) -> float:
    """d에서 가장 가까운 차선 경계까지의 거리."""
    w = max(wl + wr, 1e-3)
    if d >= 0:
        return abs(((d - wl) + w / 2) % w - w / 2)
    return abs(((-d - wr) + w / 2) % w - w / 2)


def classify(f: Dict[str, float], th: Optional[Dict[str, float]] = None) -> Decision:
    """특징과 기준값으로 결정을 정한다."""
    th = th or load_thresholds()
    if not f.get("valid"):
        return Decision([], valid=False, reason=str(f.get("reason", "")))
    segs, reasons = [], []
    for k in range(1, len(SEGMENTS) + 1):
        g = lambda name: f[f"seg{k}_{name}"]  # noqa: E731

        # 앞뒤 행동
        v_end, v_start = g("v_end"), g("v_start")
        if v_end < th["v_stop"]:
            lon = "stop"
            lon_strength = max(0.0, v_start - v_end) / SEG_DURATION / th["a_max"]
        else:
            lon = "go"
            lon_strength = v_end / th["v_max"]
        alt_lon = ""
        if abs(v_end - th["v_stop"]) < th["v_stop_margin"]:  # 끝 속도가 v_stop 근처 (go/stop 경계)
            alt_lon = "go" if lon == "stop" else "stop"
            reasons.append(f"seg{k}:stop_speed_margin")

        # 좌우 행동: 회전이 아닐 때의 라벨(lane change / offset / keep)을 먼저 정한다
        dh = np.degrees(g("dh"))
        on_conn = g("conn_overlap") > 0 or g("on_conn_start") > 0
        i0 = lane_index(g("d_start"), g("wl_start"), g("wr_start"))
        i1 = lane_index(g("d_end"), g("wl_end"), g("wr_end"))
        dd = g("d_end") - g("d_start")
        # 도로가 굽은 구간에서는 사람이 코너 안쪽으로 가로지르므로 offset 기준을 크게 둔다
        curved = abs(np.degrees(g("lane_dh"))) >= th["curve_deg"]
        off_min = th["offset_min_curve"] if curved else th["offset_min"]
        offset_dir = "offset_left" if dd > 0 else "offset_right"
        near_offset = abs(abs(dd) - off_min) < th["offset_margin"]
        if i1 != i0:
            nt_lat = "lane_change_left" if i1 > i0 else "lane_change_right"
            nt_strength, nt_alt = g("vlat_peak") / th["vlat_max"], ""
        elif abs(dd) >= off_min:
            nt_lat, nt_strength = offset_dir, abs(dd) / th["offset_max"]
            nt_alt = "keep_lane" if near_offset else ""
        else:
            nt_lat, nt_strength = "keep_lane", 0.0
            nt_alt = offset_dir if near_offset else ""

        # turn (1) 교차로: 연결 차선을 지나며 경로 방향이 turn_min_deg 이상 바뀜
        #      (2) 교차로 밖: 연결 차선을 지나지 않는 구간 끝에서 경로 방향이 차선 방향과 turn_offlane_deg 이상 틀어짐
        #          굽은 길을 따라가면 차선도 같이 굽으므로 (2)에 해당하지 않는다 (사람 경로 p99 16도)
        hrel = np.degrees(g("hrel_end"))
        conn_turn = on_conn and abs(dh) >= th["turn_min_deg"]
        off_turn = (not on_conn) and abs(hrel) >= th["turn_offlane_deg"]  # 교차로(연결 차선) 위에서는 (1)만 쓴다
        near_conn = on_conn and abs(abs(dh) - th["turn_min_deg"]) < th["turn_margin_deg"]
        near_off = (not on_conn) and abs(abs(hrel) - th["turn_offlane_deg"]) < th["turn_offlane_margin_deg"]
        if conn_turn:
            turn_dir, turn_strength = ("turn_left" if dh > 0 else "turn_right"), abs(dh) / th["turn_max_deg"]
        else:
            turn_dir, turn_strength = ("turn_left" if hrel > 0 else "turn_right"), abs(hrel) / th["turn_max_deg"]
        near_turn = (near_conn and not off_turn) or (near_off and not conn_turn)
        if near_turn:
            reasons.append(f"seg{k}:turn_margin")  # turn 기준 근처 (turn/keep 경계)
        if conn_turn or off_turn:
            lat, lat_strength = turn_dir, turn_strength
            alt_lat = nt_lat if near_turn else ""
        else:
            lat, lat_strength = nt_lat, nt_strength
            alt_lat = turn_dir if near_turn else nt_alt
            if nt_alt:
                reasons.append(f"seg{k}:offset_margin")  # 옆 거리 변화가 offset 기준 근처 (offset/keep 경계)

        # 구조적 애매함: 어느 차선 기준인지부터 불분명하다 (학습과 평가에서 뺀다)
        structural = False
        if not lat.startswith("turn"):
            # 차선 경계 근처에서 끝나거나 시작하면 lane change와 offset/keep이 흔들린다
            if min(boundary_margin(g("d_start"), g("wl_start"), g("wr_start")),
                   boundary_margin(g("d_end"), g("wl_end"), g("wr_end"))) < th["lat_boundary_margin"]:
                structural = True
                reasons.append(f"seg{k}:near_boundary")
            if np.degrees(g("hrel_max")) > th["heading_rel_max_deg"]:
                structural = True
                reasons.append(f"seg{k}:off_lane_heading")
        if g("beyond") > 0:
            structural = True
            reasons.append(f"seg{k}:beyond_chain")
        segs.append(Segment(lon, float(np.clip(lon_strength, 0, 1)), lat, float(np.clip(lat_strength, 0, 1)),
                            ambiguous_lon=bool(alt_lon), ambiguous_lat=bool(alt_lat) or structural,
                            alt_lon=alt_lon, alt_lat=alt_lat, structural=structural))
    return Decision(segs, valid=True, reason=";".join(reasons))


def l1(poses: np.ndarray, v0: float, ego_pose: Sequence[float], map_api: AbstractMap,
       th: Optional[Dict[str, float]] = None) -> Decision:
    """경로 → 결정. 라벨을 만들 때와 평가할 때 모두 이 함수를 쓴다."""
    return classify(extract_features(poses, v0, ego_pose, map_api), th)


def l1_from_scene(scene, poses: Optional[np.ndarray] = None, th: Optional[Dict[str, float]] = None) -> Decision:
    """NAVSIM Scene에서 L1을 부른다. poses가 없으면 사람 미래 경로를 쓴다."""
    f = features_from_scene(scene, poses)
    return classify(f, th)


def features_from_scene(scene, poses: Optional[np.ndarray] = None, debug: Optional[dict] = None) -> Dict[str, float]:
    cur = scene.scene_metadata.num_history_frames - 1
    status = scene.frames[cur].ego_status
    if poses is None:
        poses = scene.get_future_trajectory(8).poses
    v0 = float(np.linalg.norm(status.ego_velocity[:2]))
    return extract_features(poses, v0, status.ego_pose, scene.map_api, debug)


# ----------------------------------------------------------------------------------------------------------------
# 차선 기준으로 경로 다시 그리기 (5단계 L2: 다른 장면의 경로를 이 장면의 차선 위에 옮긴다)
# ----------------------------------------------------------------------------------------------------------------

@dataclass
class LaneReference:
    """장면 하나의 기준 차선열: 출발 차선에서 가장 곧게 이어지는 차선열과 ego의 투영 위치."""
    pl: _Polyline
    s0: float  # ego 뒤차축의 진행 거리
    d0: float  # ego 뒤차축의 옆 거리 (왼쪽 +)
    ego_pose: np.ndarray


REF_LENGTH = 120.0  # 기준 차선열 길이 (4초 동안 30 m/s도 들어간다) [m]
REF_HEADING_WINDOW = 40.0  # 곧은 정도를 잴 구간 길이 [m]


def lane_reference(ego_pose: Sequence[float], map_api: AbstractMap) -> Optional[LaneReference]:
    """출발 차선에서 이어지는 차선열 중, 앞 40 m 동안 방향이 가장 적게 바뀌는 것(직진에 가장 가까운 것)을 고른다.
    비용 = |출발 옆 거리| + CHAIN_HEADING_COST x |앞 40 m 방향 변화(도)|. 출발 차선이 없으면 None."""
    x, y, h = ego_pose
    cands = _start_candidates(map_api, x, y, h)
    if not cands:
        return None
    starts = {e.id: e for e, _ in cands}
    for e, _ in cands:
        if _Polyline.from_chain([e]).project(np.array([x, y]))[4]:
            for o in e.obj.incoming_edges:
                starts.setdefault(o.id, _edge(map_api, o))
    best = None
    for e in starts.values():
        s0, _, _, _, _ = _Polyline.from_chain([e]).project(np.array([x, y]))
        for chain in _chains(map_api, e, need=s0 + REF_LENGTH):
            pl = _Polyline.from_chain(chain)
            s, d, _, th0, beyond = pl.project(np.array([x, y]), -np.inf, s0 + 1.0)
            if beyond or pl.s[-1] - s < 30.0:
                continue
            s1 = min(s + REF_HEADING_WINDOW, pl.s[-1])
            i1 = min(int(np.searchsorted(pl.s, s1)), len(pl.xy) - 2)
            th1 = np.arctan2(*(pl.xy[i1 + 1] - pl.xy[i1])[::-1])
            cost = abs(d) + CHAIN_HEADING_COST * abs(np.degrees(_wrap(th1 - th0)))
            if best is None or cost < best[0]:
                best = (cost, LaneReference(pl, s, d, np.asarray(ego_pose, dtype=np.float64)))
    return best[1] if best else None


def embed_on_lane(ref: LaneReference, s_rel: np.ndarray, d_rel: np.ndarray, hrel: np.ndarray) -> Optional[np.ndarray]:
    """다른 장면의 경로를 그 장면 차선 기준의 (진행 거리, 옆 거리, 차선 대비 방향) 시간표로 받아 이 장면의 기준 차선 위에
    다시 그린다. 출발 옆 거리는 이 장면의 ego 위치(d0)에 맞추고, 그 뒤의 변화량은 원래 경로 그대로 쓴다.

    :param s_rel, d_rel, hrel: 점 9개(t = 0..4초)의 진행 거리, 옆 거리, 차선 대비 방향 (원래 장면 기준)
    :return: (8, 3) ego 좌표계 경로. 기준 차선열이 모자라면 None
    """
    s = ref.s0 + (np.asarray(s_rel, dtype=np.float64) - s_rel[0])
    d = ref.d0 + (np.asarray(d_rel, dtype=np.float64) - d_rel[0])
    if s.max() > ref.pl.s[-1] - 1.0 or s.min() < 0:
        return None
    px, py = np.interp(s, ref.pl.s, ref.pl.xy[:, 0]), np.interp(s, ref.pl.s, ref.pl.xy[:, 1])
    ds = 0.5
    tx = np.interp(s + ds, ref.pl.s, ref.pl.xy[:, 0]) - np.interp(s - ds, ref.pl.s, ref.pl.xy[:, 0])
    ty = np.interp(s + ds, ref.pl.s, ref.pl.xy[:, 1]) - np.interp(s - ds, ref.pl.s, ref.pl.xy[:, 1])
    tang = np.arctan2(ty, tx)
    gx, gy = px - np.sin(tang) * d, py + np.cos(tang) * d
    gh = tang + np.asarray(hrel, dtype=np.float64)
    x0, y0, h0 = ref.ego_pose
    c, sn = np.cos(h0), np.sin(h0)
    dx, dy = gx - x0, gy - y0
    loc = np.stack([c * dx + sn * dy, -sn * dx + c * dy, _wrap(gh - h0)], axis=1)
    # 출발점은 ego 원점이어야 한다 (d0, 방향 차이를 맞췄으므로 거의 0). 남은 작은 차이는 빼서 원점에 맞춘다
    loc[:, :2] -= loc[0, :2]
    loc[:, 2] = _wrap(loc[:, 2] - loc[0, 2])
    return loc[1:]
