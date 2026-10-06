"""결정 d의 형식 (yesman.md 6.1).

4초 경로를 구간(기본 2개: 0~2초, 2~4초)으로 나누고, 구간마다 앞뒤 행동, 좌우 행동, 각각의 세기(0~1)를 정한다.
L1(경로 → 결정), L3(counterfactual decision), M2(결정 인코더), VLM 출력 파싱이 모두 이 형식을 쓴다.
"""

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

LON_ACTIONS = ("go", "stop")
LAT_ACTIONS = ("keep_lane", "turn_left", "turn_right", "lane_change_left", "lane_change_right", "offset_left",
               "offset_right")


@dataclass
class Segment:
    lon: str  # LON_ACTIONS
    lon_strength: float  # 0~1
    lat: str  # LAT_ACTIONS
    lat_strength: float  # 0~1 (keep_lane은 0)
    ambiguous_lon: bool = False  # 기준값 경계에 걸쳐 라벨이 흔들릴 수 있음
    ambiguous_lat: bool = False
    alt_lon: str = ""  # 앞뒤 행동 값이 기준값 근처이면, 기준선 너머의 이웃 라벨 (d̂ ≈ d 판정 때 허용)
    alt_lat: str = ""  # 좌우 행동 값이 기준값(turn, offset) 근처이면, 기준선 너머의 이웃 라벨
    structural: bool = False  # 구조적 애매함 (차선 경계 근처, 차선 밖 방향 등). 학습과 평가에서 뺀다

    def __post_init__(self):
        assert self.lon in LON_ACTIONS, self.lon
        assert self.lat in LAT_ACTIONS, self.lat


@dataclass
class Decision:
    segments: List[Segment]
    valid: bool = True  # False: 출발 차선을 찾지 못하는 등 결정을 정할 수 없음 (학습과 평가에서 뺀다)
    reason: str = ""  # valid=False 또는 애매한 이유
    extra: Dict[str, float] = field(default_factory=dict)  # 디버그용 값

    @property
    def ambiguous(self) -> bool:
        return (not self.valid) or any(s.ambiguous_lon or s.ambiguous_lat for s in self.segments)

    @property
    def usable(self) -> bool:
        """학습과 평가에 쓸 수 있는가: valid이고 구조적 애매함이 없다 (4단계 결정)."""
        return self.valid and not any(s.structural for s in self.segments)

    def to_flat(self) -> Dict[str, object]:
        """표(parquet) 한 줄로 바꾼다. 열 이름: seg{k}_lon, seg{k}_lon_strength, ..."""
        row: Dict[str, object] = {"valid": self.valid, "reason": self.reason, "ambiguous": self.ambiguous}
        for k, s in enumerate(self.segments, 1):
            for key, val in asdict(s).items():
                row[f"seg{k}_{key}"] = val
        return row

    def __str__(self) -> str:
        if not self.valid:
            return f"invalid ({self.reason})"
        parts = []
        for k, s in enumerate(self.segments, 1):
            a = "?" if (s.ambiguous_lon or s.ambiguous_lat) else ""
            lat = s.lat if s.lat == "keep_lane" else f"{s.lat} ({s.lat_strength:.2f})"
            parts.append(f"[{k}] {s.lon} ({s.lon_strength:.2f}), {lat}{a}")
        return "  ".join(parts)


def _match(a: str, a_alt: str, b: str, b_alt: str, allow_neighbor: bool) -> Tuple[bool, bool]:
    """(일치하는가, 라벨이 정확히 같은가)."""
    if a == b:
        return True, True
    if allow_neighbor and ((a_alt and a_alt == b) or (b_alt and b_alt == a)):
        return True, False
    return False, False


def same_decision(d: Decision, d_hat: Decision, strength_tol: float = 0.2,
                  allow_neighbor: bool = True) -> Optional[bool]:
    """d̂ ≈ d 판정 (yesman.md 6.2, 14절, 4단계 결정).

    구간마다 행동 종류가 같고 세기 차이가 strength_tol 이내이면 같다고 본다. keep_lane의 세기는 비교하지 않는다.
    allow_neighbor이면, 한쪽 라벨의 값이 기준값 근처여서 기준선 너머의 이웃 라벨(alt_lon, alt_lat)이 다른 쪽 라벨과
    같을 때에도 그 행동은 일치로 본다. 이때 세기는 비교하지 않는다 (기준선 근처라 세기도 경계값이다).
    둘 중 하나가 invalid이면 None을 낸다.
    """
    if not (d.valid and d_hat.valid) or len(d.segments) != len(d_hat.segments):
        return None
    for a, b in zip(d.segments, d_hat.segments):
        ok, exact = _match(a.lon, a.alt_lon, b.lon, b.alt_lon, allow_neighbor)
        if not ok or (exact and abs(a.lon_strength - b.lon_strength) > strength_tol):
            return False
        ok, exact = _match(a.lat, a.alt_lat, b.lat, b.alt_lat, allow_neighbor)
        if not ok or (exact and a.lat != "keep_lane" and abs(a.lat_strength - b.lat_strength) > strength_tol):
            return False
    return True
