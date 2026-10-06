"""L2 경로 모음 + 실행 가능 판정 (yesman.md 7.2, 5단계 사전 결정은 docs/step05_l2.md).

경로 모음: navtrain의 모든 사람 경로(ego 좌표계 8x3)와 출발 속도, 원래 장면에서의 L1 라벨 (scripts/l2_build_bank.py).

판정 (장면, 결정 d):
1. 후보 고르기: 출발 속도가 ±2 m/s 안이고, 판정할 장면과 다른 로그이며, 원래 장면에서의 L1 라벨이 d와 맞는(d̂ ≈ d)
   경로 중에서 무작위로 최대 k_relabel개를 고른다.
2. 다시 라벨 붙이기: 고른 경로를 판정할 장면에 그대로 옮겨 놓고(ego 좌표계) L1을 다시 돌린다. 이 장면에서도 d̂ ≈ d이고
   구조적 애매함이 없는 경로만 후보로 남긴다. 후보가 n_min개 미만이면 "판정 불가"이다.
3. 채점: 후보 중 무작위로 최대 k_score개를 NAVSIM v2 채점 도구로 한 번에 채점한다(human filter 적용).
   NC = DAC = DDC = 1인 후보가 하나라도 있으면 "가능", 없으면 "불가능"이다.
4. 출력: 가능이면 통과한 후보 중 점수(EC를 뺀 EPDMS)가 가장 높은 경로. 불가능이면 판단 근거 = 떨어진 후보 중
   점수가 가장 높았던 후보가 떨어진 항목. 점수는 곱하는 항목을 뺀 가중 평균 점수에서 떨어진 항목 수가 적은 순으로 고른다.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from navsim.planning.metric_caching.metric_cache import MetricCache
from yesman.decision import Decision, Segment, same_decision
from yesman.l1 import classify, extract_features, load_thresholds
from yesman.scoring import BatchScorer, feasibility

EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
BANK_PATH = EXP / "l2/path_bank.parquet"
FEAS_ITEMS = ("no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance")
REASON_NAME = {"no_at_fault_collisions": "collision", "drivable_area_compliance": "off_road",
               "driving_direction_compliance": "wrong_way"}


@dataclass
class L2Config:
    speed_tol: float = 2.0  # 출발 속도 차이 [m/s]
    n_min: int = 5  # 후보가 이보다 적으면 판정 불가
    k_relabel: int = 200  # 다시 라벨을 붙일 최대 경로 수
    k_score: int = 50  # 채점할 최대 후보 수
    seed: int = 0


@dataclass
class L2Result:
    status: str  # feasible / infeasible / undetermined / invalid
    reason: str = ""  # 불가능의 판단 근거: collision / off_road / wrong_way (여럿이면 '+'로 잇는다)
    n_prefilter: int = 0  # 속도, 로그, 원래 라벨 조건을 만족한 경로 수
    n_relabeled: int = 0  # 다시 라벨을 붙인 경로 수
    n_match: int = 0  # 이 장면에서도 d̂ ≈ d인 경로 수 (후보)
    n_scored: int = 0
    n_pass: int = 0
    fail_frac: Dict[str, float] = field(default_factory=dict)  # 채점한 후보 중 항목별로 떨어진 비율
    best_bank_idx: int = -1  # 가능: 통과한 최고 후보, 불가능: 떨어진 후보 중 최고 (경로 모음의 행 번호)
    best_score: float = float("nan")
    best_poses: Optional[np.ndarray] = None
    scored_bank_idx: List[int] = field(default_factory=list)
    scored_pass: List[bool] = field(default_factory=list)

    def to_flat(self) -> Dict[str, object]:
        row = {k: v for k, v in self.__dict__.items() if k not in ("fail_frac", "best_poses", "scored_bank_idx",
                                                                   "scored_pass")}
        for m, v in self.fail_frac.items():
            row[f"fail_{REASON_NAME.get(m, m)}"] = v
        return row


class PathBank:
    """경로 모음. 라벨 열은 Decision으로 바꿔 둔다."""

    def __init__(self, path: Path = BANK_PATH):
        df = pd.read_parquet(path)
        self.df = df.reset_index(drop=True)
        self.poses = np.stack(df.poses.to_numpy()).reshape(-1, 8, 3).astype(np.float64)
        self.v0 = df.v0.to_numpy()
        self.log = df.log_name.to_numpy()
        self.decisions = [row_to_decision(r) for r in df.to_dict("records")]
        # 빠른 1차 거르기용: 구간별 (lon, lat, alt_lon, alt_lat)
        self._lab = {c: df[c].to_numpy() for c in df.columns if c.startswith("seg")}

    def __len__(self):
        return len(self.df)

    def prefilter(self, d: Decision, v0: float, exclude_log: str, speed_tol: float) -> np.ndarray:
        """속도, 로그, 원래 라벨(행동 종류만, 이웃 라벨 허용) 조건을 만족하는 행 번호."""
        m = (np.abs(self.v0 - v0) <= speed_tol) & (self.log != exclude_log)
        for k, s in enumerate(d.segments, 1):
            for kind in ("lon", "lat"):
                lab, alt = self._lab[f"seg{k}_{kind}"], self._lab[f"seg{k}_alt_{kind}"]
                want, want_alt = getattr(s, kind), getattr(s, f"alt_{kind}")
                ok = (lab == want) | (alt == want)
                if want_alt:
                    ok |= lab == want_alt
                m &= ok
        return np.flatnonzero(m)


def row_to_decision(r: Dict) -> Decision:
    """라벨 표(parquet) 한 줄을 Decision으로 바꾼다 (Decision.to_flat의 반대)."""
    if not r.get("valid", True):
        return Decision([], valid=False, reason=str(r.get("reason", "")))
    segs = []
    k = 1
    while f"seg{k}_lon" in r:
        segs.append(Segment(r[f"seg{k}_lon"], float(r[f"seg{k}_lon_strength"]), r[f"seg{k}_lat"],
                            float(r[f"seg{k}_lat_strength"]), bool(r[f"seg{k}_ambiguous_lon"]),
                            bool(r[f"seg{k}_ambiguous_lat"]), r.get(f"seg{k}_alt_lon", "") or "",
                            r.get(f"seg{k}_alt_lat", "") or "", bool(r.get(f"seg{k}_structural", False))))
        k += 1
    return Decision(segs, valid=True, reason=str(r.get("reason", "")))


@dataclass
class SceneContext:
    """판정할 장면 하나의 정보 (결정마다 다시 계산하지 않는다)."""
    token: str
    log_name: str
    v0: float
    ego_pose: np.ndarray
    map_api: object
    metric_cache: MetricCache
    human: Dict[str, float]  # human filter용 사람 경로 채점 결과
    relabel_cache: Dict[int, Decision] = field(default_factory=dict)  # 경로 모음 행 번호 → 이 장면에서의 L1


class L2:
    def __init__(self, bank: PathBank, cfg: L2Config = L2Config(), scorer: Optional[BatchScorer] = None):
        self.bank, self.cfg = bank, cfg
        self.scorer = scorer or BatchScorer(human_penalty_filter=True)
        self.th = load_thresholds()

    def context(self, scene, metric_cache: MetricCache) -> SceneContext:
        cur = scene.scene_metadata.num_history_frames - 1
        st = scene.frames[cur].ego_status
        return SceneContext(scene.scene_metadata.initial_token, scene.scene_metadata.log_name,
                            float(np.linalg.norm(st.ego_velocity[:2])), np.asarray(st.ego_pose), scene.map_api,
                            metric_cache, self.scorer.human_metrics(metric_cache))

    def relabel(self, ctx: SceneContext, idx: int) -> Decision:
        if idx not in ctx.relabel_cache:
            f = extract_features(self.bank.poses[idx], ctx.v0, ctx.ego_pose, ctx.map_api)
            ctx.relabel_cache[idx] = classify(f, self.th)
        return ctx.relabel_cache[idx]

    def judge(self, ctx: SceneContext, d: Decision, rng: Optional[np.random.Generator] = None) -> L2Result:
        if not d.usable:
            return L2Result("invalid", reason="decision_not_usable")
        rng = rng or np.random.default_rng(self.cfg.seed)
        pre = self.bank.prefilter(d, ctx.v0, ctx.log_name, self.cfg.speed_tol)
        res = L2Result("undetermined", n_prefilter=len(pre))
        if len(pre) > self.cfg.k_relabel:
            pre = rng.choice(pre, size=self.cfg.k_relabel, replace=False)
        match = []
        for idx in pre:
            dh = self.relabel(ctx, int(idx))
            if dh.usable and same_decision(d, dh):
                match.append(int(idx))
        res.n_relabeled, res.n_match = len(pre), len(match)
        if len(match) < self.cfg.n_min:
            return res
        cand = list(rng.choice(match, size=min(self.cfg.k_score, len(match)), replace=False))
        df = self.scorer.score(ctx.metric_cache, [self.bank.poses[i] for i in cand], ctx.human)
        ok = feasibility(df, FEAS_ITEMS)
        res.n_scored, res.n_pass = len(cand), int(ok.sum())
        res.scored_bank_idx, res.scored_pass = [int(i) for i in cand], [bool(x) for x in ok]
        res.fail_frac = {m: float((df[m] < 1).mean()) for m in FEAS_ITEMS}
        # EC를 뺀 EPDMS: 곱하는 항목(NC, DAC, DDC, TLC) x 가중 평균(EP, TTC, LK, HC)
        w = {"ego_progress": 5, "time_to_collision_within_bound": 5, "lane_keeping": 2, "history_comfort": 2}
        weighted = sum(df[m] * v for m, v in w.items()) / sum(w.values())
        mult = df[list(FEAS_ITEMS) + ["traffic_light_compliance"]].prod(axis=1)
        score = (mult * weighted).to_numpy()
        if ok.any():
            best = int(np.argmax(np.where(ok, score, -np.inf)))
            res.status = "feasible"
        else:
            n_fail = sum((df[m] < 1).to_numpy().astype(int) for m in FEAS_ITEMS)
            best = int(np.lexsort((-weighted.to_numpy(), n_fail))[0])  # 떨어진 항목 수가 적고 가중 점수가 높은 후보
            res.status = "infeasible"
            res.reason = "+".join(REASON_NAME[m] for m in FEAS_ITEMS if df[m].iloc[best] < 1)
        res.best_bank_idx, res.best_score = cand[best], float(score[best])
        res.best_poses = self.bank.poses[cand[best]]
        return res
