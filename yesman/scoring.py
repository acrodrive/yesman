"""NAVSIM v2 채점 도구로 경로 여러 개를 한 장면에서 한 번에 채점한다 (5단계 L2용).

공식 navsim.evaluate.pdm_score.pdm_score()는 경로 하나마다 [PDM-Closed 기준 경로, 평가 경로] 두 개를 시뮬레이션하고 채점한다.
여기서는 [PDM-Closed 기준 경로, 후보 1..N]을 한 번에 시뮬레이션하고 채점한다. 채점 항목은 경로마다 독립이고
(EP의 정규화만 다른 경로에 따라 바뀐다), non-reactive 교통은 ego 경로와 무관하게 기록된 움직임을 쓴다.
따라서 NC, DAC, DDC, TLC, TTC, LK, HC는 공식 함수와 같아야 한다 (scripts/l2_check_batch_scoring.py로 확인).

human penalty filter: 공식 구현과 같이, 사람 경로가 0점을 받은 항목은 후보에서도 1로 바꾼다 (원래 장면만).
"""

import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from navsim.common.dataclasses import Trajectory
from navsim.common.enums import SceneFrameType
from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.metric_caching.metric_cache import MetricCache

DEVKIT = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
METRICS = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
           "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound", "lane_keeping",
           "history_comfort"]
SHORT = {"no_at_fault_collisions": "NC", "drivable_area_compliance": "DAC", "driving_direction_compliance": "DDC",
         "traffic_light_compliance": "TLC", "ego_progress": "EP", "time_to_collision_within_bound": "TTC",
         "lane_keeping": "LK", "history_comfort": "HC"}


class BatchScorer:
    """공식 navtest 채점과 같은 설정(default_run_pdm_score.yaml, non-reactive)으로 만든 simulator, scorer, 교통 정책."""

    def __init__(self, human_penalty_filter: bool = True):
        with initialize_config_dir(config_dir=str(DEVKIT / "navsim/planning/script/config/pdm_scoring"),
                                   version_base=None):
            cfg = compose(config_name="default_run_pdm_score", overrides=[
                "train_test_split=navtest", "experiment_name=batch_scorer", "traffic_agents=non_reactive",
                "metric_cache_path=/tmp"])
        self.simulator = instantiate(cfg.simulator)
        self.scorer = instantiate(cfg.scorer)
        self.traffic = instantiate(cfg.traffic_agents_policy.non_reactive, self.simulator.proposal_sampling)
        self.human_penalty_filter = human_penalty_filter
        self.sampling = self.simulator.proposal_sampling

    def _states(self, poses: np.ndarray, mc: MetricCache) -> np.ndarray:
        traj = transform_trajectory(Trajectory(np.asarray(poses, dtype=np.float32)), mc.ego_state)
        return get_trajectory_as_array(traj, self.sampling, mc.ego_state.time_point)

    def _score(self, states: np.ndarray, mc: MetricCache, with_history: bool) -> List[pd.DataFrame]:
        simulated = self.simulator.simulate_proposals(states, mc.ego_state)
        tracks = self.traffic.simulate_environment(simulated[0], mc)
        return self.scorer.score_proposals(simulated, mc.observation, mc.centerline, mc.route_lane_ids,
                                           mc.drivable_area_map, mc.map_parameters, tracks,
                                           mc.past_human_trajectory if with_history else None)

    def human_metrics(self, mc: MetricCache) -> Dict[str, float]:
        """사람 경로만 따로 채점한 결과 (공식 human filter와 같은 방식: 과거 경로 없이)."""
        states = self._states(mc.human_trajectory.poses, mc)[None]
        res = self._score(states, mc, with_history=False)[0]
        return {m: float(res[m].iloc[0]) for m in METRICS}

    def score(self, mc: MetricCache, poses_list: Sequence[np.ndarray],
              human: Dict[str, float] = None) -> pd.DataFrame:
        """경로 N개(각 (8, 3), ego 좌표계)를 채점한다. 행 i = 경로 i. human filter를 적용한 값과 원래 값(raw_*)을 낸다."""
        pdm_states = get_trajectory_as_array(mc.trajectory, self.sampling, mc.ego_state.time_point)
        states = np.stack([pdm_states] + [self._states(p, mc) for p in poses_list])
        res = self._score(states, mc, with_history=True)[1:]
        df = pd.DataFrame([{m: float(r[m].iloc[0]) for m in METRICS} for r in res])
        for m in METRICS:
            df[f"raw_{m}"] = df[m]
        if self.human_penalty_filter and mc.scene_type == SceneFrameType.ORIGINAL:
            human = human if human is not None else self.human_metrics(mc)
            for m in METRICS:
                if human[m] == 0:
                    df[m] = 1.0
        return df


def feasibility(df: pd.DataFrame, items: Tuple[str, ...] = ("no_at_fault_collisions", "drivable_area_compliance",
                                                             "driving_direction_compliance")) -> np.ndarray:
    """L2의 통과 조건 (5단계 사전 결정): NC = DAC = DDC = 1 (human filter 적용 후)."""
    return np.all(np.stack([df[m].to_numpy() >= 1.0 for m in items]), axis=0)
