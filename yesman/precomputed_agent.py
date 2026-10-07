"""미리 계산한 경로를 NAVSIM 공식 채점 스크립트(run_pdm_score_one_stage.py)에 넘기는 agent.

pred_path: scripts/predict_planner.py가 만든 D0 예측(token, pred_poses). None이면 모든 장면에 LTF가 예측한 경로
(7단계 특징의 ltf_traj)를 쓴다(인코더 자체의 점수, 참고용).
D0에 없는 navtest 장면(구조적 애매함으로 뺀 395개)에는 모든 모델에 같은 LTF 경로를 채운다. 이 장면들은 평균에서
빼므로(scripts/score_planner.py), 영향은 바로 다음 프레임의 EC(two-frame extended comfort) 계산뿐이고 모든 모델에 같다.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory

FEAT = Path(os.environ["YESMAN_ROOT"]) / "exp/features/ltf/navtest"


class PrecomputedAgent(AbstractAgent):
    requires_scene = True

    def __init__(self, pred_path: Optional[str] = None,
                 trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)):
        super().__init__(trajectory_sampling, requires_scene=True)
        self._pred_path = pred_path
        self._poses = None

    def name(self) -> str:
        return "PrecomputedAgent"

    def initialize(self) -> None:
        tokens = FEAT.joinpath("tokens.txt").read_text().split()
        ltf = np.load(FEAT / "ltf_traj.npy")
        self._poses = {t: ltf[i] for i, t in enumerate(tokens)}
        if self._pred_path:
            df = pd.read_parquet(self._pred_path, columns=["token", "pred_poses"])
            assert df.token.is_unique
            for t, p in zip(df.token, df.pred_poses):
                self._poses[t] = np.asarray(p, dtype=np.float32).reshape(8, 3)

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
        if self._poses is None:
            self.initialize()
        return Trajectory(self._poses[scene.scene_metadata.initial_token], self._trajectory_sampling)
