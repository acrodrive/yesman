"""3단계: NAVSIM v2 채점 도구를 직접 불러 경로를 채점하고, 공식 채점 결과(run_score_navtest.sh의 CSV)와 맞는지 확인한다.

5단계의 L2는 장면마다 후보 경로 여러 개를 채점해야 하므로 공식 스크립트(agent 단위) 대신 pdm_score()를 직접 부른다.
이 스크립트는 그 경로가 공식 결과와 같은 점수를 내는지 navtest 장면 일부에서 확인한다.
- 설정(simulator, scorer, 교통 정책)은 공식 설정 파일(default_run_pdm_score.yaml)을 hydra compose로 그대로 읽는다.
- 채점 대상: 사람 경로(HumanAgent와 같은 scene.get_future_trajectory)와 등속 경로(ConstantVelocityAgent).
- 장면별 항목(NC, DAC, DDC, TLC, EP, TTC, LK, HC)을 공식 CSV와 비교한다.
  EC(two-frame extended comfort)는 이웃 프레임의 결과가 있어야 계산되므로 공식 CSV의 집계 단계에서만 나온다.
- 같은 장면에서 metric cache의 ego 뒤차축(채점의 원점)이 로그의 현재 프레임 ego pose와 같은지도 확인한다.
- human penalty filter(사람도 0점인 항목은 1로 바꿈)를 끈 채점도 사람 경로에 한 번 더 하여, 사람 경로가 실제로
  각 항목을 위반하는 비율을 출력한다. filter를 켜면 사람 경로는 NC, DAC 등이 항상 1이 되기 때문이다 (5단계 L2 참고용).

사용법 (source scripts/setup/env.sh 후):
    python scripts/score_trajectories.py --n 200 \
        --human_csv exp/step03/human_agent/<시각>/<시각>.csv --cv_csv exp/step03/constant_velocity_agent/<시각>/<시각>.csv
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from navsim.agents.constant_velocity_agent import ConstantVelocityAgent
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.evaluate.pdm_score import pdm_score

DATA_ROOT = Path(os.environ["OPENSCENE_DATA_ROOT"])
DEVKIT = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
METRICS = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
           "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound", "lane_keeping",
           "history_comfort"]


def build_scoring(metric_cache_path: Path):
    """공식 navtest 채점과 같은 설정으로 simulator, scorer, 교통 정책(non-reactive)을 만든다."""
    with initialize_config_dir(config_dir=str(DEVKIT / "navsim/planning/script/config/pdm_scoring"), version_base=None):
        cfg = compose(config_name="default_run_pdm_score", overrides=[
            "train_test_split=navtest", "experiment_name=score_trajectories", "traffic_agents=non_reactive",
            f"metric_cache_path={metric_cache_path}"])
    simulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    cfg.scorer.config.human_penalty_filter = False
    scorer_nofilter = instantiate(cfg.scorer)
    traffic = instantiate(cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling)
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    return cfg, simulator, scorer, scorer_nofilter, traffic, scene_filter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric_cache", type=Path, default=EXP / "metric_cache/navtest")
    ap.add_argument("--n", type=int, default=200, help="확인할 장면 수 (무작위)")
    ap.add_argument("--human_csv", type=Path, required=True)
    ap.add_argument("--cv_csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=EXP / "step03/score_trajectories.csv")
    args = ap.parse_args()

    cfg, simulator, scorer, scorer_nofilter, traffic, scene_filter = build_scoring(args.metric_cache)
    official = {name: pd.read_csv(p).set_index("token") for name, p in [("human", args.human_csv), ("cv", args.cv_csv)]}
    cache_loader = MetricCacheLoader(args.metric_cache)
    tokens = sorted(set(cache_loader.tokens) & set(official["human"].index) & set(official["cv"].index))
    tokens = list(np.random.default_rng(0).choice(tokens, size=min(args.n, len(tokens)), replace=False))

    # 고른 장면이 들어 있는 로그만 읽는다
    log_of = {t: cache_loader.get_from_token(t).log_name for t in tokens}
    scene_filter.log_names = sorted(set(log_of.values()))
    scene_filter.tokens = tokens
    loader = SceneLoader(DATA_ROOT / "navsim_logs/test", None, scene_filter, sensor_config=SensorConfig.build_no_sensors())
    cv_agent = ConstantVelocityAgent()

    rows, origin_err = [], []
    for i, t in enumerate(tokens):
        mc = cache_loader.get_from_token(t)
        scene = loader.get_scene_from_token(t)
        cur = scene.scene_metadata.num_history_frames - 1
        ra = mc.ego_state.rear_axle
        pose = scene.frames[cur].ego_status.ego_pose
        origin_err.append(max(abs(ra.x - pose[0]), abs(ra.y - pose[1]),
                              abs(np.angle(np.exp(1j * (ra.heading - pose[2]))))))
        trajs = {"human": scene.get_future_trajectory(8), "cv": cv_agent.compute_trajectory(scene.get_agent_input())}
        for name, traj in trajs.items():
            res, _ = pdm_score(mc, traj, simulator.proposal_sampling, simulator, scorer, traffic)
            row = {"token": t, "traj": name, **{m: float(res[m].iloc[0]) for m in METRICS}}
            off = official[name].loc[t]
            row.update({f"official_{m}": float(off[m]) for m in METRICS})
            rows.append(row)
        raw, _ = pdm_score(mc, trajs["human"], simulator.proposal_sampling, simulator, scorer_nofilter, traffic)
        rows.append({"token": t, "traj": "human_nofilter", **{m: float(raw[m].iloc[0]) for m in METRICS}})
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(tokens)}", flush=True)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"scenes {len(tokens)}, saved {args.out}")
    print(f"metric cache rear axle vs log ego pose (current frame): max |err| = {max(origin_err):.2e}")
    for name in ["human", "cv"]:
        d = df[df.traj == name]
        diffs = {m: float(np.nanmax(np.abs(d[m] - d[f"official_{m}"]))) for m in METRICS}
        print(f"[{name}] max |direct - official| per metric: " + ", ".join(f"{m}={v:.2e}" for m, v in diffs.items()))
    d = df[df.traj == "human_nofilter"]
    print("[human, filter off] share of scenes with metric < 1: "
          + ", ".join(f"{m}={(d[m] < 1).mean() * 100:.1f}%" for m in METRICS[:4] + METRICS[5:7]))


if __name__ == "__main__":
    main()
