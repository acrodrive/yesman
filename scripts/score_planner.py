"""D0 주행 점수: 미리 계산한 경로를 NAVSIM v2 공식 채점(run_pdm_score_one_stage.py, navtest, non-reactive)으로
채점하고, D0 장면(11,751개)만 평균한다 (9단계, 11단계).

사용법:
  python scripts/score_planner.py --run B2_seed0          # exp/eval/B2_seed0/D0.parquet 채점
  python scripts/score_planner.py --run LTF               # LTF 자체 경로 (참고)
  python scripts/score_planner.py --summarize_only --run B2_seed0
출력: exp/eval/<run>/score/<시각>/<시각>.csv (공식 장면별 CSV), exp/eval/<run>/D0_score.json (D0 평균)
사람, 등속 경로의 D0 평균은 3단계 CSV에서 같은 방법으로 계산한다(--reference).
"""

import argparse
import glob
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(os.environ["YESMAN_ROOT"])
EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
DEVKIT = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
ITEMS = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
         "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound", "lane_keeping",
         "history_comfort", "two_frame_extended_comfort", "score"]
SHORT = {"no_at_fault_collisions": "NC", "drivable_area_compliance": "DAC", "driving_direction_compliance": "DDC",
         "traffic_light_compliance": "TLC", "ego_progress": "EP", "time_to_collision_within_bound": "TTC",
         "lane_keeping": "LK", "history_comfort": "HC", "two_frame_extended_comfort": "EC", "score": "EPDMS"}


def d0_mean(csv: Path) -> dict:
    df = pd.read_csv(csv)
    df = df[df.token != "average_all_frames"]
    d0 = set(pd.read_parquet(ROOT / "data_lists/eval/D0.parquet", columns=["token"]).token)
    sub = df[df.token.isin(d0)]
    out = {"n": int(len(sub)), "n_invalid": int((~sub.valid.astype(bool)).sum()), "csv": str(csv)}
    for k in ITEMS:
        out[SHORT[k]] = float(sub[k].mean())  # EC가 없는 장면(NaN)은 EC 평균에서 빠진다. score에는 이미 반영됨
    return out


def run_official(run: str, threads: int) -> Path:
    pred = None if run == "LTF" else ROOT / f"exp/eval/{run}/D0.parquet"
    exp_name = f"eval/{run}/score"  # NAVSIM_EXP_ROOT(= exp/) 기준
    cmd = ["python", str(DEVKIT / "navsim/planning/script/run_pdm_score_one_stage.py"), "train_test_split=navtest",
           "agent=human_agent", "agent._target_=yesman.precomputed_agent.PrecomputedAgent",
           f"+agent.pred_path={pred if pred else 'null'}", f"experiment_name={exp_name}",
           "traffic_agents=non_reactive", f"metric_cache_path={EXP / 'metric_cache/navtest'}",
           # ray 워커는 이 GPU pod에서 시작 뒤 멈춘다(7~9단계). 한 기계의 process pool을 쓴다
           "worker=single_machine_thread_pool", "worker.use_process_pool=true", f"worker.max_workers={threads}"]
    subprocess.run(cmd, check=True)
    return latest_csv(run)


def latest_csv(run: str) -> Path:
    csvs = sorted(glob.glob(str(EXP / f"eval/{run}/score/*/*.csv")))
    assert csvs, f"채점 CSV가 없다: {run}"
    return Path(csvs[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--threads", type=int, default=30)
    ap.add_argument("--summarize_only", action="store_true")
    ap.add_argument("--reference", action="store_true", help="3단계 사람, 등속 CSV의 D0 평균")
    args = ap.parse_args()
    if args.reference:
        for agent in ("human_agent", "constant_velocity_agent"):
            csv = Path(sorted(glob.glob(str(EXP / f"step03/{agent}/*/*.csv")))[-1])
            res = d0_mean(csv)
            (EXP / f"eval/ref_{agent}_D0_score.json").parent.mkdir(parents=True, exist_ok=True)
            json.dump(res, open(EXP / f"eval/ref_{agent}_D0_score.json", "w"), indent=1)
            print(agent, {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items() if k != "csv"})
        return
    csv = latest_csv(args.run) if args.summarize_only else run_official(args.run, args.threads)
    res = d0_mean(csv)
    out = EXP / f"eval/{args.run}/D0_score.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(args.run, {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items() if k != "csv"})


if __name__ == "__main__":
    main()
