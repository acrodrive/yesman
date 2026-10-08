"""12단계 (2) 추가: 실제 VLM의 '불가능' 결정을 받았을 때 planner 경로의 결과 안전성 (E1을 실제 VLM 결정에서).

D3x_*의 L2 불가능 행(사람과 같음 제외)만 골라, 각 planner의 경로를 행마다 채점한다(scripts/score_rows.py와 같은 채점).
안전 = NC, DAC, DDC 모두 통과. 불가능 기준(category)별 구성과 기준별 안전도 낸다.

사용법: python scripts/analyze_step12_d3safety.py
출력: exp/step12/d3x_infeasible_safety.json, logs/step12/d3x_safety.md
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analyze_step11 import tau, wilson  # noqa: E402

ROOT = Path(os.environ["YESMAN_ROOT"])
E = ROOT / "exp/eval"
RUNS = {"B1": "B1", "B2": "B2", "B-순응": "Bcomply_rt", "B3": "B3_rt", "Ours": "Ours_w025_rt"}
if "--synth" in sys.argv:  # 12단계 (3): CF⁺ 목표를 합성으로 바꾼 모델
    RUNS = {"B1": "B1", "B2": "B2", "B-순응": "Bcomply_syn", "B3": "B3_syn", "Ours": "Ours_syn"}
VLMS = {"Gemma 4 12B": "gemma12b", "Gemma 4 31B": "gemma31b", "Qwen3-VL 8B": "qwen8b"}
SAFE = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance"]


def main():
    out, L = {}, ["# 실제 VLM의 '불가능' 결정에서 planner 경로의 결과 안전성 (시드 평균, [95% CI]는 두 시드를 합친 Wilson)", ""]
    tmp = ROOT / "exp/step12/d3x_infeasible"
    tmp.mkdir(parents=True, exist_ok=True)
    for vname, v in VLMS.items():
        d = pd.read_parquet(ROOT / f"data_lists/eval/D3x_{v}.parquet")
        d = d[d.parse_status == "ok"].reset_index(drop=True)
        inf = ((d.status == "infeasible") & ~d.same_as_human.fillna(False)).to_numpy()
        cat = d.category[inf].fillna("").value_counts().to_dict()
        res = {"n": int(inf.sum()), "category": cat, "models": {}}
        for m, r in RUNS.items():
            safes, execs, rejs = [], [], []
            for s in (0, 1):
                run = f"{r}_seed{s}"
                pr = pd.read_parquet(E / f"{run}/D3x_{v}.parquet")
                assert (pr.token.values == d.token.values).all()
                sub = pr[inf].reset_index(drop=True)
                f_in = tmp / f"{run}_{v}.parquet"
                sub[["token", "pred_poses"]].to_parquet(f_in, index=False)
                f_out = tmp / f"{run}_{v}_scores.parquet"
                if not f_out.exists():
                    subprocess.run([sys.executable, str(ROOT / "scripts/score_rows.py"), "--table", str(f_in),
                                    "--out", str(f_out), "--workers", "16"], check=True, capture_output=True)
                sc = pd.read_parquet(f_out)
                safe = (sc[SAFE].min(axis=1) >= 1).to_numpy()
                l1 = pd.read_parquet(E / f"{run}/D3x_{v}_l1.parquet").follow.to_numpy()[inf]
                t = tau(run)
                acc = np.ones(len(sub), bool) if t is None else sub.p.to_numpy() >= t
                safes.append(safe), execs.append(acc & l1), rejs.append(~acc)
            s_all, e_all, r_all = np.concatenate(safes), np.concatenate(execs), np.concatenate(rejs)
            res["models"][m] = {"safe": wilson(int(s_all.sum()), len(s_all)), "executed": float(e_all.mean()),
                                "rejected": float(r_all.mean()),
                                "safe_by_cat": {c: float(np.concatenate([x[(d.category[inf] == c).to_numpy()] for x in safes]).mean())
                                                for c in ("road", "agent", "strength") if (d.category[inf] == c).any()}}
        out[vname] = res
        L += [f"## {vname} (불가능 결정 n = {res['n']}, 기준: {cat})", "",
              "| 모델 | 결과 안전 [95% CI] | 결정대로 실행 | 거부 | 안전: 기준별 |", "|---|---|---|---|---|"]
        for m, x in res["models"].items():
            L.append(f"| {m} | {x['safe'][0]:.1%} [{x['safe'][1]:.1%}, {x['safe'][2]:.1%}] | {x['executed']:.1%} | "
                     f"{x['rejected']:.1%} | " + ", ".join(f"{k} {vv:.0%}" for k, vv in x["safe_by_cat"].items()) + " |")
        L.append("")
    tag = "_synth" if "--synth" in sys.argv else ""
    json.dump(out, open(ROOT / f"exp/step12/d3x_infeasible_safety{tag}.json", "w"), indent=1, ensure_ascii=False, default=float)
    (ROOT / f"logs/step12/d3x_safety{tag}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
