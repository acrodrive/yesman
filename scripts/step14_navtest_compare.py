"""14단계 최종 모델과 13단계 최종 모델을 navtest(개발 세트)에서 같은 지표로 나란히 놓는다. 탐색 분석이다.

지표 (모델마다 시드 0·1 평균)
  follow_D1     D1 결정–경로 일치율: ACCEPT이고 d̂ ≈ d (판단 모델은 τ = flag_threshold.json)
  over_reject   과잉 거부: D0 + D1에서 REJECT한 비율 (판단 모델만)
  safe_D2       D2 주 결과(보이지 않는 원인 제외)에서 그린 경로가 NC·DAC·DDC를 모두 통과한 비율, 불가능 기준별도
  safe_<VLM>    D3x_<VLM>의 실행 불가능한 결정(사람 결정과 같은 것 제외, 12단계와 같은 정의)에서의 같은 비율
  EPDMS_D0      D0 공식 채점 (score_planner.py)
13단계 모델의 D3x 채점은 exp/step12/d3x_infeasible/ (실행 불가능한 결정만 채점한 파일)을 쓴다.

사용법: python scripts/step14_navtest_compare.py [--out docs/step14_navtest.md]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ["YESMAN_ROOT"])
E = Path(os.environ["NAVSIM_EXP_ROOT"]) / "eval"
SAFE = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance"]
VLMS = {"gemma31b": "Gemma 4 31B", "gemma12b": "Gemma 4 12B", "qwen8b": "Qwen3-VL 8B"}
STEP13 = [("B0", "B1"), ("B1", "B2"), ("B2", "Bcomply_syn"), ("B3", "B3_syn"), ("이전 Ours", "Ours_syn")]


def step14_models():
    rows = [l.split() for l in (ROOT / "exp/step14/select/final_models.txt").read_text().splitlines() if l.strip()]
    name = {"B0": "B0", "B1": "B1", "B2": "B2", "B3": "B3", "OursPrev": "이전 Ours", "Ours": "Ours"}
    return [(name[k], v) for k, v in rows if k in name]


def accept(run, f):
    th = E / run / "flag_threshold.json"
    if "p" in f and f.p.notna().any():
        return f.p.to_numpy() >= json.load(open(th))["tau"]
    return np.ones(len(f), bool)


def metrics(run, step13):
    m = {}
    d0, d1 = pd.read_parquet(E / run / "D0_l1.parquet"), pd.read_parquet(E / run / "D1_l1.parquet")
    a0, a1 = accept(run, d0), accept(run, d1)
    m["follow_D1"] = float((a1 & d1.follow.to_numpy(bool)).mean())
    m["over_reject"] = float(np.concatenate([~a0, ~a1]).mean())
    D2 = pd.read_parquet(ROOT / "data_lists/eval/D2.parquet", columns=["category", "visibility"])
    sc = pd.read_parquet(E / run / "D2_scores.parquet")
    safe = (sc[SAFE].min(axis=1) >= 1).to_numpy()
    main = (D2.visibility != "unseen").to_numpy()
    m["safe_D2"] = float(safe[main].mean())
    for c in ("road", "agent", "strength"):
        m[f"safe_D2_{c}"] = float(safe[main & (D2.category == c).to_numpy()].mean())
    for v in VLMS:
        t = pd.read_parquet(ROOT / f"data_lists/eval/D3x_{v}.parquet", columns=["token", "status", "same_as_human", "parse_status"])
        t = t[t.parse_status == "ok"].reset_index(drop=True)
        inf = ((t.status == "infeasible") & ~t.same_as_human.fillna(False)).to_numpy()
        if step13:
            s = pd.read_parquet(Path(os.environ["NAVSIM_EXP_ROOT"]) / f"step12/d3x_infeasible/{run}_{v}_scores.parquet")
        else:
            s = pd.read_parquet(E / run / f"D3x_{v}_scores.parquet")
            assert len(s) == len(t), (run, v)
            s = s[inf]
        m[f"safe_{v}"] = float((s[SAFE].min(axis=1) >= 1).mean())
        m[f"n_{v}"] = int(inf.sum())
    j = E / run / "D0_score.json"
    m["EPDMS_D0"] = json.load(open(j))["EPDMS"] if j.exists() else np.nan
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = []
    for step, models, step13 in (("13단계", STEP13, True), ("14단계", step14_models(), False)):
        for name, prefix in models:
            ms = [metrics(f"{prefix}_seed{s}", step13) for s in (0, 1)]
            row = {"단계": step, "모델": name, "실행": prefix}
            row.update({k: float(np.mean([m[k] for m in ms])) for k in ms[0]})
            rows.append(row)
    t = pd.DataFrame(rows)
    cols = ["단계", "모델", "follow_D1", "over_reject", "safe_D2", "safe_D2_road", "safe_D2_agent", "safe_D2_strength",
            "safe_gemma31b", "safe_gemma12b", "safe_qwen8b", "EPDMS_D0"]
    fmt = lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)  # noqa: E731
    text = "\n".join(["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)] +
                     ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for _, r in t.iterrows()])
    n = {v: int(t[f"n_{v}"].iloc[0]) for v in VLMS}
    text = ("# 14단계 모델 navtest(개발 세트) 비교 (탐색, 시드 0·1 평균)\n\n" + text +
            f"\n\nD3x 실행 불가능한 결정 수: {n}\n")
    print(text)
    if args.out:
        args.out.write_text(text)
    t.to_csv(Path(os.environ["NAVSIM_EXP_ROOT"]) / "step14/navtest_compare.csv", index=False)


if __name__ == "__main__":
    main()
