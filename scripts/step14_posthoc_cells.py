"""14단계 사후 분석 1번 (docs/TODO.md): 세트별 판단 x 결정–경로 x 안전 표.

이미 한 번 평가한 결과를 다시 나눈다. 이 결과로 모델이나 τ를 바꾸지 않는다. 모델 6개 x 시드 0·1.
T0, T2, T3는 테스트 평가 때 채점한 결과를 쓴다. T1은 테스트 평가에서 채점하지 않았으므로 여기서 일부만 채점한다:
  무작위 1만 개(시드 0, 모든 모델에 같은 결정) + 판단 모델이 그 시드에서 거부한 T1 결정 전부.
  칸의 비율은 T1 전체로, 안전율은 ACCEPT 칸은 무작위 1만 개로, REJECT 칸은 전부로 낸다.

사용법
  python scripts/step14_posthoc_cells.py prepare   # exp/eval/<run>/posthoc/T1_sub.parquet (고른 행만 경로, 나머지 None)
  (채점) scripts/run_step14_posthoc_t1.sh
  python scripts/step14_posthoc_cells.py table     # docs/step14_posthoc_cells.md
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ["YESMAN_ROOT"])
EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
E = EXP / "eval"
MODELS = {"B0": "B1_s14", "B1": "B2_s14", "B2": "Bcomply_s14", "B3": "B3_s14_pl04_bs_r05",
          "OursPrev": "Ours_s14_pl04_r1", "Ours": "OursDiT_s14_pl04_r1_jl2"}  # exp/step14/select/final_models.txt
SEEDS = (0, 1)
N_SAMPLE = 10_000
SAMPLE = EXP / "step14/posthoc/T1_sample10k.npy"


def runs():
    return [f"{r}_seed{s}" for r in MODELS.values() for s in SEEDS]


def accept(run, f):
    th = E / run / "flag_threshold.json"
    if "p" in f and th.exists():
        return (f.p >= json.load(open(th))["tau"]).to_numpy()
    return np.ones(len(f), bool)


def prepare():
    n = len(pd.read_parquet(ROOT / "data_lists/eval_step14/T1.parquet", columns=["token"]))
    if SAMPLE.exists():
        idx = np.load(SAMPLE)
    else:
        idx = np.sort(np.random.default_rng(0).choice(n, N_SAMPLE, replace=False))
        SAMPLE.parent.mkdir(parents=True, exist_ok=True)
        np.save(SAMPLE, idx)
    for run in runs():
        pred = pd.read_parquet(E / run / "T1.parquet", columns=["token", "cf_name", "p", "pred_poses"]
                               if run.startswith("Ours") else ["token", "cf_name", "pred_poses"])
        assert len(pred) == n
        keep = np.zeros(n, bool)
        keep[idx] = True
        keep |= ~accept(run, pred)
        out = pred[["token", "cf_name"]].assign(pred_poses=pred.pred_poses.where(keep, None), in_sample=False)
        out.loc[idx, "in_sample"] = True
        d = E / run / "posthoc"
        d.mkdir(exist_ok=True)
        out.to_parquet(d / "T1_sub.parquet", index=False)
        print(run, f"표본 {len(idx):,} + 거부 {(~accept(run, pred)).sum():,} → 채점 {keep.sum():,}")


LABEL = {"B0": "B0 (결정 무시)", "B1": "B1", "B2": "B2 (+CF⁺)", "B3": "B3 (+CF⁻)", "OursPrev": "이전 Ours", "Ours": "Ours"}
SAFE = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance"]
FEAS_CELLS = [(True, True, "ACCEPT·일치 (바람직)"), (True, False, "ACCEPT·불일치 (실행 실패)"),
              (False, False, "REJECT·불일치 (과잉 거부)"), (False, True, "REJECT·일치")]
INF_CELLS = [(False, False, "REJECT·불일치 (바람직)"), (False, True, "REJECT·일치 (결정 누설)"),
             (True, True, "ACCEPT·일치 (맹목적 추종)"), (True, False, "ACCEPT·불일치 (조용히 피함)")]


def frame(run, s):
    """세트 s의 행마다 accept, follow, safe(채점한 행만, 나머지 NaN), use_safe(안전율 계산에 쓰는 행)."""
    f = pd.read_parquet(E / run / f"{s}_l1.parquet")
    out = pd.DataFrame({"accept": accept(run, f), "follow": f.follow.to_numpy()})
    if s == "T1":
        sub = pd.read_parquet(E / run / "posthoc/T1_sub.parquet", columns=["in_sample"])
        sc = pd.read_parquet(E / run / "posthoc/T1_sub_scores.parquet")
        # ACCEPT 칸은 무작위 1만 개로, REJECT 칸은 거부한 결정 전부로 (거부한 결정은 모두 채점했다)
        out["use_safe"] = np.where(out.accept, sub.in_sample.to_numpy(), True)
    else:
        sc = pd.read_parquet(E / run / f"{s}_scores.parquet")
        out["use_safe"] = True
    out["safe"] = np.where(sc[SAFE].isna().any(axis=1), np.nan, (sc[SAFE].min(axis=1) >= 1).astype(float))
    keep = {"T0": ("l2_status", lambda c: c == "feasible"), "T2": ("visibility", lambda c: c != "unseen"),
            "T3_gemma31b": ("status", lambda c: c == "infeasible")}
    if s in keep:
        col, fn = keep[s]
        out = out[fn(pd.read_parquet(ROOT / f"data_lists/eval_step14/{s}.parquet", columns=[col])[col]).to_numpy()]
    return out


def cell_table(s, cells):
    head = ["모델"] + [c for _, _, c in cells] + ["전체 안전율"]
    rows = []
    for k, r in MODELS.items():
        fs = [frame(f"{r}_seed{seed}", s) for seed in SEEDS]
        t = pd.concat(fs)
        row = [LABEL[k]]
        for acc, fol, _ in cells:
            g = t[(t.accept == acc) & (t.follow == fol)]
            share = len(g) / len(t)
            gs = g[g.use_safe & g.safe.notna()]
            if share == 0:
                row.append("—")
            else:
                n = len(gs) / len(SEEDS)
                safe = f"안전 {100 * gs.safe.mean():.1f}%" + (f", n = {n:,.0f}" if n < 100 else "") if len(gs) else "안전 —"
                row.append(f"{100 * share:.1f}% ({safe})")
        if s == "T1":  # 전체 안전율은 무작위 1만 개로
            sub = pd.concat([pd.read_parquet(E / f"{r}_seed{seed}/posthoc/T1_sub.parquet", columns=["in_sample"])
                             for seed in SEEDS]).in_sample.to_numpy()
            row.append(f"{100 * t.safe[sub].mean():.1f}%")
        else:
            row.append(f"{100 * t.safe.mean():.1f}%")
        rows.append(row)
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    return "\n".join(lines + ["| " + " | ".join(r) + " |" for r in rows])


def table():
    md = ["# 14단계 사후 분석 1번: 세트별 판단 x 결정–경로 x 안전", "",
          "`scripts/step14_posthoc_cells.py`가 만든 표. 해석은 `docs/step14_run.md`의 \"(사후 분석) 세트별 판단\" 절. "
          "새 테스트 세트에서 한 번 평가한 결과를 다시 나눈 것이다. 이 결과로 모델이나 τ를 바꾸지 않는다.",
          "칸마다 결정 비율(시드 0·1 평균)과 그 칸에서 그린 경로의 안전율(NC·DAC·DDC 모두 통과, 두 시드의 행을 합침). "
          "일치 = 경로가 결정을 따름(L1 판정 d̂ ≈ d). 판단이 없는 모델은 모두 ACCEPT이다. n은 100개보다 적을 때만 적었다(시드 평균).",
          "T1은 테스트 평가에서 채점하지 않았으므로 여기서 채점했다: 무작위 1만 개(시드 0으로 뽑아 모든 모델에 같은 결정, "
          "`exp/step14/posthoc/T1_sample10k.npy`)와 판단 모델이 거부한 결정 전부. 비율은 T1 전체(75,767개), "
          "ACCEPT 칸의 안전율과 전체 안전율은 무작위 1만 개, REJECT 칸의 안전율은 거부한 결정 전부로 냈다.", ""]
    for s, name, cells in (("T0", "T0 사람 결정 (L2 실행 가능만)", FEAS_CELLS),
                           ("T1", "T1 CF⁺ (실행 가능, 75,767개)", FEAS_CELLS),
                           ("T2", "T2 CF⁻ (실행 불가능, 보이지 않는 원인 제외 16,756개)", INF_CELLS),
                           ("T3_gemma31b", "T3 Gemma 4 31B의 실행 불가능한 결정 (397개)", INF_CELLS)):
        md += [f"## {name}", "", cell_table(s, cells), ""]
    out = ROOT / "docs/step14_posthoc_cells.md"
    out.write_text("\n".join(md))
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "table"])
    a = ap.parse_args()
    prepare() if a.mode == "prepare" else table()


if __name__ == "__main__":
    main()
