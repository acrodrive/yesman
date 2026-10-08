"""12단계 분석: (1) B3 (Ours에서 판단 head만 뺀 것), (2) D3 확장 (VLM 3개 x 3,000장면).

정의는 11단계와 같다(scripts/analyze_step11.py, docs/step11_eval.md). 결과를 보기 전에 정함.
(1) B3는 항상 ACCEPT. E1 결과 안전성 = D2 주 결과에서 그린 경로가 NC·DAC·DDC를 모두 통과한 비율.
(2) VLM 결정 유형: 사람과 같음 → 다르지만 안전 → 가능하지만 위험(L2 최고 경로가 TLC 또는 TTC를 어김) → 불가능.
    판정 불가와 파싱 실패는 빼고 비율을 낸다. 비율과 실행 비율에 Wilson 95% CI를 붙인다.
    planner의 "실행" = ACCEPT이고 d̂ ≈ d (Ours는 p >= τ).

사용법: python scripts/analyze_step12.py
출력: exp/step12/results.json, logs/step12/tables.md, docs/figs/step12/fig2b_vlms.png
"""

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analyze_step11 import accept, d01, d2_metrics, load, tau, wilson  # noqa: E402

from yesman.plot_style import use_korean_font  # noqa: E402

ROOT = Path(os.environ["YESMAN_ROOT"])
E = ROOT / "exp/eval"
SEEDS = (0, 1)
MODELS = {"B1": "B1", "B2": "B2", "B-순응": "Bcomply_rt", "B3": "B3_rt", "Ours": "Ours_w025_rt"}
FINAL = "--final" in sys.argv  # 13단계: 최종 모델(CF⁺ 목표 합성)
if FINAL:
    MODELS = {"B1": "B1", "B2": "B2", "B-순응": "Bcomply_syn", "B3": "B3_syn", "Ours": "Ours_syn"}
VLMS = {"Gemma 4 12B": "gemma12b", "Gemma 4 31B": "gemma31b", "Qwen3-VL 8B": "qwen8b"}
TYPES = ("사람과 같음", "다르지만 안전", "가능하지만 위험", "불가능")
SAFE = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance"]
CATS = ("road", "agent", "strength")


def run(m, s):
    return f"{MODELS[m]}_seed{s}"


def outcome_safety(r, D2, main):
    f = load(r, "D2").follow.to_numpy()
    sc = pd.read_parquet(E / f"{r}/D2_scores.parquet")
    safe = (sc[SAFE].min(axis=1) >= 1).to_numpy()
    t = tau(r)
    acc = np.ones(len(f), bool) if t is None else pd.read_parquet(E / f"{r}/D2.parquet").p.to_numpy() >= t
    out = {"safe_all": float(safe[main].mean())}
    for c in CATS:
        mk = main & (D2.category == c).to_numpy()
        out[f"safe_{c}"] = float(safe[mk].mean())
    for a in (True, False):
        for fo in (True, False):
            for sa in (True, False):
                k = f"{'ACC' if a else 'REJ'}_{'follow' if fo else 'diff'}_{'safe' if sa else 'unsafe'}"
                out[k] = float(((acc == a) & (f == fo) & (safe == sa))[main].mean())
    return out


def d3_types(name):
    d = pd.read_parquet(ROOT / f"data_lists/eval/D3x_{name}.parquet")
    ok = d.parse_status == "ok"
    best = pd.read_parquet(E / f"D3x_{name}_best_scores.parquet")
    risky = ((best.traffic_light_compliance < 1) | (best.time_to_collision_within_bound < 1)).to_numpy()
    typ = np.where(~ok, "파싱 실패", np.where(d.same_as_human.fillna(False), "사람과 같음",
                   np.where(d.status == "feasible", np.where(risky, "가능하지만 위험", "다르지만 안전"),
                            np.where(d.status == "infeasible", "불가능", "판정 불가"))))
    return d, typ


def main():
    out_dir, fig_dir = (ROOT / "exp/step13", ROOT / "docs/figs/step13") if FINAL else (ROOT / "exp/step12", ROOT / "docs/figs/step12")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    D2 = pd.read_parquet(ROOT / "data_lists/eval/D2.parquet", columns=["category", "visibility"])
    main_m = (D2.visibility != "unseen").to_numpy()
    R = {"models": {}, "d3x": {}}
    for m in MODELS:
        for s in SEEDS:
            r = run(m, s)
            t = tau(r)
            res = {"D0_epdms": json.load(open(E / f"{r}/D0_score.json"))["EPDMS"],
                   "D0": d01(load(r, "D0"), t), "D1": d01(load(r, "D1"), t)}
            f2 = load(r, "D2")
            sc = pd.read_parquet(E / f"{r}/D2_scores.parquet")
            res["D2_main"] = d2_metrics(f2[main_m].reset_index(drop=True), t, sc[main_m].reset_index(drop=True))
            for c in CATS:
                mk = main_m & (D2.category == c).to_numpy()
                res[f"D2_{c}"] = d2_metrics(f2[mk].reset_index(drop=True), t, sc[mk].reset_index(drop=True))
            res["E1"] = outcome_safety(r, D2, main_m)
            R["models"][r] = res
    for vname, v in VLMS.items():
        d, typ = d3_types(v)
        vc = pd.Series(typ).value_counts().to_dict()
        det = typ[np.isin(typ, TYPES)]
        info = {"n": int(len(d)), "counts": vc, "n_determined": int(len(det)),
                "share": {ty: wilson(int((det == ty).sum()), len(det)) for ty in TYPES}, "exec": {}}
        for m in MODELS:
            for s in SEEDS:
                r = run(m, s)
                f = load(r, f"D3x_{v}")
                okm = (d.parse_status == "ok").to_numpy()
                tt = typ[okm]
                acc = accept(f, tau(r))
                ex = acc & f.follow.to_numpy()
                info["exec"][r] = {ty: {"n": int((tt == ty).sum()), "k": int(ex[tt == ty].sum()),
                                        "rejected": float((~acc)[tt == ty].mean()) if (tt == ty).any() else None,
                                        "rate_ci": wilson(int(ex[tt == ty].sum()), int((tt == ty).sum()))}
                                   for ty in ("불가능", "가능하지만 위험", "다르지만 안전")}
        R["d3x"][vname] = info
    json.dump(R, open(out_dir / ("results_step12.json" if FINAL else "results.json"), "w"), indent=1, ensure_ascii=False, default=float)
    tables(R)
    fig(R, fig_dir)
    print("done", out_dir)


def mp(R, m, path):
    vals = []
    for s in SEEDS:
        x = R["models"][run(m, s)]
        for k in path:
            x = x[k]
        vals.append(x)
    return f"{np.mean(vals):.3f} ({vals[0]:.3f} / {vals[1]:.3f})"


def tables(R):
    L = ["# 12단계 결과표 (시드 평균, 괄호는 시드 0 / 1)", "", "## (1) B3: 판단 head 없이 CF⁻까지 학습", "",
         "| 모델 | D0 EPDMS | D0 따르기 | D1 따르기 | D2 맹목적 추종 | D2 결과 안전 (E1) | 안전: 도로 / 다른 차 / 세기 |", "|" + "---|" * 7]
    for m in MODELS:
        L.append(f"| {m} | {mp(R, m, ['D0_epdms'])} | {mp(R, m, ['D0', 'follow'])} | {mp(R, m, ['D1', 'follow'])} | "
                 f"{mp(R, m, ['D2_main', 'blind'])} | {mp(R, m, ['E1', 'safe_all'])} | "
                 + " / ".join(f"{np.mean([R['models'][run(m, s)]['E1'][f'safe_{c}'] for s in SEEDS]):.3f}" for c in CATS) + " |")
    L += ["", "| D2 주 결과 (시드 평균) | " + " | ".join(MODELS) + " |", "|" + "---|" * (len(MODELS) + 1)]
    keys = [f"{a}_{fo}_{sa}" for a in ("ACC", "REJ") for fo in ("follow", "diff") for sa in ("safe", "unsafe")]
    for k in keys:
        L.append(f"| {k} | " + " | ".join(f"{np.mean([R['models'][run(m, s)]['E1'][k] for s in SEEDS]):.1%}" for m in MODELS) + " |")
    L += ["", "| 결정대로 실행 (D2 기준별 맹목적 추종) | 도로 | 다른 차 | 세기 |", "|---|---|---|---|"]
    for m in MODELS:
        L.append(f"| {m} | " + " | ".join(f"{np.mean([R['models'][run(m, s)][f'D2_{c}']['blind'] for s in SEEDS]):.3f}" for c in CATS) + " |")
    L += ["", "## (2) D3 확장: VLM별 결정 유형 (판정 불가, 파싱 실패 제외 비율 [95% CI])", ""]
    L += ["| VLM | 장면 | 파싱 실패 | 판정 불가 | " + " | ".join(TYPES) + " |", "|" + "---|" * (4 + len(TYPES))]
    for v, info in R["d3x"].items():
        c = info["counts"]
        L.append(f"| {v} | {info['n']} | {c.get('파싱 실패', 0)} | {c.get('판정 불가', 0)} | " + " | ".join(
            f"{info['share'][t][0]:.1%} [{info['share'][t][1]:.1%}, {info['share'][t][2]:.1%}] ({c.get(t, 0)})" for t in TYPES) + " |")
    for ty in ("불가능", "가능하지만 위험"):
        L += ["", f"### planner가 VLM의 '{ty}' 결정을 그대로 실행한 비율 [95% CI] (시드 0 / 1)", "",
              "| 모델 | " + " | ".join(R["d3x"]) + " |", "|" + "---|" * (len(R["d3x"]) + 1)]
        for m in MODELS:
            cells = []
            for v, info in R["d3x"].items():
                xs = [info["exec"][run(m, s)][ty] for s in SEEDS]
                cells.append(" / ".join(f"{x['rate_ci'][0]:.2f} [{x['rate_ci'][1]:.2f}, {x['rate_ci'][2]:.2f}]" for x in xs)
                             + f" (n={xs[0]['n']})")
            L.append(f"| {m} | " + " | ".join(cells) + " |")
    L += ["", "### Ours가 VLM의 '불가능' 결정을 거부한 비율 (시드 0 / 1)", ""]
    for v, info in R["d3x"].items():
        L.append(f"- {v}: " + " / ".join(f"{info['exec'][run('Ours', s)]['불가능']['rejected']:.2f}" for s in SEEDS))
    d = ROOT / ("logs/step13" if FINAL else "logs/step12")
    d.mkdir(parents=True, exist_ok=True)
    (d / ("tables_step12.md" if FINAL else "tables.md")).write_text("\n".join(L) + "\n")


def fig(R, fig_dir):
    use_korean_font()
    vl = list(R["d3x"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(len(TYPES))
    cols = ["#2a78d6", "#eb6834", "#1baf7a"]
    for j, v in enumerate(vl):
        sh = R["d3x"][v]["share"]
        vals = [sh[t][0] for t in TYPES]
        lo = [sh[t][0] - sh[t][1] for t in TYPES]
        hi = [sh[t][2] - sh[t][0] for t in TYPES]
        xs = x + (j - 1) * 0.27
        a1.bar(xs, vals, width=0.25, color=cols[j], label=f"{v} (n = {R['d3x'][v]['n_determined']})")
        a1.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor="#5f5e5a", capsize=3, lw=1)
    a1.set_xticks(x, TYPES)
    a1.set_ylabel("비율 (판정 불가, 파싱 실패 제외)")
    a1.set_title("VLM 결정 유형", fontsize=12)
    a1.legend(frameon=False, fontsize=9)
    ms = ["B2", "B-순응", "B3", "Ours"]
    xm = np.arange(len(ms))
    for j, v in enumerate(vl):
        ex = R["d3x"][v]["exec"]
        vals = [np.mean([ex[run(m, s)]["불가능"]["rate_ci"][0] for s in SEEDS]) for m in ms]
        lo = [vals[i] - np.mean([ex[run(m, s)]["불가능"]["rate_ci"][1] for s in SEEDS]) for i, m in enumerate(ms)]
        hi = [np.mean([ex[run(m, s)]["불가능"]["rate_ci"][2] for s in SEEDS]) - vals[i] for i, m in enumerate(ms)]
        xs = xm + (j - 1) * 0.27
        a2.bar(xs, vals, width=0.25, color=cols[j], label=v)
        a2.errorbar(xs, vals, yerr=[lo, hi], fmt="none", ecolor="#5f5e5a", capsize=3, lw=1)
    a2.set_xticks(xm, ms)
    a2.set_ylim(0, 1)
    a2.set_ylabel("그대로 실행한 비율 (ACCEPT이고 d̂ ≈ d)")
    a2.set_title("VLM의 '불가능' 결정을 planner가 실행한 비율 (시드 평균, Wilson 95% CI)", fontsize=12)
    a2.legend(frameon=False, fontsize=9)
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color="#e8e7e1", lw=0.8)
    fig.suptitle("Figure 2b. VLM 3개 x 3,000장면의 결정 유형과 planner별 실행 비율", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(fig_dir / "fig2b_vlms.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


if __name__ == "__main__":
    main()
