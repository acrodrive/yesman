"""14단계 사후 분석 2번 (docs/TODO.md): 새 테스트 세트 T3에서 판단이 REJECT를 적게 내는 이유와 p의 분포.

이미 한 번 평가한 결과(exp/eval/<run>_seed<s>/)를 다시 나눠 보기만 한다. 학습·추론·채점은 하지 않고,
이 결과로 모델이나 τ를 바꾸지 않는다. 판단 모델 두 개(이전 Ours, Ours) x 시드 0·1. 비율은 시드 평균.

볼 것
  (a) T3의 실행 가능 / 불가능을 거부 점수(1 − p)로 가르는 AUC. 전체, 사람과 다른 결정만, 결정 종류 안(가중 평균)
  (b) p 구간별 비율과 τ를 바꿨을 때(T0+T1 과잉 거부를 기준으로) T3 불가능 결정의 거부율. 설명용이며 τ는 바꾸지 않는다
  (c) 실행 불가능한 결정(T2, T3)을 판단(ACCEPT / REJECT) x 경로(결정을 따름 / 안 따름) 네 칸으로 나눈 비율, 안전율, p,
      같은 결정에서 B3(판단 없음)의 안전율
  (d) 바꾼 종류(change_type의 앞부분)별, 불가능 이유별 거부율(검증 세트 CF⁻, T2, T3). T2의 종류별 거부율을 T3의 종류 구성으로
      다시 무게 준 값. 사람 결정과의 강도 차이(종류가 같은 축에서 강도 차이의 최댓값)별 거부율
  그림: 검증 세트 / T1·T2 / T3의 p 히스토그램 (log10 odds, 시드 0)

사용법: python scripts/step14_posthoc_judge.py
출력: docs/step14_posthoc_judge.md, docs/figs/step14/judge_p_hist.png
"""

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from step14_eval import SAFE, auc, load  # noqa: E402
from yesman.decision import decision_kind  # noqa: E402
from yesman.plot_style import use_korean_font  # noqa: E402

ROOT = Path(os.environ["YESMAN_ROOT"])
EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
OUT = ROOT / "docs/step14_posthoc_judge.md"
FIG = ROOT / "docs/figs/step14/judge_p_hist.png"
MODELS = {"OursPrev": "Ours_s14_pl04_r1", "Ours": "OursDiT_s14_pl04_r1_jl2"}  # exp/step14/select/final_models.txt
LABEL = {"OursPrev": "이전 Ours", "Ours": "Ours"}
B3 = "B3_s14_pl04_bs_r05"
SEEDS = (0, 1)
OVER = [None, 0.075, 0.10, 1.0]  # τ를 바꿨을 때 볼 T0+T1 과잉 거부 (None = 지금 τ, 1.0 = p < 1이면 모두 거부)
DSTR_BINS = [0, 0.1, 0.2, 0.3, 0.5, 1.01]  # 사람 결정과의 강도 차이 구간
KIND_ORDER = ["lat_type", "lon_type", "strength", "lat_type+strength", "lon_type+strength", "lat_type+lon_type",
              "lat_type+lon_type+strength"]
CELLS = [("REJECT", False, "REJECT·불일치 (바람직)"), ("REJECT", True, "REJECT·일치 (결정 누설)"),
         ("ACCEPT", True, "ACCEPT·일치 (맹목적 추종)"), ("ACCEPT", False, "ACCEPT·불일치 (조용히 피함)")]


def kind(change_type):
    return change_type.fillna("").str.split("/").str[0].replace("", "none")


def delta_strength(src, t0):
    """사람 결정(T0)과의 강도 차이: 두 구간 x 앞뒤·좌우 축 중 행동 종류가 같은 축에서 |강도 차이|의 최댓값 (없으면 NaN)."""
    h = t0.set_index("token").loc[src.token]
    out = np.full(len(src), np.nan)
    for k in (1, 2):
        for ax in ("lon", "lat"):
            c = f"d_seg{k}_{ax}"
            same = src[c].to_numpy() == h[c].to_numpy()
            d = np.abs(src[f"{c}_strength"].to_numpy(float) - h[f"{c}_strength"].to_numpy(float))
            d = np.where(same & np.isfinite(d), d, np.nan)
            out = np.fmax(out, d)
    return out


def sets():
    """세트별 원본 표(결정 정보)."""
    d = ROOT / "data_lists/eval_step14"
    t0 = pd.read_parquet(d / "T0.parquet")
    t3 = pd.read_parquet(d / "T3_gemma31b.parquet")
    t3 = t3.assign(kind=kind(t3.change_type), dkind=decision_kind(t3), dstr=delta_strength(t3, t0))
    t2 = pd.read_parquet(d / "T2.parquet")
    t2 = t2.assign(kind=kind(t2.change_type), dstr=delta_strength(t2, t0))[
        ["token", "cf_name", "change_type", "category", "visibility", "kind", "dstr"]]
    val = pd.read_parquet(ROOT / "exp/l3/train_bundle_val_step14_plaus.parquet",
                          columns=["token", "cf_name", "sample_type", "change_type", "source"])
    return t3, t2, val.assign(kind=kind(val.change_type))


def frames(run, t3, t2, val):
    """시드마다 세트별 결과 표 (원본 표와 같은 행 순서)."""
    out = []
    for s in SEEDS:
        r = f"{run}_seed{s}"
        f = {k: load(r, k) for k in ("T0", "T1", "T2", "T3_gemma31b")}
        for k, src in (("T2", t2), ("T3_gemma31b", t3)):
            for c in ("token", "cf_name"):
                assert (f[k][c].to_numpy() == src[c].to_numpy()).all(), (r, k, c)
        f["T3"] = pd.concat([f.pop("T3_gemma31b").drop(columns=["category", "status", "cf_name", "token", "log_name"]),
                             t3[["token", "cf_name", "status", "same_as_human", "category", "kind", "dkind",
                                 "dstr"]]], axis=1)
        f["T2"] = pd.concat([f["T2"][["p", "accept", "follow", "safe_path"]], t2], axis=1)
        v = load(r, "val").merge(val, on=["token", "cf_name", "sample_type"], how="left", validate="1:1")
        f["val"] = v[v.visibility != "unseen"]
        f["tau"] = json.load(open(EXP / f"eval/{r}/flag_threshold.json"))["tau"]
        out.append(f)
    return out


def pct(x):
    return f"{100 * x:.1f}%" if np.isfinite(x) else "—"


def summary(F):
    rows = []
    for m, fs in F.items():
        def mean(fn):
            return np.mean([fn(f) for f in fs])
        inf = lambda f: f["T3"][f["T3"].status == "infeasible"]  # noqa: E731
        fea = lambda f: f["T3"][f["T3"].status == "feasible"]  # noqa: E731

        def auc_t3(f, drop_same=False):
            t = f["T3"][f["T3"].status.isin(["feasible", "infeasible"])]
            if drop_same:
                t = t[~((t.status == "feasible") & t.same_as_human.fillna(False))]
            return auc(1 - t.p.to_numpy(), (t.status == "infeasible").to_numpy())

        def auc_kind(f):
            t = f["T3"][f["T3"].status.isin(["feasible", "infeasible"])]
            a = [(auc(1 - g.p.to_numpy(), (g.status == "infeasible").to_numpy()), len(g))
                 for _, g in t.groupby("dkind")]
            a = [(x, n) for x, n in a if np.isfinite(x)]
            return sum(x * n for x, n in a) / sum(n for _, n in a)

        def auc_t12(f):
            t = pd.concat([f["T1"].assign(bad=False), f["T2"][f["T2"].visibility != "unseen"].assign(bad=True)])
            return auc(1 - t.p.to_numpy(), t.bad.to_numpy())
        over = lambda f: 1 - pd.concat([f["T0"], f["T1"]]).accept.mean()  # noqa: E731
        t2v = lambda f: f["T2"][f["T2"].visibility != "unseen"]  # noqa: E731
        rows.append([LABEL[m], " / ".join(f"{f['tau']:.4g}" for f in fs), pct(mean(over)),
                     pct(mean(lambda f: 1 - fea(f).accept.mean())), pct(mean(lambda f: 1 - t2v(f).accept.mean())),
                     pct(mean(lambda f: 1 - inf(f).accept.mean())), f"{mean(auc_t12):.3f}", f"{mean(auc_t3):.3f}",
                     f"{mean(lambda f: auc_t3(f, True)):.3f}", f"{mean(auc_kind):.3f}"])
    return table(["모델", "τ (시드 0 / 1)", "거부 T0+T1 (과잉 거부)", "거부 T3 실행 가능", "거부 T2", "**거부 T3 불가능**",
                  "AUC T1∪T2", "AUC T3", "AUC T3 (사람과 다른 결정만)", "AUC T3 결정 종류 안"], rows)


def pbins(F):
    rows = []
    for m, fs in F.items():
        for st, name in (("infeasible", "불가능"), ("feasible", "가능")):
            def share(lo, hi, f):
                p = f["T3"][f["T3"].status == st].p.to_numpy()
                return ((p >= lo) & (p < hi)).mean()
            b = [np.mean([share(*fn(f), f) for f in fs]) for fn in (
                lambda f: (-1, f["tau"]), lambda f: (f["tau"], 0.5), lambda f: (max(f["tau"], 0.5), 0.99),
                lambda f: (max(f["tau"], 0.99), 0.9999), lambda f: (max(f["tau"], 0.9999), 1.0), lambda f: (1.0, 2))]
            rows.append([LABEL[m], name] + [pct(x) for x in b])
    return table(["모델", "T3 결정", "p < τ (REJECT)", "τ ≤ p < 0.5", "0.5 ≤ p < 0.99", "0.99 ≤ p < 0.9999",
                  "0.9999 ≤ p < 1", "p = 1 (float32)"], rows)


def sweep(F):
    rows = []
    for m, fs in F.items():
        for o in OVER:
            vals = []
            for f in fs:
                feas = pd.concat([f["T0"], f["T1"]]).p.to_numpy()
                tau = f["tau"] if o is None else np.quantile(feas, o)
                t3 = f["T3"]
                vals.append([(feas < tau).mean(), (t3[t3.status == "feasible"].p < tau).mean(),
                             (f["T2"][f["T2"].visibility != "unseen"].p < tau).mean(),
                             (t3[t3.status == "infeasible"].p < tau).mean()])
            v = np.mean(vals, 0)
            name = "지금 τ" if o is None else "p < 1이면 모두 거부 (상한)" if o == 1.0 else f"과잉 거부 {pct(o)}가 되는 τ"
            rows.append([LABEL[m], name] + [pct(x) for x in v])
    return table(["모델", "τ", "거부 T0+T1", "거부 T3 실행 가능", "거부 T2", "거부 T3 불가능"], rows)


def cells(F, b3_safe):
    """실행 불가능한 결정(T2 보이지 않는 원인 제외, T3 L2 불가능)의 판단 x 경로 네 칸.
    b3_safe[세트][시드] = 같은 결정에서 B3의 안전 여부 (세트 원본 표와 행 순서 같음)."""
    rows = []
    for st, keep in (("T2", lambda t: t.visibility != "unseen"), ("T3", lambda t: t.status == "infeasible")):
        for m, fs in F.items():
            t = pd.concat([f[st].assign(b3_safe=b3_safe[st][i])[keep(f[st])] for i, f in enumerate(fs)])
            for acc, fol, name in CELLS:
                g = t[(t.accept == (acc == "ACCEPT")) & (t.follow == fol)]
                rows.append([st, LABEL[m], name, pct(len(g) / len(t)), f"{len(g) / len(SEEDS):,.0f}",
                             pct(g.safe_path.mean()), pct(g.b3_safe.mean()), f"{g.p.median():.3g}" if len(g) else "—"])
    return table(["세트", "모델", "칸", "비율", "결정 수 (시드 평균)", "안전율", "같은 결정에서 B3 안전율", "p 중앙값"], rows)


def by_group(F, col, order=None):
    """col 값별 거부율: 검증 세트 CF⁻, T2, T3 불가능. T3는 '조용히 피함' 비율과 안전율도."""
    rows = []
    t3s = {m: pd.concat([f["T3"][f["T3"].status == "infeasible"] for f in fs]) for m, fs in F.items()}
    t2s = {m: pd.concat([f["T2"][f["T2"].visibility != "unseen"] for f in fs]) for m, fs in F.items()}
    vas = {m: pd.concat([f["val"][f["val"].sample_type == "cf_neg"] for f in fs]) for m, fs in F.items()}
    keys = order or sorted(set(t3s["Ours"][col].dropna()) | set(t2s["Ours"][col].dropna()))
    for k in keys:
        row = [k]
        for m in F:
            v, t2, t3 = (x[m][x[m][col] == k] if col in x[m] else x[m].iloc[:0] for x in (vas, t2s, t3s))
            row += [f"{pct(1 - v.accept.mean())} ({len(v) // 2:,})" if len(v) else "—",
                    f"{pct(1 - t2.accept.mean())} ({len(t2) // 2:,})" if len(t2) else "—",
                    f"{pct(1 - t3.accept.mean())} ({len(t3) // 2})" if len(t3) else "—",
                    pct((t3.accept & ~t3.follow).mean()) if len(t3) else "—",
                    pct(t3.safe_path.mean()) if len(t3) else "—"]
        rows.append(row)
    head = [col]
    for m in F:
        head += [f"{LABEL[m]} 거부 val CF⁻ (n)", "거부 T2 (n)", "**거부 T3 (n)**", "T3 조용히 피함", "T3 안전율"]
    return table(head, rows)


def reweight(F, col, order):
    """T2(없으면 검증 세트 CF⁻)의 col 값별 거부율을 T3 불가능 결정의 col 구성으로 다시 무게 준 거부율."""
    rows = []
    for m, fs in F.items():
        vals = []
        for f in fs:
            t2, t3 = f["T2"][f["T2"].visibility != "unseen"], f["T3"][f["T3"].status == "infeasible"]
            v = f["val"][f["val"].sample_type == "cf_neg"]
            w = t3[col].value_counts(normalize=True)
            r2 = (1 - t2.groupby(col).accept.mean()).combine_first(1 - v.groupby(col).accept.mean())
            vals.append([1 - t2.accept.mean(), float((w * r2.reindex(w.index)).sum()), 1 - t3.accept.mean()])
        v = np.mean(vals, 0)
        rows.append([LABEL[m]] + [pct(x) for x in v])
    return table(["모델", "거부 T2", f"T2의 {col}별 거부율 x T3의 {col} 구성", "거부 T3 불가능"], rows)


def by_dstr(F):
    """바꾼 종류가 강도뿐인 결정(kind = strength)의 사람 결정과의 강도 차이별 거부율 (T2, T3 불가능)."""
    rows = []
    cut = lambda t: pd.cut(t.dstr, DSTR_BINS, right=False)  # noqa: E731
    t2s = {m: pd.concat([f["T2"][(f["T2"].visibility != "unseen") & (f["T2"].kind == "strength")] for f in fs])
           for m, fs in F.items()}
    t3s = {m: pd.concat([f["T3"][(f["T3"].status == "infeasible") & (f["T3"].kind == "strength")] for f in fs])
           for m, fs in F.items()}
    for b in pd.IntervalIndex.from_breaks(DSTR_BINS, closed="left"):
        row = [f"{b.left:.1f} ~ {min(b.right, 1):.1f}"]
        for m in F:
            t2, t3 = t2s[m][cut(t2s[m]) == b], t3s[m][cut(t3s[m]) == b]
            row += [f"{pct(1 - t2.accept.mean())} ({len(t2) // 2})" if len(t2) else "—",
                    f"{pct(1 - t3.accept.mean())} ({len(t3) // 2})" if len(t3) else "—"]
        rows.append(row)
    med = [f"{t2s[m].dstr.median():.2f} / {t3s[m].dstr.median():.2f}" for m in F][0]
    head = ["강도 차이"]
    for m in F:
        head += [f"{LABEL[m]} 거부 T2 (n)", "**거부 T3 (n)**"]
    return table(head, rows) + f"\n\n강도 차이 중앙값 T2 / T3: {med}"


def table(head, rows):
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    return "\n".join(lines + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def log_odds(p):
    p = np.clip(p.astype(np.float64), 1e-7, 1 - 1e-7)
    return np.log10(p / (1 - p))


def figure(F):
    use_korean_font()
    ink, ink2, grid, surf = "#1f1e1c", "#5f5e5a", "#e6e4dd", "#fcfcfb"
    col = {"가능": "#2a78d6", "불가능": "#eb6834"}  # 팔레트 1, 2번
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": ink2,
                         "axes.labelcolor": ink, "xtick.color": ink2, "ytick.color": ink2, "figure.facecolor": surf,
                         "axes.facecolor": surf, "axes.unicode_minus": False})
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True, sharey=True)
    bins = np.linspace(-7, 7, 57)
    for i, (m, fs) in enumerate(F.items()):
        f = fs[0]
        v, t3 = f["val"], f["T3"]
        panels = [("검증 세트 (CF⁺ / CF⁻)", v[v.sample_type == "cf_pos"].p, v[v.sample_type == "cf_neg"].p),
                  ("테스트 T1 / T2 (규칙 기반 메뉴 CF)", f["T1"].p, f["T2"][f["T2"].visibility != "unseen"].p),
                  ("테스트 T3 (Gemma 4 31B 결정)", t3[t3.status == "feasible"].p, t3[t3.status == "infeasible"].p)]
        for j, (title, pos, neg) in enumerate(panels):
            ax = axes[i, j]
            ax.grid(True, axis="y", color=grid, linewidth=0.8)
            for name, x in (("가능", pos), ("불가능", neg)):
                w = np.full(len(x), 1 / len(x))
                ax.hist(log_odds(x.to_numpy()), bins=bins, weights=w, histtype="step", linewidth=2, color=col[name],
                        label=f"실행 {name}")
            ax.axvline(np.log10(f["tau"] / (1 - f["tau"])), color=ink, linestyle="--", linewidth=1)
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 1.2)
            ax.set_yticks([1e-3, 1e-2, 1e-1, 1], ["0.1%", "1%", "10%", "100%"])
            ax.minorticks_off()
            ax.text(0.5, 0.97, f"가능 {len(pos):,} / 불가능 {len(neg):,}", transform=ax.transAxes, ha="center",
                    va="top", fontsize=9, color=ink2)
            if i == 0:
                ax.set_title(title, fontsize=11, color=ink)
            if j == 0:
                ax.set_ylabel(f"{LABEL[m]}\n결정 비율")
            if i == 1:
                ax.set_xlabel("log10(p / (1 − p))   ← REJECT 쪽 | ACCEPT 쪽 →")
    fig.text(0.5, 0.005, "점선 = τ (검증 세트에서 정함). 양 끝 막대는 p가 0 또는 1에 붙은 결정(±7로 자름). 시드 0",
             ha="center", fontsize=9, color=ink2)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, fontsize=10, labelcolor=ink)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)


def main():
    t3, t2, val = sets()
    F = {m: frames(r, t3, t2, val) for m, r in MODELS.items()}
    b3_safe = {st: [] for st in ("T2", "T3")}
    for s in SEEDS:
        for st, name in (("T2", "T2"), ("T3", "T3_gemma31b")):
            sc = pd.read_parquet(EXP / f"eval/{B3}_seed{s}/{name}_scores.parquet")
            b3_safe[st].append((sc[SAFE].min(axis=1) >= 1).to_numpy())
    md = ["# 14단계 사후 분석 2번: T3에서 판단이 REJECT를 적게 내는 이유와 p의 분포",
          "", "`scripts/step14_posthoc_judge.py`가 만든 표. 해석은 `docs/step14_run.md`의 \"(사후 분석) T3 판단\" 절. 이미 한 번 평가한 결과를 다시 나눈 것이다(사후 분석). "
          "이 결과로 모델이나 τ를 바꾸지 않는다. 시드 0·1 평균. T2는 보이지 않는 원인 제외, 검증 세트 CF⁻도 같다.",
          f"이전 Ours = `{MODELS['OursPrev']}`, Ours = `{MODELS['Ours']}`, B3 = `{B3}`.",
          "", "## (a) 거부율과 AUC (거부 점수 = 1 − p)", "", summary(F),
          "", "## (b) T3의 p 구간별 비율", "", pbins(F),
          "", "### τ를 바꿨다면 (설명용, τ는 바꾸지 않음)", "",
          "τ를 T0+T1(실행 가능한 결정)의 과잉 거부가 주어진 값이 되도록 옮겼을 때의 거부율.", "", sweep(F),
          "", "## (c) 실행 불가능한 결정의 네 칸 (판단 x 경로)", "", "일치 = 경로가 결정을 따름(L1 판정 d̂ ≈ d). "
          "T2 = 보이지 않는 원인 제외 16,756개, T3 = 397개.", "", cells(F, b3_safe),
          "", "## (d) 바꾼 종류별 거부율", "",
          "바꾼 종류 = change_type의 앞부분(사람 결정에서 무엇을 바꿨나). n은 결정 수(시드 평균). "
          "'조용히 피함' = ACCEPT인데 경로가 결정을 따르지 않은 비율.", "", by_group(F, "kind", KIND_ORDER),
          "", "### 불가능 이유별", "", by_group(F, "category", ["road", "agent", "strength"]),
          "", "강도 기준(category = strength)은 강도를 한 단계 낮춘 결정이 실행 가능한 경우와 같다(T2 1,533개, T3 206개 일치).",
          "", "### 종류 구성으로 설명되는 몫", "",
          "T2의 종류별(이유별) 거부율이 그대로이고 종류(이유) 구성만 T3처럼 바뀌었다면의 거부율. "
          "T2에 없는 종류는 검증 세트 CF⁻의 거부율을 쓴다.", "",
          reweight(F, "kind", KIND_ORDER), "", reweight(F, "category", ["road", "agent", "strength"]),
          "", "### 사람 결정과의 강도 차이 (바꾼 종류가 강도뿐인 결정)", "", by_dstr(F),
          "", f"그림: `docs/figs/step14/{FIG.name}`", ""]
    OUT.write_text("\n".join(md))
    figure(F)
    print("\n".join(md))


if __name__ == "__main__":
    main()
