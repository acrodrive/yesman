"""14단계 보고서 그림.

Figure 1 (새 테스트 세트, 시드 평균): 가로 = T1 결정–경로 일치율(ACCEPT이고 d̂ ≈ d), 세로 = 불가능한 결정의 안전율.
  왼쪽 = 규칙 기반 CF⁻(T2 주 결과), 오른쪽 = 실제 Gemma 4 31B의 실행할 수 없는 결정(T3). 별 = 상한선
  (결정 따르기는 B2, 안전은 결정을 무시하는 B0).
Figure 2 (navtest 개발 세트, 탐색): 13단계 → 14단계 모델의 실제 VLM 오답에서의 안전율 (VLM 3개 x B3, Ours).

색: 기본 범주형 팔레트(dataviz skill references/palette.md) 1~6번을 모델 순서대로 고정해서 쓴다. 점마다 이름을 적어
색만으로 구분하지 않는다.

사용법: python scripts/viz_step14.py
출력: docs/figs/step14/fig1_follow_safety.png, fig2_navtest_13_to_14.png
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from step14_eval import infeasible_rows, metrics, per_decision  # noqa: E402
from yesman.plot_style import use_korean_font  # noqa: E402

ROOT = Path(os.environ["YESMAN_ROOT"])
EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
OUT = ROOT / "docs/figs/step14"
ORDER = ["B0", "B1", "B2", "B3", "OursPrev", "Ours"]
LABEL = {"B0": "B0 (결정 무시)", "B1": "B1", "B2": "B2 (+CF⁺)", "B3": "B3 (+CF⁻)", "OursPrev": "이전 Ours", "Ours": "Ours"}
COLOR = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))  # 팔레트 1~6번, 순서 고정
INK, INK2, GRID = "#1f1e1c", "#5f5e5a", "#e6e4dd"


def style():
    use_korean_font()
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": "#fcfcfb",
                         "axes.facecolor": "#fcfcfb"})


def fig1():
    runs = dict(l.split() for l in (EXP / "step14/select/final_models.txt").read_text().splitlines() if l.strip())
    sets = ["T1", "T2", "T3_gemma31b"]
    R = {}
    for k in ORDER:
        pd_ = per_decision(runs[k], "", sets)
        R[k] = metrics(pd_, ["T2", "T3_gemma31b"], k in ("OursPrev", "Ours"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (key, title) in zip(axes, [("safe_T2", "규칙 기반 CF⁻ (T2, n = 16,756)"),
                                       ("safe_T3_gemma31b", "실제 Gemma 4 31B의 틀린 결정 (T3, n = 397)")]):
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        # 오른쪽 위에 몰린 세 점은 이름표를 떼어 놓고 가는 선으로 잇는다 (점 단위 오프셋)
        off = {"B0": (8, 6), "B1": (8, 6), "B2": (8, 6), "B3": (40, -34), "OursPrev": (-20, -52), "Ours": (-62, -26)}
        for k in ORDER:
            x, y = R[k]["follow"], R[k][key]
            ax.scatter(x, y, s=70, color=COLOR[k], edgecolor="#fcfcfb", linewidth=2, zorder=3, label=LABEL[k])
            ox, oy = off[k]
            lead = k in ("B3", "OursPrev", "Ours")
            ax.annotate(LABEL[k], (x, y), xytext=(ox, oy), textcoords="offset points", fontsize=10, color=INK,
                        ha="left" if ox >= 0 else "right", va="center",
                        arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8, shrinkA=0, shrinkB=4) if lead else None)
        ux, uy = R["B2"]["follow"], R["B0"][key]
        ax.scatter(ux, uy, marker="*", s=180, color=INK2, zorder=3)
        ax.annotate("상한선", (ux, uy), xytext=(10, 4), textcoords="offset points", fontsize=10, color=INK2, ha="left")
        ax.set_title(title, fontsize=12, color=INK)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("결정–경로 일치율 (T1, 실행 가능한 결정을 ACCEPT하고 결정대로)", fontsize=10)
    axes[0].set_ylabel("불가능한 결정의 안전율", fontsize=10)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "fig1_follow_safety.png", dpi=150)
    return R


def fig2():
    t = pd.read_csv(EXP / "step14/navtest_compare.csv")
    vl = [("safe_gemma31b", "Gemma 4 31B (n = 94)"), ("safe_gemma12b", "Gemma 4 12B (n = 127)"), ("safe_qwen8b", "Qwen3-VL 8B (n = 194)")]
    pairs = [("B3", "B3", "B3"), ("Ours", "이전 Ours", "Ours")]  # (색 키, 13단계 모델, 14단계 모델)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    yt, yl, y = [], [], 0
    for key, vname in vl:
        for ck, m13, m14 in pairs:
            a = t[(t["단계"] == "13단계") & (t["모델"] == m13)][key].iloc[0]
            b = t[(t["단계"] == "14단계") & (t["모델"] == m14)][key].iloc[0]
            ax.plot([a, b], [y, y], color=GRID, linewidth=2, zorder=1)
            ax.scatter(a, y, s=60, facecolor="#fcfcfb", edgecolor=COLOR[ck], linewidth=2, zorder=3)
            ax.scatter(b, y, s=70, color=COLOR[ck], edgecolor="#fcfcfb", linewidth=2, zorder=3)
            ax.text(b + 0.012, y, f"{a:.2f} → {b:.2f}", va="center", fontsize=9, color=INK2)
            yt.append(y)
            yl.append(f"{vname}  ·  {'B3' if ck == 'B3' else 'Ours'}")
            y += 1
        y += 0.6
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("실제 VLM의 틀린 결정에서 불가능한 결정의 안전율 (navtest)", fontsize=10)
    ax.scatter([], [], s=60, facecolor="#fcfcfb", edgecolor=INK2, linewidth=2, label="13단계 (빈 점)")
    ax.scatter([], [], s=70, color=INK2, label="14단계 (찬 점)")
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_navtest_13_to_14.png", dpi=150)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    R = fig1()
    fig2()
    for k in ORDER:
        print(k, {m: round(R[k][m], 3) for m in ("follow", "safe_T2", "safe_T3_gemma31b")})


if __name__ == "__main__":
    main()
