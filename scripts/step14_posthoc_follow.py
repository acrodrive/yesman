"""14단계 사후 분석 5번 (docs/TODO.md): Ours가 B3보다 결정을 덜 따르는 이유.

결정–경로 일치율(ACCEPT이고 d̂ ≈ d)을 두 몫으로 나눈다.
  경로 일치율 = 판단과 상관없이 경로가 결정을 따른 비율 (d̂ ≈ d)
  판단 몫     = 경로 일치율 − 결정–경로 일치율 = 경로는 따랐는데 판단이 거부한 비율
그리고 B3 → Ours의 차이를 고르기 후보(시드 0)로 단계별로 나눈다(같은 결정끼리 짝지은 부트스트랩 95% 구간).
  CF⁻ 무게 0.5 → 1          B3 pl04_bs_r05 → pl04_bs_r1 (둘 다 강도 기준 무게 켬)
  강도 기준 무게 끔           B3 pl04_bs_r1 → oursconf (Ours 구성의 B3: plausible 0.4, CF⁻ 무게 1)
  판단만 붙임 (경로 같음)     oursconf → OursDiT jl2_frz (oursconf의 DiT를 고정하고 2층에서 판단 head만 학습)
  함께 학습                   jl2_frz → OursDiT jl2 (최종 Ours)
세트: 검증 세트(CF⁺, 고르기에 쓴 세트), navtest D1(개발 세트). 테스트 T1은 최종 B3 / Ours(시드 0·1)의 차이를 두 몫으로만 나눈다.
이미 낸 결과를 다시 나누는 사후 분석이며, 이 결과로 모델이나 τ를 바꾸지 않는다.

사용법: python scripts/step14_posthoc_follow.py
출력: docs/step14_posthoc_follow.md
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ["YESMAN_ROOT"])
E = Path(os.environ["NAVSIM_EXP_ROOT"]) / "eval"
OUT = ROOT / "docs/step14_posthoc_follow.md"
CHAIN = [("B3 CF⁻ 무게 0.5 (최종 B3)", "B3_s14_pl04_bs_r05"), ("B3 CF⁻ 무게 1", "B3_s14_pl04_bs_r1"),
         ("B3 Ours 구성 (강도 기준 무게 끔)", "B3_s14_oursconf"), ("Ours DiT 고정 + 판단 (2층)", "OursDiT_s14_pl04_r1_jl2_frz"),
         ("Ours 함께 학습 (2층, 최종 Ours)", "OursDiT_s14_pl04_r1_jl2")]
EXTRA = [("B3 CF⁻ 무게 0.25", "B3_s14_pl04_bs"), ("Ours 마지막 층 CF⁻ 무게 0.25", "OursDiT_s14_pl04"),
         ("Ours 마지막 층 CF⁻ 무게 0.5", "OursDiT_s14_pl04_r05"), ("Ours 마지막 층 CF⁻ 무게 1", "OursDiT_s14_pl04_r1"),
         ("이전 Ours (판단 토큰)", "Ours_s14_pl04_r1")]
FINAL = {"B3": "B3_s14_pl04_bs_r05", "Ours": "OursDiT_s14_pl04_r1_jl2"}
N_BOOT = 2000


def load(run, s):
    f = pd.read_parquet(E / run / f"{s}_l1.parquet")
    if s == "val":
        f = f[f.sample_type == "cf_pos"]
    th = E / run / "flag_threshold.json"
    acc = (f.p >= json.load(open(th))["tau"]).to_numpy() if "p" in f and th.exists() else np.ones(len(f), bool)
    path = f.follow.to_numpy()
    return pd.DataFrame({"path": path.astype(float), "follow": (acc & path).astype(float), "reject": (~acc).astype(float)},
                        index=pd.MultiIndex.from_arrays([f.token, f.cf_name]))


def boot(a, b, rng):
    """같은 결정에서 b − a의 평균과 95% 구간."""
    d = b - a
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(N_BOOT)])
    return d.mean(), np.quantile(m, 0.025), np.quantile(m, 0.975)


def pp(x):
    return f"{100 * x:+.1f}%p"


def pc(x):
    return f"{100 * x:.1f}%"


def section(s, rng):
    L = []
    have = [(n, r) for n, r in CHAIN + EXTRA if (E / f"{r}_seed0" / f"{s}_l1.parquet").exists()]
    F = {r: load(f"{r}_seed0", s) for _, r in have}
    L += ["| 모델 (시드 0) | 실행 | 결정–경로 일치율 | 경로 일치율 | 판단 몫 (경로는 따랐는데 거부) | 거부율 |",
          "| --- | --- | --- | --- | --- | --- |"]
    for n, r in have:
        f = F[r]
        L.append(f"| {n} | `{r}` | {pc(f.follow.mean())} | {pc(f.path.mean())} | {pc(f.path.mean() - f.follow.mean())} | "
                 f"{pc(f.reject.mean())} |")
    L += ["", "단계별 차이 (뒤 − 앞, 짝지은 부트스트랩 95% 구간)", "",
          "| 단계 | 결정–경로 일치율 | 경로 일치율 |", "| --- | --- | --- |"]
    chain = [(n, r) for n, r in CHAIN if r in F]
    for (n0, r0), (n1, r1) in zip(chain, chain[1:]):
        a, b = F[r0].align(F[r1], join="inner", axis=0)
        f = boot(a.follow.to_numpy(), b.follow.to_numpy(), rng)
        p = boot(a.path.to_numpy(), b.path.to_numpy(), rng)
        L.append(f"| {n0} → {n1} | {pp(f[0])} [{pp(f[1])}, {pp(f[2])}] | {pp(p[0])} [{pp(p[1])}, {pp(p[2])}] |")
    if len(chain) >= 2:
        a, b = F[chain[0][1]].align(F[chain[-1][1]], join="inner", axis=0)
        f = boot(a.follow.to_numpy(), b.follow.to_numpy(), rng)
        L.append(f"| **합: {chain[0][0]} → {chain[-1][0]}** | {pp(f[0])} [{pp(f[1])}, {pp(f[2])}] | "
                 f"{pp(b.path.mean() - a.path.mean())} |")
    return L + [""]


def test_section(rng):
    F = {}
    for k, r in FINAL.items():
        fs = [load(f"{r}_seed{s}", "T1") for s in (0, 1)]
        F[k] = (fs[0] + fs[1]) / 2
    a, b = F["B3"].align(F["Ours"], join="inner", axis=0)
    tot = boot(a.follow.to_numpy(), b.follow.to_numpy(), rng)
    path = boot(a.path.to_numpy(), b.path.to_numpy(), rng)
    judge = boot((b.path - b.follow).to_numpy(), (a.path - a.follow).to_numpy(), rng)  # 일치율 차이에 대한 판단의 몫 (음수)
    return ["| 모델 (시드 0·1 평균) | 결정–경로 일치율 | 경로 일치율 | 판단 몫 | 거부율 |", "| --- | --- | --- | --- | --- |",
            *[f"| {k} | {pc(f.follow.mean())} | {pc(f.path.mean())} | {pc(f.path.mean() - f.follow.mean())} | "
              f"{pc(f.reject.mean())} |" for k, f in (("B3", a), ("Ours", b))], "",
            f"- Ours − B3 결정–경로 일치율: {pp(tot[0])} [{pp(tot[1])}, {pp(tot[2])}]",
            f"  - 경로가 덜 따른 몫: {pp(path[0])} [{pp(path[1])}, {pp(path[2])}]",
            f"  - 판단이 거부한 몫(경로는 따름): {pp(judge[0])} [{pp(judge[1])}, {pp(judge[2])}]", ""]


def main():
    rng = np.random.default_rng(0)
    md = ["# 14단계 사후 분석 5번: Ours가 결정을 덜 따르는 이유", "",
          "`scripts/step14_posthoc_follow.py`가 만든 표. 해석은 `docs/step14_run.md`의 \"(사후 분석) 결정 따르기\" 절. "
          "결정–경로 일치율 = ACCEPT이고 d̂ ≈ d, 경로 일치율 = 판단과 상관없이 d̂ ≈ d. 판단이 없는 B3는 둘이 같다.", "",
          "## 검증 세트 CF⁺ (고르기에 쓴 세트)", "", *section("val", rng),
          "## navtest D1 (개발 세트)", "", *section("D1", rng),
          "## 새 테스트 T1 (최종 모델, 사후 분석: 이미 낸 결과를 두 몫으로 나누기만 함)", "", *test_section(rng)]
    OUT.write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
