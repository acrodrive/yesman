"""14단계 성능 향상의 기여 분석 (탐색): 고르기 후보 모델(시드 0)을 navtest(개발 세트)에서 요소별로 나란히 놓는다.

지표는 scripts/step14_navtest_compare.py와 같다. 다만 D0는 평가하지 않으므로 과잉 거부는 D1에서만 센다.
기준점으로 13단계 B3(B3_syn_seed0)를 함께 놓는다. 목록 exp/step14/contrib/runs.txt (줄마다 "요소_모델_값 실행").

짝 비교: 세 VLM의 실행 불가능한 결정(같은 결정)을 합쳐 안전율 차이와 짝 부트스트랩 95% 구간(5,000번)을 낸다.
D3x 이유별(road/agent/strength) 안전율도 함께 낸다.

사용법: python scripts/step14_contrib.py [--out exp/step14/contrib/table.md]
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
SAFE = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance"]
VLMS = ["gemma31b", "gemma12b", "qwen8b"]


def accept(run, f):
    if "p" in f and f.p.notna().any():
        return f.p.to_numpy() >= json.load(open(E / run / "flag_threshold.json"))["tau"]
    return np.ones(len(f), bool)


def metrics(run, step13=False):
    m = {}
    d1 = pd.read_parquet(E / run / "D1_l1.parquet")
    a1 = accept(run, d1)
    m["follow_D1"] = float((a1 & d1.follow.to_numpy(bool)).mean())
    m["reject_D1"] = float((~a1).mean())
    D2 = pd.read_parquet(ROOT / "data_lists/eval/D2.parquet", columns=["category", "visibility"])
    sc = pd.read_parquet(E / run / "D2_scores.parquet")
    safe = (sc[SAFE].min(axis=1) >= 1).to_numpy()
    main = (D2.visibility != "unseen").to_numpy()
    m["safe_D2"] = float(safe[main].mean())
    for c in ("road", "agent", "strength"):
        m[f"safe_D2_{c}"] = float(safe[main & (D2.category == c).to_numpy()].mean())
    for v in VLMS:
        t = pd.read_parquet(ROOT / f"data_lists/eval/D3x_{v}.parquet", columns=["status", "same_as_human", "parse_status"])
        t = t[t.parse_status == "ok"].reset_index(drop=True)
        inf = ((t.status == "infeasible") & ~t.same_as_human.fillna(False)).to_numpy()
        if step13:
            s = pd.read_parquet(EXP / f"step12/d3x_infeasible/{run}_{v}_scores.parquet")
        else:
            s = pd.read_parquet(E / run / f"D3x_{v}_scores.parquet")
            assert len(s) == len(t), (run, v)
            s = s[inf]
        m[f"safe_{v}"] = float((s[SAFE].min(axis=1) >= 1).mean())
        m[f"n_{v}"] = int(inf.sum())
    return m


def d3x_safe(run, step13=False, col="category"):
    """세 VLM의 실행 불가능한 결정을 이어 붙인 (col 값, 안전 여부). col = category(이유) 또는 kind(바뀐 종류)."""
    cats, safe = [], []
    for v in VLMS:
        t = pd.read_parquet(ROOT / f"data_lists/eval/D3x_{v}.parquet",
                            columns=["status", "same_as_human", "parse_status", "category", "change_type"])
        t["kind"] = t.change_type.str.split("/").str[0]
        t = t[t.parse_status == "ok"].reset_index(drop=True)
        inf = ((t.status == "infeasible") & ~t.same_as_human.fillna(False)).to_numpy()
        if step13:
            s = pd.read_parquet(EXP / f"step12/d3x_infeasible/{run}_{v}_scores.parquet")
        else:
            s = pd.read_parquet(E / run / f"D3x_{v}_scores.parquet")[inf]
        cats.append(t[col].to_numpy()[inf])
        safe.append((s[SAFE].min(axis=1) >= 1).to_numpy())
    return np.concatenate(cats), np.concatenate(safe).astype(float)


PAIRS = [("B3_syn_seed0", "B3_s14_sh0_seed0"), ("B3_syn_seed0", "B3_s14_st20k_seed0"),
         ("B3_syn_seed0", "B3_s14_pl04_bs_r05_seed0"), ("B3_s14_pl06_seed0", "B3_s14_st80k_seed0"),
         ("B3_s14_sh0_seed0", "B3_s14_st80k_seed0"), ("B3_s14_st20k_seed0", "B3_s14_st80k_seed0"),
         ("B3_s14_st80k_seed0", "B3_s14_pl04_bs_seed0"), ("B3_s14_pl04_bs_seed0", "B3_s14_pl04_bs_r05_seed0"),
         ("B3_s14_pl04_bs_seed0", "B3_s14_pl04_bs_r1_seed0"), ("OursDiT_s14_pl04_seed0", "OursDiT_s14_pl04_r1_seed0"),
         ("B3_s14_st80k_seed0", "B3_s14_nowide_seed0"), ("B3_s14_st80k_seed0", "B3_s14_neg026_seed0"),
         ("B3_syn_seed0", "B3_s14_neg026_seed0"), ("B3_s14_st80k_seed0", "B3_s14_noplaus_seed0"),
         ("B3_s14_st80k_seed0", "B3_s14_menuonly_seed0"), ("B3_s14_st80k_seed0", "B3_s14_pos024_seed0"),
         ("B3_syn_seed0", "B3_s14_menuonly_seed0"), ("B3_syn_seed0", "B3_s14_pos024_seed0"),
         ("B3_s14_st80k_seed0", "B3_s14_menuneg_seed0"), ("B3_s14_menuonly_seed0", "B3_s14_menuneg_seed0"),
         ("B3_syn_seed0", "B3_s14_menuneg_seed0"),
         # 사후 분석 4번 (2026-10-20): 메뉴만 쓰되 좌우만 바꾸지 않은 CF⁻의 무게 비중을 기준과 같게(0.705)
         ("B3_s14_menuonly_seed0", "B3_s14_menunl70_seed0"), ("B3_s14_st80k_seed0", "B3_s14_menunl70_seed0"),
         ("B3_s14_menuneg_seed0", "B3_s14_menunl70_seed0")]
BY_CAT = ["B3_syn_seed0", "B3_s14_sh0_seed0", "B3_s14_st80k_seed0", "B3_s14_nowide_seed0", "B3_s14_neg026_seed0",
          "B3_s14_noplaus_seed0", "B3_s14_menuonly_seed0", "B3_s14_menuneg_seed0", "B3_s14_pos024_seed0",
          "B3_s14_pl04_bs_r05_seed0", "OursDiT_s14_pl04_r1_jl2_seed0", "B3_s14_menunl70_seed0"]


def pair_text():
    rng = np.random.default_rng(0)
    get = lambda r: d3x_safe(r, step13=r.startswith("B3_syn"))  # noqa: E731
    lines = ["\n### 짝 비교 (D3x 세 VLM 합침, 차이 = 오른쪽 - 왼쪽, 짝 부트스트랩 95% 구간)\n",
             "| 왼쪽 | 오른쪽 | 왼쪽 안전율 | 오른쪽 안전율 | 차이 | 95% 구간 |", "|" + " --- |" * 6]
    done = lambda r: r.startswith("B3_syn") or (E / r / "D3x_qwen8b_scores.parquet").exists()  # noqa: E731
    for a, b in [p for p in PAIRS if done(p[0]) and done(p[1])]:
        _, x = get(a)
        _, y = get(b)
        d = y - x
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)])
        lines.append(f"| {a} | {b} | {x.mean():.3f} | {y.mean():.3f} | {d.mean():+.3f} | "
                     f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}] |")
    lines += ["\n### D3x 이유별 안전율 (세 VLM 합침)\n", "| 실행 | road | agent | strength |", "|" + " --- |" * 4]
    for r in [r for r in BY_CAT if done(r)]:
        cat, y = get(r)
        lines.append(f"| {r} | " + " | ".join(f"{y[cat == k].mean():.3f}" for k in ("road", "agent", "strength")) + " |")
    cat, _ = get(BY_CAT[0])
    lines.append("\n이유별 결정 수: " + str({k: int((cat == k).sum()) for k in ("road", "agent", "strength")}))
    kinds = ["lat_type+strength", "strength", "lon_type+strength", "lon_type", "lat_type+lon_type", "lat_type+lon_type+strength"]
    lines += ["\n### D3x 바뀐 종류별 안전율 (세 VLM 합침)\n", "| 실행 | " + " | ".join(kinds) + " |", "|" + " --- |" * (len(kinds) + 1)]
    for r in [r for r in BY_CAT if done(r)]:
        kd, y = d3x_safe(r, step13=r.startswith("B3_syn"), col="kind")
        lines.append(f"| {r} | " + " | ".join(f"{y[kd == k].mean():.3f}" for k in kinds) + " |")
    kd, _ = d3x_safe(BY_CAT[0], step13=True, col="kind")
    lines.append("\n바뀐 종류별 결정 수: " + str({k: int((kd == k).sum()) for k in kinds}))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    rows = [{"요소": "13단계", "값": "B3_syn", "실행": "B3_syn_seed0", **metrics("B3_syn_seed0", step13=True)}]
    lists = [ROOT / "exp/step14/contrib/runs.txt", ROOT / "exp/step14/contrib/runs2.txt",  # runs2 = 2번 (새로 학습)
             ROOT / "exp/step14/contrib/runs3.txt"]  # runs3 = 사후 분석 4번
    for line in [l for f in lists if f.exists() for l in f.read_text().splitlines()]:
        if not line.strip():
            continue
        key, run = line.split()
        factor, value = key.rsplit("_", 1) if not key.endswith("jl2") else ("neg_Ours", "1 (2층, 최종)")
        rows.append({"요소": factor, "값": value, "실행": f"{run}_seed0", **metrics(f"{run}_seed0")})
    t = pd.DataFrame(rows)
    t.to_csv(EXP / "step14/contrib/navtest_contrib.csv", index=False)
    cols = ["요소", "값", "실행", "follow_D1", "reject_D1", "safe_D2", "safe_D2_road", "safe_D2_agent", "safe_D2_strength",
            "safe_gemma31b", "safe_gemma12b", "safe_qwen8b"]
    fmt = lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)  # noqa: E731
    lines = []
    for f, g in t.groupby("요소", sort=False):
        lines += [f"\n### {f}\n", "| " + " | ".join(cols[1:]) + " |", "|" + " --- |" * (len(cols) - 1)]
        lines += ["| " + " | ".join(fmt(r[c]) for c in cols[1:]) + " |" for _, r in g.iterrows()]
    n = {v: int(t[f"n_{v}"].iloc[0]) for v in VLMS}
    text = "# 14단계 기여 분석: navtest(개발 세트) 요소별 표 (탐색, 시드 0)\n" + "\n".join(lines) + \
        f"\n\nD3x 실행 불가능한 결정 수: {n}\n" + pair_text() + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
