"""14단계 사후 분석 (탐색): T1, T2, T3를 결정 유형별로 나눈 구성과, 유형마다 모델별 판단 x 결정–경로 x 안전 칸.

대화형 페이지가 읽는 docs/step14_type_cells.json을 만든다. 이미 한 번 평가한 결과를 다시 나누기만 한다.
모델이나 τ를 바꾸지 않고, 새로 채점하지 않는다.
- 구성: data_lists/eval_step14만 읽는다(로컬에서도 된다, --no_models)
- 칸: exp/eval/<run>/의 L1 판정과 p, 채점 결과. 칸과 T1 안전율 규칙은 step14_posthoc_cells.py와 같다
  (T1은 ACCEPT 칸은 무작위 1만 개, REJECT 칸은 거부한 결정 전부로 안전율을 낸다. 채점하지 않은 행은 안전율에서 뺀다)

유형:
- 바꾼 종류: change_type의 앞부분(사람 결정에서 무엇을 바꿨나)
- 불가능 이유: category (T2, T3 실행 불가능)
- 메뉴 결정: cf_name (T1, T2)
- 그럴싸함: 강도까지 같은 그럴싸함 점수(cf_plausibility.py, 사전 등록 6절)의 구간 0 / 0~0.1 / 0.1 이상과,
  바꾼 종류 x 그럴싸함 (T2, T3). exp/step14/posthoc/<파일>_plaus.parquet가 있을 때만
  (scripts/run_step14_posthoc_plaus.sh)

사용법: python scripts/step14_type_cells.py [--no_models]   # → docs/step14_type_cells.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ["YESMAN_ROOT"])
D = ROOT / "data_lists/eval_step14"
PLAUS = Path(os.environ.get("NAVSIM_EXP_ROOT", "")) / "step14/posthoc"
OUT = ROOT / "docs/step14_type_cells.json"

SETS = [  # (id, 파일, 이름, 실행 가능 여부, 행 고르기, 유형 축)
    ("T1", "T1", "T1: 규칙 기반 CF⁺ (실행 가능)", True, None, ["kind", "cf_name"]),
    ("T2", "T2", "T2: 규칙 기반 CF⁻ (실행 불가능, 보이지 않는 원인 제외)", False, ("visibility", lambda c: c != "unseen"),
     ["kind", "category", "cf_name", "plaus", "kind_plaus"]),
    ("T3_inf", "T3_gemma31b", "T3: Gemma 4 31B의 실행 불가능한 결정", False, ("status", lambda c: c == "infeasible"),
     ["kind", "category", "plaus", "kind_plaus"]),
    ("T3_fea", "T3_gemma31b", "T3: Gemma 4 31B의 실행 가능한 결정", True, ("status", lambda c: c == "feasible"),
     ["kind", "plaus", "kind_plaus"]),
]
AXES = {"kind": "바꾼 종류", "category": "불가능 이유", "cf_name": "메뉴 결정", "plaus": "그럴싸함",
        "kind_plaus": "바꾼 종류 x 그럴싸함"}
PLAUS_BINS = {"p0": "그럴싸함 0", "p1": "그럴싸함 0~0.1", "p2": "그럴싸함 0.1 이상"}
KIND = {"lat_type": "좌우", "lon_type": "앞뒤", "strength": "강도", "none": "사람과 같음"}
LABELS = {
    "category": {"road": "도로", "agent": "다른 차", "strength": "강도"},
    "cf_name": {"faster": "faster (더 빨리)", "slower": "slower (더 천천히)", "stop": "stop (정지)",
                "keep": "keep (차로 유지)", "offset_L": "offset 왼쪽", "offset_R": "offset 오른쪽",
                "lc_L": "차로 변경 왼쪽", "lc_R": "차로 변경 오른쪽", "turn_L": "회전 왼쪽", "turn_R": "회전 오른쪽",
                "turn_late_L": "늦은 회전 왼쪽", "turn_late_R": "늦은 회전 오른쪽"},
}
CELLS = [(True, True), (True, False), (False, False), (False, True)]  # (accept, follow)


def kind(change_type):
    return change_type.fillna("").str.split("/").str[0].replace("", "none")


def label(axis, v):
    if axis == "kind":
        return " + ".join(KIND.get(p, p) for p in v.split("+"))
    if axis == "plaus":
        return PLAUS_BINS[v]
    if axis == "kind_plaus":
        k, b = v.split("|")
        return f"{label('kind', k)} · {PLAUS_BINS[b]}"
    return LABELS.get(axis, {}).get(v, v)


def table(file, pick):
    t = pd.read_parquet(D / f"{file}.parquet")
    t["kind"] = kind(t.change_type)
    pl = PLAUS / f"{file}_plaus.parquet"
    if pl.exists():
        s = pd.read_parquet(pl, columns=["token", "plaus_strength"])
        assert (s.token.to_numpy() == t.token.to_numpy()).all(), pl
        t["plaus"] = np.select([s.plaus_strength <= 0, s.plaus_strength < 0.1], ["p0", "p1"], "p2")
        t["kind_plaus"] = t.kind + "|" + t.plaus
    keep = pick[1](t[pick[0]]).to_numpy() if pick else np.ones(len(t), bool)
    return t, keep


def type_order(ts, ax):
    """유형 순서: 그럴싸함은 구간 순서, 바꾼 종류 x 그럴싸함은 종류의 결정 수 순서 다음 구간 순서, 나머지는 결정 수 순서."""
    vals = ts[ax].fillna("none")
    n = vals.value_counts()
    if ax == "plaus":
        return [(b, n[b]) for b in PLAUS_BINS if b in n]
    if ax == "kind_plaus":
        kinds = ts.kind.value_counts().index.tolist()
        return sorted(n.items(), key=lambda kv: (kinds.index(kv[0].split("|")[0]), list(PLAUS_BINS).index(kv[0].split("|")[1])))
    return list(n.items())


def model_rows(run, file):
    """행마다 accept, follow, 채점 여부와 안전·실패 항목 (step14_posthoc_cells.frame과 같은 규칙, 행 고르기 전)."""
    from step14_posthoc_cells import E, SAFE, accept
    f = pd.read_parquet(E / run / f"{file}_l1.parquet")
    out = pd.DataFrame({"accept": accept(run, f), "follow": f.follow.to_numpy(bool)})
    if file == "T1":
        sub, scp = E / run / "posthoc/T1_sub.parquet", E / run / "posthoc/T1_sub_scores.parquet"
        if scp.exists():
            sc = pd.read_parquet(scp)
            use = np.where(out.accept, pd.read_parquet(sub, columns=["in_sample"]).in_sample.to_numpy(), True)
        else:
            sc, use = None, np.zeros(len(f), bool)
    else:
        scp = E / run / f"{file}_scores.parquet"
        sc, use = (pd.read_parquet(scp), np.ones(len(f), bool)) if scp.exists() else (None, np.zeros(len(f), bool))
    if sc is None:
        sc = pd.DataFrame(np.nan, index=range(len(f)), columns=SAFE)
    assert len(sc) == len(f), (run, file)
    scored = use & ~sc[SAFE].isna().any(axis=1).to_numpy()
    out["scored"] = scored
    for col, name in zip(SAFE, ("coll", "off", "wrong")):
        out[name] = scored & (sc[col].to_numpy() < 1)
    out["safe"] = scored & (sc[SAFE].min(axis=1).to_numpy() >= 1)
    return out


def cells(rows):
    """칸마다 결정 수와, 그 칸에서 채점한 결정 중 안전·충돌·도로 이탈·역주행 수 (두 시드를 합친 수)."""
    res = []
    for acc, fol in CELLS:
        g = rows[(rows.accept == acc) & (rows.follow == fol)]
        res.append({"n": int(len(g)), "scored": int(g.scored.sum()), "safe": int(g.safe.sum()),
                    "coll": int(g.coll.sum()), "off": int(g.off.sum()), "wrong": int(g.wrong.sum())})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_models", action="store_true", help="구성만 (exp 없이)")
    a = ap.parse_args()
    if not a.no_models:
        from step14_posthoc_cells import LABEL, MODELS, SEEDS
    cache = {}
    out_sets = []
    for sid, file, title, feasible, pick, axes in SETS:
        t, keep = table(file, pick)
        ts = t[keep]
        models = {}
        if not a.no_models:
            for m, r in MODELS.items():
                fs = []
                for s in SEEDS:
                    key = (f"{r}_seed{s}", file)
                    if key not in cache:
                        cache[key] = model_rows(*key)
                    assert len(cache[key]) == len(t), key
                    fs.append(cache[key][keep].reset_index(drop=True))
                models[m] = fs
        entry = {"id": sid, "file": file, "title": title, "feasible": feasible, "n": int(len(ts)),
                 "all": {m: cells(pd.concat(fs)) for m, fs in models.items()}, "axes": []}
        for ax in axes:
            if ax not in ts:
                continue
            vals = ts[ax].fillna("none")
            types = []
            for v, n in type_order(ts, ax):
                mask = (vals == v).to_numpy()
                types.append({"id": v, "label": label(ax, v), "n": int(n),
                              "models": {m: cells(pd.concat([f[mask] for f in fs])) for m, fs in models.items()}})
            entry["axes"].append({"id": ax, "label": AXES[ax], "types": types})
        out_sets.append(entry)
        print(sid, f"{len(ts):,}", {ax["id"]: len(ax["types"]) for ax in entry["axes"]})
    meta = {"seeds": 0 if a.no_models else len(SEEDS),
            "models": [] if a.no_models else [{"id": m, "label": LABEL[m], "run": r} for m, r in MODELS.items()],
            "cells": ["ACCEPT·따름", "ACCEPT·안 따름", "REJECT·안 따름", "REJECT·따름"],
            "note": "칸의 n은 두 시드를 합친 수. 안전 = NC·DAC·DDC 모두 통과. T1 안전율은 ACCEPT 칸은 무작위 1만 개, "
                    "REJECT 칸은 거부한 결정 전부로 낸다(채점한 행만). 따름 = 경로가 결정을 따름(L1 판정 d̂ ≈ d)."}
    OUT.write_text(json.dumps({"meta": meta, "sets": out_sets}, ensure_ascii=False, indent=1))
    print("→", OUT)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    main()
