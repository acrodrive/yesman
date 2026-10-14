"""14단계: 검증 세트로 후보 중 하나를 고른다 (docs/step14_preregistration.md 4.5절, 5.2절).

후보마다 검증 세트(val) 예측에서 두 값을 잰다.
- 결정–경로 일치율: CF⁺ 샘플 중 ACCEPT이고 d̂ ≈ d인 비율 (판단이 없는 모델은 항상 ACCEPT)
- 불가능한 결정의 안전율: CF⁻ 샘플(보이지 않는 원인 제외) 중 그린 경로가 NC·DAC·DDC를 모두 통과한 비율
ACCEPT: 노이즈 투표 결과가 있으면 vote_accept, 판단 head가 있으면 p >= τ(flag_threshold.json), 없으면 항상 ACCEPT.

규칙
- upper (4.5절, CF 학습 구성과 Ours 설정 ①②): 상한선에 가장 가까운 후보
    점수 = w × 일치율 / (B2의 일치율) + (1 − w) × 안전율 / (B0의 안전율)
  상한선 값은 --upper_follow(B2 실행), --upper_safe(B0 실행)의 같은 검증 세트에서 잰다.
- frog5 (5.2절, 노이즈 투표 m): 실행 가능한 결정(원래 + CF⁺)을 거부하는 비율이 --frog_max 이하인 후보 중 안전율이 가장 높은 것

입력 파일 (exp/eval/<run>/): <set>_l1.parquet (follow_eval.py), <set>_scores.parquet (score_rows.py),
판단 모델은 flag_threshold.json (flag_threshold.py). 후보는 "run" 또는 "run:set" (set 기본값 val, 예: OursDiT_seed0:val_vote10_3)

사용법:
  python scripts/select_config.py --rule upper --upper_follow Bcomply_s14_seed0 --upper_safe B1_s14_seed0 \\
      --cands B3_s14_pl04_seed0 B3_s14_pl06_seed0 B3_s14_pl085_seed0 --out exp/step14/select_B3_stage2a.json
  python scripts/select_config.py --rule frog5 --cands OursDiT_seed0:val_vote10_1 OursDiT_seed0:val_vote10_2 ...
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
SAFE = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance"]


def parse(c: str):
    run, _, s = c.partition(":")
    return run, (s or "val")


def accept(f: pd.DataFrame, run: str) -> np.ndarray:
    if "vote_accept" in f:
        return f.vote_accept.to_numpy(bool)
    th = EXP / f"eval/{run}/flag_threshold.json"
    if "p" in f and th.exists():
        return f.p.to_numpy() >= json.load(open(th))["tau"]
    return np.ones(len(f), bool)


def measure(run: str, s: str) -> dict:
    f = pd.read_parquet(EXP / f"eval/{run}/{s}_l1.parquet")
    sc = pd.read_parquet(EXP / f"eval/{run}/{s}_scores.parquet")
    assert len(f) == len(sc), (run, s, len(f), len(sc))
    acc = accept(f, run)
    st = f.sample_type.to_numpy()
    pos = st == "cf_pos"
    neg = (st == "cf_neg") & (f.visibility.fillna("").to_numpy() != "unseen")
    feasible = (st == "original") | pos
    safe = (sc[SAFE].min(axis=1) >= 1).to_numpy()
    return {"run": run, "set": s, "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
            "follow": float((acc & f.follow.to_numpy(bool))[pos].mean()),
            "safe": float(safe[neg].mean()),
            "frog": float((~acc)[feasible].mean()),
            "reject_neg": float((~acc)[neg].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", choices=["upper", "frog5"], required=True)
    ap.add_argument("--cands", nargs="+", required=True)
    ap.add_argument("--upper_follow", default="", help="upper: 일치율 상한선을 잴 실행 (B2, 코드 이름 Bcomply_*)")
    ap.add_argument("--upper_safe", default="", help="upper: 안전율 상한선을 잴 실행 (B0, 코드 이름 B1_*)")
    ap.add_argument("--w", type=float, default=0.5, help="upper: 일치율 쪽 무게 (사전 등록에서 확정)")
    ap.add_argument("--frog_max", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = [measure(*parse(c)) for c in args.cands]
    res = {"rule": args.rule, "cands": rows}
    if args.rule == "upper":
        assert args.upper_follow and args.upper_safe, "upper 규칙은 --upper_follow와 --upper_safe가 필요하다"
        uf, us = measure(*parse(args.upper_follow))["follow"], measure(*parse(args.upper_safe))["safe"]
        res.update(upper_follow=uf, upper_safe=us, w=args.w)
        for r in rows:
            r["score"] = args.w * r["follow"] / uf + (1 - args.w) * r["safe"] / us
        best = max(rows, key=lambda r: r["score"])
    else:
        ok = [r for r in rows if r["frog"] <= args.frog_max]
        best = max(ok, key=lambda r: r["safe"]) if ok else None
        res["frog_max"] = args.frog_max
    res["best"] = None if best is None else f"{best['run']}:{best['set']}"

    cols = ["run", "set", "follow", "safe", "frog", "reject_neg"] + (["score"] if args.rule == "upper" else [])
    print(pd.DataFrame(rows)[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if args.rule == "upper":
        print(f"상한선: 일치율 {res['upper_follow']:.4f} ({args.upper_follow}), 안전율 {res['upper_safe']:.4f} ({args.upper_safe})")
    print("고른 후보:", res["best"] or "없음 (조건을 만족하는 후보가 없다)")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
