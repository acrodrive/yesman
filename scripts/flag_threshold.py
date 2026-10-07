"""10단계: 검증 세트에서 flag 기준값 p를 정하고, 판단 head의 정확도와 D0/D1의 flag 포함 지표를 낸다.

- 기준값 (14절, 결과를 보기 전에 정함): 검증 세트에서 청개구리 비율(실행할 수 있는 결정 = 원래 + CF⁺를 거부한 비율)이
  5% 이하가 되는 가장 큰 기준값 τ. p(= sigmoid(flag logit), 실행 가능 점수) < τ이면 REJECT.
  보이지 않는 원인 CF⁻는 검증 세트에서도 뺀다(학습과 같게).
- 판단 head 정확도 (검증 세트): AUC, τ에서 샘플 종류별 거부율, CF⁻ 불가능 기준별 거부율, 정확도,
  근거 head 정확도(참고, REJECT 샘플에서 가장 높은 logit의 근거가 L2 판단 근거에 들어 있는 비율).
- D0, D1 (6.2): 따르기 = ACCEPT이고 d̂ ≈ d, 언행 불일치 = ACCEPT인데 d̂ ≉ d, 청개구리 = REJECT.

사용법: python scripts/flag_threshold.py --run Ours_seed0 [--target 0.05]
출력: exp/eval/<run>/flag_threshold.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from yesman.model import REASONS

EXP = Path(os.environ["NAVSIM_EXP_ROOT"])


def auc(score, label):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    n1, n0 = label.sum(), (~label).sum()
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--target", type=float, default=0.05, help="검증 세트 청개구리 비율 상한")
    args = ap.parse_args()
    d = EXP / f"eval/{args.run}"
    val = pd.read_parquet(d / "val.parquet")
    val = val[val.visibility != "unseen"].reset_index(drop=True)
    feas = val.sample_type.isin(["original", "cf_pos"]).to_numpy()
    p = val.p.to_numpy()
    # 거부 비율(p < τ)이 target 이하인 가장 큰 τ = 실행 가능 샘플 p의 target 분위수
    tau = float(np.quantile(p[feas], args.target))
    rej = p < tau
    out = {"tau": tau, "target_frog_rate": args.target, "val_n": int(len(val)),
           "val_auc": auc(p, feas), "val_acc": float((rej != feas).mean()),
           "val_reject_rate": {st: float(rej[(val.sample_type == st).to_numpy()].mean())
                               for st in ("original", "cf_pos", "cf_neg")},
           "val_cf_neg_reject_by_category": {c: float(rej[(val.category == c).to_numpy()].mean())
                                             for c in ("road", "agent", "strength")}}
    neg = (val.sample_type == "cf_neg").to_numpy()
    logits = val[[f"reason_logit_{r}" for r in REASONS]].to_numpy()
    top = np.array(REASONS)[logits.argmax(1)]
    out["val_reason_top1_in_label"] = float(np.mean([t in str(r).split("+") for t, r in
                                                     zip(top[neg], val.reason.to_numpy()[neg])]))
    for name in ("D0", "D1"):
        f = pd.read_parquet(d / f"{name}_l1.parquet")
        acc = f.p.to_numpy() >= tau
        out[name] = {"n": int(len(f)), "follow": float((acc & f.follow).mean()),
                     "mismatch": float((acc & ~f.follow).mean()), "frog": float((~acc).mean()),
                     "follow_if_ignore_flag": float(f.follow.mean())}
    json.dump(out, open(d / "flag_threshold.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
