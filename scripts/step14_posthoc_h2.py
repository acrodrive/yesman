"""14단계 사후 분석 (2026-10-20, 탐색, navtest D3x): 판단 head가 "경로 생성기가 속는 결정"을 못 잡는 이유가 입력 정보인가.

T3에서 head는 경로 생성기(B3)가 그대로 따른 실행 불가능 결정을 거의 거부하지 않았다(Ours 7~13%, 이전 Ours 16~18%).
12단계 B에서 정답 물체 위치·속도를 입력으로 받은 Ours가 있으므로, 이 정보가 있으면 그런 결정을 더 잡는지 본다.
  어려운 결정 = 기준 경로 생성기(13단계 B3_syn, 판단 없음, 같은 시드)가 그대로 따른(d̂ ≈ d) 실행 불가능 결정
  쉬운 결정   = 기준 경로 생성기가 피한 실행 불가능 결정
모델: 12단계 B 사다리(Ours_syn / +LTF 검출 / +GT 위치 / +GT 위치·속도)와 2x2(WV / +강도 무게 / +GT 위치·속도 / 둘 다).
세트: navtest D3x 세 VLM(실행 불가능 = L2 불가능이고 사람 결정과 다름). 이미 낸 예측과 τ를 다시 세기만 한다.

사용법: python scripts/step14_posthoc_h2.py
출력: 표준 출력(markdown 표)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from step14_eval import auc  # noqa: E402

ROOT = Path(os.environ["YESMAN_ROOT"])
E = Path(os.environ["NAVSIM_EXP_ROOT"]) / "eval"
VLMS = ["gemma31b", "gemma12b", "qwen8b"]
REF = "B3_syn"
MODELS = [("사다리", "Ours", "Ours_syn"), ("사다리", "+ LTF 검출", "Ours_objltf"), ("사다리", "+ GT 위치", "Ours_objgt"),
          ("사다리", "+ GT 위치·속도", "Ours_objgtvel"), ("2x2", "① WV", "Ours_WV"), ("2x2", "② + 강도 무게", "Ours_WVbal"),
          ("2x2", "③ + GT 위치·속도", "Ours_WVgtvel"), ("2x2", "④ 둘 다", "Ours_WVbalgtvel")]
SEEDS = (0, 1)


def rows(run, v):
    """D3x 한 VLM의 (상태, p, 경로가 따름). 예측 파일은 parse_status = ok 행과 같은 순서."""
    t = pd.read_parquet(ROOT / f"data_lists/eval/D3x_{v}.parquet", columns=["token", "status", "same_as_human"])
    l1 = pd.read_parquet(E / run / f"D3x_{v}_l1.parquet")
    assert (l1.token.to_numpy() == t.token.to_numpy()).all(), (run, v)
    inf = ((t.status == "infeasible") & ~t.same_as_human.fillna(False)).to_numpy()
    fea = (t.status == "feasible").to_numpy()
    p = l1.p.to_numpy(float) if "p" in l1 else np.ones(len(l1))  # 판단이 없는 모델은 늘 ACCEPT
    return inf, fea, p, l1.follow.to_numpy(bool)


def stats(run, seed, vlms):
    tau = json.load(open(E / f"{run}_seed{seed}" / "flag_threshold.json"))["tau"]
    hard, easy, fea_rej, sc, lab = [], [], [], [], []
    for v in vlms:
        inf, fea, p, _ = rows(f"{run}_seed{seed}", v)
        _, _, _, ref_follow = rows(f"{REF}_seed{seed}", v)
        rej = p < tau
        hard.append(rej[inf & ref_follow])
        easy.append(rej[inf & ~ref_follow])
        fea_rej.append(rej[fea])
        m = inf | fea
        sc.append(1 - p[m])
        lab.append(inf[m])
    c = np.concatenate
    return dict(hard=c(hard).mean(), n_hard=len(c(hard)), easy=c(easy).mean(), n_easy=len(c(easy)),
                fea=c(fea_rej).mean(), auc=auc(c(sc), c(lab)))


def main():
    for title, vlms in (("Gemma 4 31B (학습에 안 씀)", ["gemma31b"]), ("세 VLM 합침", VLMS)):
        print(f"\n### {title}\n")
        print("| 묶음 | 모델 | 어려운 결정 거부 (n) | 쉬운 결정 거부 (n) | 실행 가능 거부 | AUC |")
        print("| --- | --- | --- | --- | --- | --- |")
        for group, label, run in MODELS:
            s = [stats(run, seed, vlms) for seed in SEEDS]
            m = {k: np.mean([x[k] for x in s]) for k in s[0]}
            print(f"| {group} | {label} | {m['hard']:.1%} ({m['n_hard']:.0f}) | {m['easy']:.1%} ({m['n_easy']:.0f}) | "
                  f"{m['fea']:.1%} | {m['auc']:.3f} |")


if __name__ == "__main__":
    main()
