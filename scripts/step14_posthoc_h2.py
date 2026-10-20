"""14단계 사후 분석 (2026-10-20, 탐색, navtest D3x): 판단 head가 "경로 생성기가 속는 결정"을 못 잡는 이유가 입력 정보인가.

T3에서 head는 경로 생성기(B3)가 그대로 따른 실행 불가능 결정을 거의 거부하지 않았다(Ours 7~13%, 이전 Ours 16~18%).
12단계 B에서 정답 물체 위치·속도를 입력으로 받은 Ours가 있으므로, 이 정보가 있으면 그런 결정을 더 잡는지 본다.
  어려운 결정 = 기준 경로 생성기(13단계 B3_syn, 판단 없음, 같은 시드)가 그대로 따른(d̂ ≈ d) 실행 불가능 결정
  쉬운 결정   = 기준 경로 생성기가 피한 실행 불가능 결정
모델: 12단계 B 사다리(Ours_syn / +LTF 검출 / +GT 위치 / +GT 위치·속도)와 2x2(WV / +강도 무게 / +GT 위치·속도 / 둘 다).
세트: navtest D3x 세 VLM(실행 불가능 = L2 불가능이고 사람 결정과 다름). 이미 낸 예측과 τ를 다시 세기만 한다.

함께 내는 것(테스트 T3를 다시 나누기만 함): 판단 REJECT와 "경로가 결정을 안 따름"의 정밀도, B3가 따른 / 피한 결정별 판단 거부율,
불가능 이유별 네 칸 비율과 안전하지 못한 경로 중 ACCEPT·일치의 몫.

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


def t3_tables():
    """테스트 T3 (Gemma 4 31B) 다시 나누기. 판단 모델 두 개 x 시드 0·1, 시드 평균."""
    import step14_posthoc_judge as J
    from step14_eval import load
    t3, t2, val = J.sets()
    inf = (t3.status == "infeasible").to_numpy()
    judged = t3.status.isin(["feasible", "infeasible"]).to_numpy()
    print("\n### T3 신호의 정밀도 (판정된 결정만)\n")
    print("| 모델 | 신호 | 불가능을 잡음 | 가능을 잘못 잡음 | 정밀도 |")
    print("| --- | --- | --- | --- | --- |")
    cells_md, b3_md = [], []
    for m, run in J.MODELS.items():
        fs = J.frames(run, t3, t2, val)
        sig, b3r, cells = {}, [], {}
        for s, f in zip(J.SEEDS, fs):
            t = f["T3"]
            rej = ~t.accept.to_numpy(bool)
            nof = ~t.follow.to_numpy(bool)
            for name, flag in (("head REJECT", rej), ("경로가 결정을 안 따름", nof), ("둘 중 하나", rej | nof)):
                sig.setdefault(name, []).append((flag[inf].mean(), flag[judged & ~inf].mean(), inf[flag & judged].mean()))
            b3 = load(f"{J.B3}_seed{s}", "T3_gemma31b")
            assert (b3.token.to_numpy() == t3.token.to_numpy()).all()
            bf = b3.follow.to_numpy(bool)
            b3r.append((rej[inf & bf].mean(), rej[inf & ~bf].mean(), (inf & bf).sum(), (inf & ~bf).sum()))
            for cat in ("road", "agent", "strength"):
                k = inf & (t3.category == cat).to_numpy()
                a, fo, sf = ~rej[k], ~nof[k], t.safe_path.to_numpy(bool)[k]
                cells.setdefault(cat, []).append(((~a & ~fo).mean(), (~a & fo).mean(), (a & fo).mean(), (a & ~fo).mean(),
                                                  (a & fo & ~sf).sum() / max((~sf).sum(), 1), k.sum()))
        for name, v in sig.items():
            v = np.mean(v, 0)
            print(f"| {J.LABEL[m]} | {name} | {v[0]:.1%} | {v[1]:.1%} | {v[2]:.1%} |")
        b3_md.append(f"| {J.LABEL[m]} | " + " / ".join(f"{x[0]:.1%}" for x in b3r) + f" ({np.mean([x[2] for x in b3r]):.0f}) | "
                     + " / ".join(f"{x[1]:.1%}" for x in b3r) + f" ({np.mean([x[3] for x in b3r]):.0f}) |")
        for cat, v in cells.items():
            v = np.mean(v, 0)
            cells_md.append(f"| {J.LABEL[m]} | {cat} ({v[5]:.0f}) | {v[0]:.1%} | {v[1]:.1%} | {v[2]:.1%} | {v[3]:.1%} | {v[4]:.1%} |")
    print("\n### T3 실행 불가능 결정: B3(같은 시드) 경로가 따른 / 피한 결정별 판단 거부율 (시드 0 / 1)\n")
    print("| 모델 | B3가 따른 결정 (n) | B3가 피한 결정 (n) |")
    print("| --- | --- | --- |")
    print("\n".join(b3_md))
    print("\n### T3 실행 불가능 결정: 불가능 이유별 네 칸\n")
    print("| 모델 | 이유 (n) | REJECT·안 따름 | REJECT·따름 | ACCEPT·따름 | ACCEPT·안 따름 | 안전하지 못한 경로 중 ACCEPT·따름 |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    print("\n".join(cells_md))


PAIRS = [("Ours_syn", "Ours_objgtvel"), ("Ours_WV", "Ours_WVgtvel"), ("Ours_WV", "Ours_WVbalgtvel"),
         ("Ours_syn", "Ours_WV")]


def hard_reject(run, vlms):
    """결정별 어려운 결정 거부 (두 시드 평균, 한 시드에서만 어려우면 그 시드 값, 어디서도 아니면 NaN)."""
    out = []
    for s in SEEDS:
        tau = json.load(open(E / f"{run}_seed{s}" / "flag_threshold.json"))["tau"]
        r = []
        for v in vlms:
            inf, _, p, _ = rows(f"{run}_seed{s}", v)
            _, _, _, ref_follow = rows(f"{REF}_seed{s}", v)
            r.append(np.where(inf & ref_follow, (p < tau).astype(float), np.nan))
        out.append(np.concatenate(r))
    with np.errstate(all="ignore"):
        return np.nanmean(np.stack(out), 0)


def main():
    t3_tables()
    rng = np.random.default_rng(0)
    for title, vlms in (("Gemma 4 31B (학습에 안 씀)", ["gemma31b"]), ("세 VLM 합침", VLMS)):
        print(f"\n### {title}\n")
        print("| 묶음 | 모델 | 어려운 결정 거부 (n) | 쉬운 결정 거부 (n) | 실행 가능 거부 | AUC |")
        print("| --- | --- | --- | --- | --- | --- |")
        for group, label, run in MODELS:
            s = [stats(run, seed, vlms) for seed in SEEDS]
            m = {k: np.mean([x[k] for x in s]) for k in s[0]}
            print(f"| {group} | {label} | {m['hard']:.1%} ({m['n_hard']:.0f}) | {m['easy']:.1%} ({m['n_easy']:.0f}) | "
                  f"{m['fea']:.1%} | {m['auc']:.3f} |")
        print("\n| 어려운 결정 거부 짝 비교 (결정별 시드 평균, 부트스트랩 95% 구간 5,000번) | 차이 | n |")
        print("| --- | --- | --- |")
        for a, b in PAIRS:
            x, y = hard_reject(a, vlms), hard_reject(b, vlms)
            k = ~np.isnan(x) & ~np.isnan(y)
            d = (y - x)[k]
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
            print(f"| {a} → {b} | {100 * d.mean():+.1f}%p [{100 * np.percentile(bs, 2.5):+.1f}, "
                  f"{100 * np.percentile(bs, 97.5):+.1f}] | {k.sum()} |")


if __name__ == "__main__":
    main()
