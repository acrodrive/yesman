"""14단계 평가 (docs/step14_preregistration.md 7절). 미리 정한 지표와 비교만 낸다.

입력 (exp/eval/<run>_seed<s>/): <set>_l1.parquet (follow_eval.py), <set>_scores.parquet (score_rows.py),
판단 모델은 flag_threshold.json(τ, 검증 세트에서 정함). 노이즈 투표 결과는 세트 이름 뒤에 _vote<K>_<M>이 붙은 파일.
세트: T0(사람 결정), T1(CF⁺), T2(CF⁻), T3_gemma31b(VLM 결정, 실행 불가능한 것만 안전율에 씀). --sets로 val 등도 가능.

지표 (모델마다 시드 평균)
  follow       결정–경로 일치율: T1에서 ACCEPT이고 d̂ ≈ d인 비율
  over_reject  과잉 거부: 실행 가능한 결정(T0 + T1)을 거부한 비율
  safe         불가능한 결정의 안전율: 실행 불가능한 결정(T2 주 결과, T3의 L2 불가능)에서 그린 경로가 NC·DAC·DDC를 모두 통과한 비율
  safe_<기준>  불가능한 이유별(road / agent / strength) 안전율
  flag_path_mismatch  flag–경로 불일치(판단 모델만): ACCEPT인데 d̂ ≉ d이거나 REJECT인데 d̂ ≈ d인 비율 (T1 + T2)
  auc_kind     결정 종류 안 AUC(판단 모델만): T1 ∪ T2에서 결정 종류(yesman/decision.py decision_kind)마다 실행 가능/불가능을
               거부 점수로 가르는 AUC를 개수로 가중 평균. 거부 점수 = 1 − p (투표면 REJECT 비율)
  auc_all      전체 AUC (보조)
  vote_flip    투표 모델: 노이즈마다 판단이 갈린 결정의 비율, 그리고 갈린 결정과 만장일치 결정의 L2 통과 후보 비율(경계 정도)
비교: 짝지은 차이 = 같은 결정 위에서 (시드 평균한 결정별 값)의 차이, 결정 단위로 5,000번 다시 뽑은 95% 구간.
주 비교의 Ours는 노이즈 1개이다. 성공 판정(7.1절)은 비교마다 따로 하고, 여러 번 비교한 보정은 하지 않는다.
  Ours − B2 safe > 0, Ours − B3 safe > −5%p, Ours − B3 follow > −2%p (구간의 아래끝),
  Ours − OursPrev flag–경로 불일치 < 0 (구간의 위끝). 구간이 기준선에 걸치면 "결론 없음".
OursPrev = 이전 Ours (기존 Ours, 판단 head가 DiT와 병렬. 사다리 밖의 ablation, Ours가 고른 CF 학습 구성으로 학습)

사용법: python scripts/step14_eval.py --models B0=B1_s14 B1=B2_s14 B2=Bcomply_s14 B3=B3_s14 OursPrev=Ours_s14 Ours=OursDiT_s14 \\
            --vote Ours=vote10_3 --out logs/step14/eval.md
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from yesman.decision import decision_kind

EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
SAFE = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance"]
SEEDS = (0, 1)
COMPARE = [("Ours", "B2", "safe"), ("Ours", "B3", "safe"), ("Ours", "B3", "follow"),
           ("Ours", "OursPrev", "flag_path_mismatch"), ("Ours", "OursPrev", "auc_kind")]
# 성공 기준 (7.1절): (a, b, col) → (구간의 어느 끝, 기준선). lo: 아래끝 > 기준선, hi: 위끝 < 기준선
SUCCESS = {("Ours", "B2", "safe"): ("lo", 0.0), ("Ours", "B3", "safe"): ("lo", -0.05),
           ("Ours", "B3", "follow"): ("lo", -0.02), ("Ours", "OursPrev", "flag_path_mismatch"): ("hi", 0.0)}


def auc(score, label):
    if label.all() or not label.any():
        return np.nan
    r = pd.Series(score).rank().to_numpy()
    n1, n0 = label.sum(), (~label).sum()
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def load(run, s, vote=""):
    name = f"{s}_{vote}" if vote else s
    d = EXP / f"eval/{run}"
    f = pd.read_parquet(d / f"{name}_l1.parquet")
    sc_path = d / f"{name}_scores.parquet"
    if sc_path.exists():
        f["safe_path"] = (pd.read_parquet(sc_path)[SAFE].min(axis=1) >= 1).to_numpy()
    th = d / "flag_threshold.json"
    if "vote_accept" in f:
        f["accept"] = f.vote_accept.astype(bool)
        f["reject_score"] = f.vote_n_reject / int(vote.split("_")[0][4:])
    elif "p" in f and th.exists():
        f["accept"] = f.p >= json.load(open(th))["tau"]
        f["reject_score"] = 1 - f.p
    else:
        f["accept"] = True
    return f


def per_decision(run, vote, sets):
    """결정마다 시드 평균한 값 (짝지은 비교에 쓴다)."""
    out = {}
    for s in sets:
        fs = [load(f"{run}_seed{k}", s, vote) for k in SEEDS]
        base = fs[0][["token", "cf_name"] + [c for c in ("category", "visibility", "main", "status", "sample_type")
                                              if c in fs[0]]].copy()
        base["follow"] = np.mean([(f.accept & f.follow).to_numpy() for f in fs], 0)
        base["reject"] = np.mean([(~f.accept).to_numpy() for f in fs], 0)
        base["mismatch"] = np.mean([((f.accept & ~f.follow) | (~f.accept & f.follow)).to_numpy() for f in fs], 0)
        if "safe_path" in fs[0]:
            base["safe"] = np.mean([f.safe_path.to_numpy() for f in fs], 0)
        if "reject_score" in fs[0]:
            base["reject_score"] = np.mean([f.reject_score.to_numpy() for f in fs], 0)
        if "vote_n_reject" in fs[0]:
            k = int(vote.split("_")[0][4:])
            base["flip"] = np.mean([((f.vote_n_reject > 0) & (f.vote_n_reject < k)).to_numpy() for f in fs], 0)
        out[s] = base
    return out


def infeasible_rows(pd_, s):
    t = pd_[s]
    if s.startswith("T2"):  # 주 결과: 보이지 않는 원인 제외
        return t[t.visibility != "unseen"] if "visibility" in t else t
    if s.startswith("T3"):
        return t[t.status == "infeasible"]
    return t.iloc[:0]


def metrics(pd_, eval_sets, judge):
    m = {}
    if "T1" in pd_:
        m["follow"] = pd_["T1"].follow.mean()
    feas = [pd_[s] for s in ("T0", "T1") if s in pd_]
    if feas:
        m["over_reject"] = pd.concat(feas).reject.mean()
    for s in eval_sets:
        bad = infeasible_rows(pd_, s)
        if len(bad) and "safe" in bad:
            m[f"safe_{s}"] = bad.safe.mean()
            if "category" in bad:
                for c, g in bad.groupby("category"):
                    m[f"safe_{s}_{c}"] = g.safe.mean()
    m["safe"] = m.get("safe_T3_gemma31b", m.get("safe_T2", np.nan))  # 주 지표: 처음 보는 VLM(Gemma 4 31B)
    if judge and "T1" in pd_ and "T2" in pd_:
        both = pd.concat([pd_["T1"].assign(bad=False), pd_["T2"].assign(bad=True)], ignore_index=True)
        m["flag_path_mismatch"] = both.mismatch.mean()
        if "reject_score" in both:
            m["auc_all"] = auc(both.reject_score.to_numpy(), both.bad.to_numpy())
            src = pd.concat([pd.read_parquet(Path(os.environ["YESMAN_ROOT"]) / f"data_lists/eval_step14/{s}.parquet")
                             for s in ("T1", "T2")], ignore_index=True)
            both["kind"] = decision_kind(src)
            a = [(auc(g.reject_score.to_numpy(), g.bad.to_numpy()), len(g)) for _, g in both.groupby("kind")]
            a = [(x, n) for x, n in a if np.isfinite(x)]
            m["auc_kind"] = sum(x * n for x, n in a) / sum(n for _, n in a) if a else np.nan
    if "flip" in pd_.get("T1", pd.DataFrame()):
        t1 = pd_["T1"].merge(pd.read_parquet(Path(os.environ["YESMAN_ROOT"]) / "data_lists/eval_step14/T1.parquet",
                                             columns=["token", "cf_name", "n_pass", "n_scored"]), on=["token", "cf_name"])
        t1["pass_ratio"] = t1.n_pass / t1.n_scored
        m["vote_flip"] = t1.flip.mean()
        m["pass_ratio_flipped"] = t1[t1.flip > 0].pass_ratio.mean()
        m["pass_ratio_unanimous"] = t1[t1.flip == 0].pass_ratio.mean()
    return m


def paired(a, b, col, n_boot=5000, seed=0):
    x = a.merge(b, on=["token", "cf_name"], suffixes=("_a", "_b"))
    d = (x[f"{col}_a"] - x[f"{col}_b"]).to_numpy()
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return d.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5), len(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="문서이름=코드 실행 이름 앞부분 (뒤에 _seed0, _seed1)")
    ap.add_argument("--vote", nargs="*", default=[], help="문서이름=vote<K>_<M> (노이즈 투표 결과로 평가)")
    ap.add_argument("--judge", nargs="*", default=["OursPrev", "Ours"], help="판단 head가 있는 모델")
    ap.add_argument("--sets", nargs="+", default=["T0", "T1", "T2", "T3_gemma31b"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    runs = dict(x.split("=", 1) for x in args.models)
    votes = dict(x.split("=", 1) for x in args.vote)
    variants = [(m, "") for m in runs] + [(f"{m}+{v}", v) for m, v in votes.items()]
    R, PD = {}, {}
    for name, vote in variants:
        base = name.split("+")[0]
        PD[name] = per_decision(runs[base], vote, args.sets)
        R[name] = metrics(PD[name], [s for s in args.sets if s.startswith(("T2", "T3"))], base in args.judge)
    lines = ["# 14단계 평가 (새 테스트 세트, 시드 평균)\n", pd.DataFrame(R).T.to_string(float_format=lambda x: f"{x:.4f}"), ""]
    lines.append("## 미리 정한 비교 (짝지은 차이, 95% 구간)\n")
    for a, b, col in COMPARE + [(f"Ours+{votes['Ours']}", "Ours", "safe"), (f"Ours+{votes['Ours']}", "Ours", "follow")] \
            if "Ours" in votes else COMPARE:
        if a not in PD or b not in PD:
            continue
        crit = SUCCESS.get((a, b, col))
        if col in ("safe",):
            s = "T3_gemma31b" if "T3_gemma31b" in args.sets else "T2"
            pa, pb = infeasible_rows(PD[a], s), infeasible_rows(PD[b], s)
        elif col in ("follow",):
            pa, pb = PD[a]["T1"], PD[b]["T1"]
        elif col == "flag_path_mismatch":
            pa, pb = [pd.concat([PD[x]["T1"], PD[x]["T2"]]) for x in (a, b)]
            col = "mismatch"
        else:  # AUC는 짝지은 다시 뽑기 대신 값만 비교한다
            lines.append(f"- {a} vs {b} {col}: {R[a].get(col, np.nan):.4f} vs {R[b].get(col, np.nan):.4f}")
            continue
        mean, lo, hi, n = paired(pa, pb, col)
        verdict = ""
        if crit:
            end, line = crit
            ok = lo > line if end == "lo" else hi < line
            over = hi < line if end == "lo" else lo > line  # 구간 전체가 기준선의 반대쪽
            verdict = f" → 기준 {'아래끝 >' if end == 'lo' else '위끝 <'} {line:+.2f}: " + \
                ("성공" if ok else "실패" if over else "결론 없음")
        lines.append(f"- {a} − {b} {col}: {mean:+.4f} [{lo:+.4f}, {hi:+.4f}] (n = {n:,}){verdict}")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)


if __name__ == "__main__":
    main()
