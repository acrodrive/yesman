"""학습 묶음의 CF⁺·CF⁻를 유형별로 센다 (14단계 기여 분석, 보고서 표).

유형 = 원래 결정에서 바꾼 것(change_type의 앞부분: 좌우 행동 / 앞뒤 행동 / 강도와 그 조합). CF⁻는 실패 이유도 센다.
13단계 묶음(train_bundle_train_synth)과 14단계 묶음(train_bundle_train_step14_plaus, 출처별)을 나란히 놓는다.
보이지 않는 원인 CF⁻는 학습에 쓰지 않으므로 뺀다.

사용법: python scripts/step14_bundle_types.py [--out exp/step14/contrib/bundle_types.md]
"""

import argparse
import os
from pathlib import Path

import pandas as pd

EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
KIND = {"lat_type": "좌우", "lon_type": "앞뒤", "strength": "강도"}
KINDS = ["좌우", "강도", "앞뒤", "좌우 + 강도", "앞뒤 + 강도", "좌우 + 앞뒤", "좌우 + 앞뒤 + 강도"]
REASON = {"road": "도로 모양", "agent": "다른 차·보행자", "strength": "강도 기준"}


def load(name):
    cols = ["token", "sample_type", "change_type", "category", "visibility"]
    b = pd.read_parquet(EXP / f"l3/train_bundle_{name}.parquet",
                        columns=cols + (["source"] if "step14" in name else []))
    b = b[b.visibility != "unseen"].copy()
    b["kind"] = b.change_type.str.split("/").str[0].map(lambda s: " + ".join(KIND.get(x, x) for x in s.split("+")))
    if "source" not in b:
        b["source"] = "menu"
    return b


def fmt(n):
    return f"{int(n):,}" if n else "0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    a, b = load("train_synth"), load("train_step14_plaus")
    cols = [("13단계", a, None), ("14단계 메뉴", b, "menu"), ("14단계 확장", b, "wide"), ("14단계 plausible", b, "plausible"),
            ("14단계 합계", b, None)]
    out = ["# 학습 묶음의 CF 유형별 개수 (보이지 않는 원인 CF⁻ 제외)\n",
           f"학습 장면(원래 결정): 13단계 {a.token[a.sample_type == 'original'].nunique():,}, "
           f"14단계 {b.token[b.sample_type == 'original'].nunique():,}\n"]
    for st, title in (("cf_pos", "CF⁺ (실행 가능한 바꾼 결정)"), ("cf_neg", "CF⁻ (실행할 수 없는 바꾼 결정)")):
        out += [f"\n## {title}: 바꾼 것별\n", "| 바꾼 것 | " + " | ".join(c[0] for c in cols) + " |", "|" + " --- |" * (len(cols) + 1)]
        for k in KINDS + ["합계"]:
            row = []
            for _, d, src in cols:
                x = d[(d.sample_type == st) & ((d.source == src) if src else True)]
                row.append(fmt(len(x) if k == "합계" else (x.kind == k).sum()))
            out.append(f"| {k} | " + " | ".join(row) + " |")
    out += ["\n## CF⁻: 실패 이유별\n", "| 실패 이유 | " + " | ".join(c[0] for c in cols) + " |", "|" + " --- |" * (len(cols) + 1)]
    for r, name in REASON.items():
        row = [fmt(((d.sample_type == "cf_neg") & (d.category == r) & ((d.source == src) if src else True)).sum())
               for _, d, src in cols]
        out.append(f"| {name} | " + " | ".join(row) + " |")
    text = "\n".join(out) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
