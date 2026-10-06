"""4단계: L1 특징 표에 기준값(yesman/l1_thresholds.yaml)을 적용하여 결정 라벨을 붙이고, 결정 분포 표를 낸다.

앞뒤 행동 값은 저장된 사람 경로(poses)와 현재 속도(v0)에서 다시 계산한다(특징 추출 뒤에 바뀐 속도 계산 방식을 반영).

사용법: python scripts/l1_label.py --split navtrain   # → exp/l1/navtrain_labels.parquet, 분포 표는 stdout (markdown)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from yesman.l1 import classify, load_thresholds, motion_features

EXP = Path(os.environ["NAVSIM_EXP_ROOT"])


def label_table(df: pd.DataFrame) -> pd.DataFrame:
    th = load_thresholds()
    rows = []
    for r in df.to_dict("records"):
        if r.get("valid") == 1:
            r.update(motion_features(np.asarray(r["poses"]), r["v0"]))
        rows.append({"token": r["token"], "log_name": r["log_name"], **classify(r, th).to_flat()})
    return pd.DataFrame(rows)


def pct(s: pd.Series) -> str:
    return f"{s.mean() * 100:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="navtrain")
    args = ap.parse_args()
    feats = pd.read_parquet(EXP / f"l1/{args.split}_features.parquet")
    lab = label_table(feats)
    out = EXP / f"l1/{args.split}_labels.parquet"
    lab.to_parquet(out, index=False)

    n = len(lab)
    v = lab[lab.valid]
    print(f"## {args.split} L1 결정 분포\n")
    print(f"- 장면 {n:,}개, 결정을 정할 수 있는 장면(valid) {len(v):,}개 ({len(v) / n * 100:.1f}%)")
    print(f"- 애매함이 하나도 없는 장면 {(~v.ambiguous).sum():,}개 ({(~v.ambiguous).mean() * 100:.1f}% of valid)\n")
    for k in (1, 2):
        print(f"### 구간 {k} ({(k - 1) * 2}~{k * 2}초)\n")
        print("| 앞뒤 행동 | 비율 | 세기 중앙값 | 애매함 |")
        print("| --- | --- | --- | --- |")
        for a, g in v.groupby(f"seg{k}_lon"):
            print(f"| {a} | {len(g) / len(v) * 100:.1f}% | {g[f'seg{k}_lon_strength'].median():.2f} | "
                  f"{pct(g[f'seg{k}_ambiguous_lon'])} |")
        print("\n| 좌우 행동 | 비율 | 개수 | 세기 중앙값 | 애매함 |")
        print("| --- | --- | --- | --- | --- |")
        for a, g in v.groupby(f"seg{k}_lat"):
            print(f"| {a} | {len(g) / len(v) * 100:.2f}% | {len(g):,} | {g[f'seg{k}_lat_strength'].median():.2f} | "
                  f"{pct(g[f'seg{k}_ambiguous_lat'])} |")
        print()
    print("### 애매함 이유 (valid 장면)\n")
    reasons = v.reason[v.reason != ""].str.split(";").explode().str.split(":").str[1].value_counts()
    print("| 이유 | 구간 수 |\n| --- | --- |")
    for r, c in reasons.items():
        print(f"| {r} | {c:,} |")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
