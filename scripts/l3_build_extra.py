"""12단계 A: 넓힌 CF(W)와 실제 VLM 결정(V)으로 학습 묶음 변형을 만든다.

입력
- 기존 묶음: exp/l3/train_bundle_{train,val}_synth.parquet (CF⁺ 목표 합성, 최종 모델과 같음)
- W: exp/l3/navtrain_sensor_cf_wide.parquet (scripts/l3_run.py --wide)
- V: exp/d3/train_{qwen8b,gemma12b}.parquet (scripts/d3_judge.py --split navtrain). 사람과 같음(6.2의 ≈)은 뺀다.
행 만들기 (6단계 l3_build_sets.py와 같은 규칙): 가능 → CF⁺(목표 = L2 최고 후보, target_rule = new_l2best, 뒤에서 합성),
불가능 → CF⁻(목표 = 사람 경로, REJECT, 판단 근거, 불가능 기준, 시야). 판정 불가는 뺀다. 검증 로그는 6단계와 같다.

단계
  python scripts/l3_build_extra.py --stage extra    # 새 행만 모은 묶음 (+ 합성에 필요한 원래 결정 행) → exp/l3/extra_{train,val}.parquet
  python scripts/l3_synth.py --input exp/l3/extra_train.parquet --out exp/l3/extra_train_synth.parquet  (val도)
  python scripts/l3_build_extra.py --stage variants  # → exp/l3/train_bundle_{train,val}_{W,V,WV}.parquet
12단계 C: python scripts/l3_build_extra.py --stage plausible  # 합성 묶음 + 그럴싸한 CF⁻(scripts/l3_plausible.py, 불가능만)
  → exp/l3/train_bundle_{train,val}_plaus.parquet (source = plausible)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from l3_build_sets import dec_cols, human_poses  # noqa: E402

ROOT = Path(os.environ["YESMAN_ROOT"])
EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
KEEP = ["token", "log_name", "sample_type", "cf_name", "change_type", "decision", "flag", "reason", "category",
        "visibility", "target_poses", "target_rule", "source"]


def to_rows(df: pd.DataFrame, source: str, hp) -> pd.DataFrame:
    df = df[df.status.isin(["feasible", "infeasible"])].copy()
    df["flag"] = np.where(df.status == "infeasible", "REJECT", "ACCEPT")
    df["sample_type"] = np.where(df.status == "infeasible", "cf_neg", "cf_pos")
    df["target_poses"] = [hp[t] if s == "infeasible" else list(b) for t, s, b in zip(df.token, df.status, df.best_poses)]
    df["target_rule"] = np.where(df.status == "infeasible", "human", "new_l2best")
    df["source"] = source
    if source.startswith("vlm"):
        df["cf_name"] = source
    for c in ("reason", "category", "visibility"):
        if c not in df:
            df[c] = ""
    return df[[c for c in KEEP + dec_cols(df) if c in df.columns]]


def load_new():
    wide = pd.read_parquet(EXP / "l3/navtrain_sensor_cf_wide.parquet")
    wide = wide[wide.cf_name != "original"]
    hp = human_poses("navtrain", wide.token.unique())
    parts = {"W": to_rows(wide, "wide", hp)}
    v = []
    for name in ("qwen8b", "gemma12b"):
        d = pd.read_parquet(EXP / f"d3/train_{name}.parquet")
        d = d[(d.parse_status == "ok") & ~d.same_as_human.fillna(False)]
        hpv = human_poses("navtrain", d.token.unique())
        v.append(to_rows(d, f"vlm_{name}", hpv))
    parts["V"] = pd.concat(v, ignore_index=True)
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["extra", "variants", "plausible"], required=True)
    args = ap.parse_args()
    val_logs = set((ROOT / "data_lists/navtrain_sensor_val_logs.txt").read_text().split())
    if args.stage == "plausible":
        pl = pd.read_parquet(EXP / "l3/navtrain_sensor_cf_plausible.parquet")
        pl = pl[pl.status == "infeasible"]
        rows = to_rows(pl, "plausible", human_poses("navtrain", pl.token.unique()))
        for part in ("train", "val"):
            base = pd.read_parquet(EXP / f"l3/train_bundle_{part}_synth.parquet").assign(source="base")
            m = rows.log_name.isin(val_logs) if part == "val" else ~rows.log_name.isin(val_logs)
            add = rows[m & rows.token.isin(set(base.token))]
            out = pd.concat([base, add], ignore_index=True)
            out.to_parquet(EXP / f"l3/train_bundle_{part}_plaus.parquet", index=False)
            print(f"{part}: +{len(add):,} plausible CF⁻ {add.category.value_counts().to_dict()}, "
                  f"unseen {(add.visibility == 'unseen').sum()}")
        return
    if args.stage == "extra":
        parts = load_new()
        new = pd.concat(parts.values(), ignore_index=True)
        for part in ("train", "val"):
            base = pd.read_parquet(EXP / f"l3/train_bundle_{part}_synth.parquet")
            m = new.log_name.isin(val_logs) if part == "val" else ~new.log_name.isin(val_logs)
            orig = base[base.sample_type == "original"]
            sub = new[m & new.token.isin(set(orig.token))]
            out = pd.concat([orig.assign(source="original"), sub], ignore_index=True)
            out.to_parquet(EXP / f"l3/extra_{part}.parquet", index=False)
            print(part, sub.groupby("source").sample_type.value_counts().unstack(fill_value=0).to_string())
        return
    for part in ("train", "val"):
        base = pd.read_parquet(EXP / f"l3/train_bundle_{part}_synth.parquet").assign(source="base")
        ex = pd.read_parquet(EXP / f"l3/extra_{part}_synth.parquet")
        ex = ex[ex.source != "original"]
        for name, srcs in (("W", ["wide"]), ("V", ["vlm_qwen8b", "vlm_gemma12b"]), ("WV", ["wide", "vlm_qwen8b", "vlm_gemma12b"])):
            add = ex[ex.source.isin(srcs)]
            out = pd.concat([base, add], ignore_index=True)
            out.to_parquet(EXP / f"l3/train_bundle_{part}_{name}.parquet", index=False)
            print(f"{part} {name}: {len(out):,} rows, {out.sample_type.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
