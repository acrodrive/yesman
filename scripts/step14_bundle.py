"""14단계: 학습 묶음을 만든다 (docs/step14_preregistration.md 4.1절, 1단계 양 늘리기).

입력 (scripts/run_step14_data.sh가 만든다. 모두 학습 + 검증 장면 = data_lists/step14/train_all_tokens.txt):
  exp/l3/step14_train_cf.parquet            규칙 기반 CF 메뉴 전부 (l3_run.py --skip_original)
  exp/l3/step14_train_cf_wide.parquet       확장된 규칙 기반 CF (l3_run.py --wide --skip_original)
  exp/l3/step14_train_cf_plausible.parquet  plausible CF (l3_plausible.py, 실행 가능한 것과 불가능한 것 모두)
행 만들기 (6단계와 같은 규칙):
  원래 결정 → target = 사람 경로, ACCEPT
  실행 가능 → CF⁺, target = 통과한 후보 중 사람 경로에 가장 가까운 것(closest_poses, 10b단계 규칙),
              target_rule = new_closest. 뒤에서 l3_synth.py가 사람 경로를 고친 경로로 바꾼다(실패하면 그대로)
  실행 불가능 → CF⁻, target = 사람 경로, REJECT, 판단 근거와 불가능 기준
  판정 불가는 뺀다. 열 source = original / menu / wide / plausible, shard = 0(기존 조각) 또는 학습 추가 조각 번호
검증 = data_lists/step14/val_logs.txt의 로그.

사용법: python scripts/step14_bundle.py
출력: exp/l3/step14_bundle_{train,val}_raw.parquet → l3_synth.py --input ... --out exp/l3/train_bundle_{train,val}_step14.parquet
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from l3_build_sets import dec_cols, human_poses  # noqa: E402

ROOT = Path(os.environ["YESMAN_ROOT"])
EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
S14 = ROOT / "data_lists/step14"
KEEP = ["token", "log_name", "sample_type", "cf_name", "change_type", "decision", "flag", "reason", "category",
        "visibility", "target_poses", "target_rule", "source", "shard"]


def cf_rows(df: pd.DataFrame, source: str, hp: pd.Series) -> pd.DataFrame:
    df = df[(df.cf_name != "original") & df.status.isin(["feasible", "infeasible"])].copy()
    neg = (df.status == "infeasible").to_numpy()
    df["flag"] = np.where(neg, "REJECT", "ACCEPT")
    df["sample_type"] = np.where(neg, "cf_neg", "cf_pos")
    close = df["closest_poses"] if "closest_poses" in df else pd.Series([[]] * len(df), index=df.index)
    df["target_poses"] = [hp[t] if n else (list(c) if c is not None and len(c) == 24 else list(b))
                          for t, n, c, b in zip(df.token, neg, close, df.best_poses)]
    df["target_rule"] = np.where(neg, "human",
                                 ["new_closest" if c is not None and len(c) == 24 else "new_l2best" for c in close])
    df["source"] = source
    for c in ("reason", "category", "visibility"):
        if c not in df:
            df[c] = ""
    return df


def main():
    menu = pd.read_parquet(EXP / "l3/step14_train_cf.parquet")
    wide = pd.read_parquet(EXP / "l3/step14_train_cf_wide.parquet")
    pl = pd.read_parquet(EXP / "l3/step14_train_cf_plausible.parquet")
    tokens = sorted(set(menu.token))
    hp = human_poses("navtrain", tokens)
    orig = menu[menu.cf_name == "original"].drop_duplicates("token").copy()
    orig["flag"], orig["sample_type"], orig["source"], orig["target_rule"] = "ACCEPT", "original", "original", "human"
    orig["target_poses"] = [hp[t] for t in orig.token]
    for c in ("reason", "category", "visibility"):
        orig[c] = ""
    # 원래 결정 행이 있는 장면만 쓴다(target 경로를 고칠 때 필요). 원래 결정이 판정 불가라 메뉴 CF가 없는 장면(약 3%)도
    # plausible CF는 만들어지므로, target 경로를 정하기 전에 뺀다 (14단계: 그 장면의 사람 경로는 hp에 없다)
    keep = set(orig.token)
    menu, wide, pl = (d[d.token.isin(keep)] for d in (menu, wide, pl))
    parts = [orig, cf_rows(menu, "menu", hp), cf_rows(wide, "wide", hp), cf_rows(pl, "plausible", hp)]
    rows = pd.concat(parts, ignore_index=True)
    rows = rows[rows.token.isin(set(orig.token))]  # 원래 결정 행이 있는 장면만 (target 경로를 고칠 때 필요)
    sh = pd.read_csv(S14 / "train_extra_shards.tsv", sep="\t").set_index("token").shard
    rows["shard"] = rows.token.map(sh).fillna(0).astype(int)
    rows = rows[[c for c in KEEP if c in rows.columns] + dec_cols(rows)]
    val_logs = set((S14 / "val_logs.txt").read_text().split())
    for part, m in (("train", ~rows.log_name.isin(val_logs)), ("val", rows.log_name.isin(val_logs))):
        out = EXP / f"l3/step14_bundle_{part}_raw.parquet"
        rows[m].to_parquet(out, index=False)
        t = rows[m].groupby(["source", "sample_type"]).size().unstack(fill_value=0)
        print(f"{part}: {int(m.sum()):,} rows, 장면 {rows[m].token.nunique():,} -> {out}\n{t.to_string()}")
        neg = rows[m & (rows.sample_type == "cf_neg")]
        print(f"  CF⁻ 불가능 기준: {neg.category.value_counts().to_dict()}, 보이지 않는 원인 {(neg.visibility == 'unseen').sum()}")
        print(f"  CF⁺ target 규칙: {rows[m & (rows.sample_type == 'cf_pos')].target_rule.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
