"""6단계: L3 결과(exp/l3/*.parquet)로 평가 세트(D0~D2)와 학습 묶음을 만들고 통계 표를 낸다.

평가 세트 (navtest, data_lists/eval/, git으로 관리. 사용자가 확인한 뒤 고정한다):
  D0.parquet: 원래 결정 (구조적 애매함이 없는 navtest 장면 전부). l2_status = 그 결정의 L2 판정
  D1.parquet: CF⁺ (L2 가능) + target_poses (L2가 찾은 경로, 8x3)
  D2.parquet: CF⁻ (L2 불가능) + category(strength/agent/road), reason, visibility.
              main = (visibility != unseen). 보이지 않는 원인(unseen)은 주 결과에서 빼고 따로 보고한다(7.3).
학습 묶음 (navtrain 센서 장면, exp/l3/train_bundle_{train,val}.parquet):
  원래 결정(목표 = 사람 경로, flag ACCEPT), CF⁺(목표 = L2 경로, ACCEPT), CF⁻(목표 = 사람 경로, REJECT, 판단 근거)
  검증 세트 = data_lists/navtrain_sensor_val_logs.txt의 로그 (로그 단위 10%)

사용법: python scripts/l3_build_sets.py --navtest exp/l3/navtest_cf.parquet [--navtrain exp/l3/navtrain_sensor_cf.parquet]
"""

import argparse
import os
import subprocess
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
ROOT = Path(os.environ["YESMAN_ROOT"])
DEC_COLS = None  # d_seg* 열


def dec_cols(df):
    return [c for c in df.columns if c.startswith("d_seg")]


def human_poses(split: str, tokens) -> pd.Series:
    f = pd.read_parquet(EXP / f"l1/{split}_features.parquet", columns=["token", "poses"])
    return f.set_index("token").poses.reindex(tokens)


def stats(df: pd.DataFrame, title: str) -> str:
    lines = [f"## {title}\n"]
    n_scene = df.token.nunique()
    lines.append(f"- 장면 {n_scene:,}개, 판정 {len(df):,}개 (원래 결정 포함), 오류 {(df.status == 'error').sum()}\n")
    t = df[df.cf_name != "error"].groupby("cf_name").status.value_counts().unstack().fillna(0).astype(int)
    t["합계"] = t.sum(axis=1)
    lines.append("| 결정 | " + " | ".join(t.columns) + " |\n" + "| --- " * (len(t.columns) + 1) + "|")
    for k, r in t.iterrows():
        lines.append(f"| {k} | " + " | ".join(f"{v:,}" for v in r) + " |")
    cf = df[(df.cf_name != "original") & (df.status == "infeasible")]
    if len(cf):
        lines.append("\n### CF⁻ 불가능 기준\n")
        c = cf.groupby("category").agg(n=("token", "size"), unseen=("visibility", lambda x: (x == "unseen").sum()),
                                       unknown=("visibility", lambda x: (x == "unknown").sum()))
        c["main(unseen 제외)"] = c.n - c.unseen
        lines.append("| 기준 | CF⁻ | 보이지 않는 원인 | 원인 차량 못 찾음 | 주 결과 |\n| --- | --- | --- | --- | --- |")
        for k, r in c.iterrows():
            lines.append(f"| {k} | {r.n:,} | {r.unseen:,} | {r.unknown:,} | {r['main(unseen 제외)']:,} |")
        lines.append("\n### CF⁻ 기준 x 결정 종류\n")
        ct = pd.crosstab(cf.cf_name, cf.category)
        lines.append("| 결정 | " + " | ".join(ct.columns) + " |\n" + "| --- " * (len(ct.columns) + 1) + "|")
        for k, r in ct.iterrows():
            lines.append(f"| {k} | " + " | ".join(f"{v:,}" for v in r) + " |")
        lines.append("\n### CF⁻ 판단 근거 (대표 후보가 떨어진 항목)\n")
        for k, v in cf.reason.value_counts().items():
            lines.append(f"- {k}: {v:,}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--navtest", type=Path, default=EXP / "l3/navtest_cf.parquet")
    ap.add_argument("--navtrain", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "data_lists/eval")
    args = ap.parse_args()
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    report = [f"# 6단계 L3 통계 ({date.today()}, 코드 {commit})\n"]

    if args.navtest and args.navtest.exists():
        df = pd.read_parquet(args.navtest)
        report.append(stats(df, "navtest (평가 세트)"))
        args.out.mkdir(parents=True, exist_ok=True)
        keep = ["token", "log_name", "cf_name", "change_type", "decision"] + dec_cols(df)
        d0 = df[df.cf_name == "original"][keep + ["status"]].rename(columns={"status": "l2_status"})
        d1 = df[(df.cf_name != "original") & (df.status == "feasible")]
        d1 = d1[keep + ["best_poses", "best_score", "n_match", "n_pass", "n_scored"]].rename(
            columns={"best_poses": "target_poses"})
        d2 = df[(df.cf_name != "original") & (df.status == "infeasible")]
        d2 = d2[keep + ["category", "reason", "visibility", "weaker_decision", "weaker_status", "fail_collision",
                        "fail_off_road", "fail_wrong_way", "n_match", "n_scored", "best_poses"]].rename(
            columns={"best_poses": "best_failed_poses"})
        d2["main"] = d2.visibility != "unseen"
        for name, t in (("D0", d0), ("D1", d1), ("D2", d2)):
            t.to_parquet(args.out / f"{name}.parquet", index=False)
        (args.out / "README.md").write_text(
            f"# 평가 세트 D0~D2 (navtest)\n\n- 만든 날: {date.today()}, 코드 커밋: {commit}, 입력: {args.navtest.name}\n"
            f"- 상태: **검토 대기** (사용자가 확인한 뒤 고정한다. 고정한 뒤에는 바꾸지 않는다)\n"
            f"- D0 {len(d0):,}개 (L2 가능 {int((d0.l2_status == 'feasible').sum()):,}), D1 {len(d1):,}개, "
            f"D2 {len(d2):,}개 (주 결과 {int(d2.main.sum()):,})\n- 열 설명과 만드는 방법: docs/step06_cf.md, scripts/l3_build_sets.py\n")
        print(f"saved D0 {len(d0)}, D1 {len(d1)}, D2 {len(d2)} -> {args.out}")

    if args.navtrain and args.navtrain.exists():
        df = pd.read_parquet(args.navtrain)
        report.append(stats(df, "navtrain 센서 장면 (학습 + 검증)"))
        val_logs = set((ROOT / "data_lists/navtrain_sensor_val_logs.txt").read_text().split())
        hp = human_poses("navtrain", df.token.unique())
        rows = df[df.status.isin(["human", "feasible", "infeasible"])].copy()
        rows["flag"] = np.where(rows.status == "infeasible", "REJECT", "ACCEPT")
        rows["target_poses"] = [hp[t] if s in ("human", "infeasible") else b
                                for t, s, b in zip(rows.token, rows.status, rows.best_poses)]
        rows["sample_type"] = np.select([rows.cf_name == "original", rows.status == "feasible"],
                                        ["original", "cf_pos"], "cf_neg")
        keep = ["token", "log_name", "sample_type", "cf_name", "change_type", "decision", "flag", "reason", "category",
                "visibility", "target_poses"] + dec_cols(rows)
        rows = rows[[c for c in keep if c in rows.columns]]
        for part, m in (("train", ~rows.log_name.isin(val_logs)), ("val", rows.log_name.isin(val_logs))):
            p = EXP / f"l3/train_bundle_{part}.parquet"
            rows[m].to_parquet(p, index=False)
            vc = rows[m].sample_type.value_counts().to_dict()
            print(f"saved {p}: {int(m.sum()):,} samples {vc}")
            report.append(f"- 학습 묶음 {part}: {int(m.sum()):,}개 {vc}\n")
    out = ROOT / "logs/step06/l3_stats.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report))
    print("\n".join(report))


if __name__ == "__main__":
    main()
