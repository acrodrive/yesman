"""14단계: 새 테스트 세트(T0~T3)를 만들고 고정한다 (docs/step14_preregistration.md 2.2절).

T0~T2는 navtest의 D0~D2와 같은 규칙으로 만든다(scripts/l3_build_sets.py). T3_gemma31b는 scripts/d3_judge.py가
data_lists/eval_step14/에 바로 쓴다. 모델 결과는 보지 않는다. 이 스크립트는 판정 통계만 낸다.

사용법:
  python scripts/step14_testset.py --stage sets       # exp/l3/step14_test_cf.parquet → T0, T1, T2
  python scripts/step14_testset.py --stage freeze     # T3까지 만든 뒤: SHA256SUMS와 README (이후 바꾸지 않는다)
출력: data_lists/eval_step14/{T0,T1,T2,T3_*}.parquet, SHA256SUMS, README.md
"""

import argparse
import hashlib
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from l3_build_sets import dec_cols, stats  # noqa: E402

ROOT = Path(os.environ["YESMAN_ROOT"])
EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
OUT = ROOT / "data_lists/eval_step14"


def build_sets(src: Path):
    df = pd.read_parquet(src)
    keep = ["token", "log_name", "cf_name", "change_type", "decision"] + dec_cols(df)
    t0 = df[df.cf_name == "original"][keep + ["status"]].rename(columns={"status": "l2_status"})
    t1 = df[(df.cf_name != "original") & (df.status == "feasible")]
    t1 = t1[keep + ["best_poses", "best_score", "n_match", "n_pass", "n_scored"]].rename(columns={"best_poses": "target_poses"})
    t2 = df[(df.cf_name != "original") & (df.status == "infeasible")]
    t2 = t2[keep + ["category", "reason", "visibility", "weaker_decision", "weaker_status", "fail_collision",
                    "fail_off_road", "fail_wrong_way", "n_match", "n_scored", "best_poses"]].rename(
        columns={"best_poses": "best_failed_poses"})
    t2["main"] = t2.visibility != "unseen"
    OUT.mkdir(parents=True, exist_ok=True)
    for name, t in (("T0", t0), ("T1", t1), ("T2", t2)):
        t.to_parquet(OUT / f"{name}.parquet", index=False)
    (OUT / "stats.md").write_text(stats(df, "14단계 새 테스트 세트 (판정 통계만, 모델 결과 없음)"))
    print(f"T0 {len(t0):,}, T1 {len(t1):,}, T2 {len(t2):,} (주 결과 {int(t2.main.sum()):,}) -> {OUT}")


def freeze():
    files = sorted(OUT.glob("T*.parquet"))
    assert any(f.name.startswith("T3") for f in files), "T3(VLM 결정)를 먼저 만든다 (scripts/d3_judge.py)"
    sums = [f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.name}" for f in files]
    (OUT / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    n = {f.stem: len(pd.read_parquet(f, columns=["token"])) for f in files}
    (OUT / "README.md").write_text(
        f"# 14단계 새 테스트 세트 T0~T3\n\n- 만든 날: {date.today()}, 코드 커밋: {commit}\n"
        f"- 장면: navtrain 카메라 조각 중 테스트 조각(data_lists/step14/test_tokens.txt). 학습에 쓰지 않은 로그\n"
        f"- 상태: **고정**. 이후 바꾸지 않는다. 확인: `cd data_lists/eval_step14 && sha256sum -c SHA256SUMS`\n"
        f"- 한 번만 평가한다 (docs/step14_preregistration.md 8절)\n- 행 수: {n}\n"
        f"- 정의: T0~T2는 navtest D0~D2와 같은 규칙(scripts/l3_build_sets.py), T3_*는 D3와 같은 규칙(scripts/d3_judge.py)\n")
    print("\n".join(sums))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["sets", "freeze"], required=True)
    ap.add_argument("--src", type=Path, default=EXP / "l3/step14_test_cf.parquet")
    args = ap.parse_args()
    build_sets(args.src) if args.stage == "sets" else freeze()


if __name__ == "__main__":
    main()
