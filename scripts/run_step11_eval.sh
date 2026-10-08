#!/usr/bin/env bash
# 11단계 CPU 평가: D2, D3 따르기 판정 + D2 경로 채점(대체 경로 품질) + D3 L2 최고 경로 채점("가능하지만 위험").
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
RUNS="B1_seed0 B1_seed1 B2_seed0 B2_seed1 Bcomply_rt_seed0 Bcomply_rt_seed1 Ours_w025_rt_seed0 Ours_w025_rt_seed1"
for r in $RUNS; do
  echo "=== follow D2 D3 $r $(date +%H:%M)"
  python scripts/follow_eval.py --run $r --sets D2 D3 2>&1 | grep --line-buffered -v -E "Warn|Loading"
done
echo "=== score D3 best $(date +%H:%M)"
python scripts/score_rows.py --table data_lists/eval/D3.parquet --col best_poses --out exp/eval/D3_best_scores.parquet 2>&1 | grep -v Warn
for t in exp/eval/Ours_w025_rt_seed0/D2_rule.parquet exp/eval/Ours_w025_rt_seed1/D2_rule.parquet $(for r in $RUNS; do echo exp/eval/$r/D2.parquet; done); do
  echo "=== score $t $(date +%H:%M)"
  python scripts/score_rows.py --table $t 2>&1 | grep -v Warn
done
echo "=== all done $(date +%H:%M)"
