#!/usr/bin/env bash
# 9단계: 학습한 비교 모델의 평가 (예측 → L1 따르기 판정 → 공식 D0 채점). 시드 0 먼저.
# 사용법: nohup setsid scripts/run_eval_step09.sh > logs/step09/eval_all.log 2>&1 &
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
for seed in 0 1; do
  for m in B1 B2 Bcomply; do
    r=${m}_seed${seed}
    echo "=== $r $(date +%H:%M)"
    python scripts/predict_planner.py --run $r 2>&1 | grep --line-buffered -v Warn
    python scripts/follow_eval.py --run $r --sets D0 D1 val 2>&1 | grep --line-buffered -v -E "Warn|Loading"
    python scripts/score_planner.py --run $r --threads 28 > logs/step09/score_$r.log 2>&1
    grep "^$r " logs/step09/score_$r.log
  done
done
echo "=== all done $(date +%H:%M)"
