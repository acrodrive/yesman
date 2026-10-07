#!/usr/bin/env bash
# 10단계: Ours 학습(시드 0, 1) → 예측 → L1 따르기 판정 → 공식 D0 채점.
# 사용법: nohup setsid scripts/run_step10.sh > logs/step10/run.log 2>&1 &
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
for seed in 0 1; do
  echo "=== train Ours_seed$seed $(date +%H:%M)"
  python scripts/train_planner.py --model Ours --steps 20000 --eval_every 5000 --seed $seed --name Ours_seed$seed \
    2>&1 | grep --line-buffered -v Warn > logs/step10/train_Ours_seed$seed.log
  grep -E "eval|done" logs/step10/train_Ours_seed$seed.log | tail -3
done
for seed in 0 1; do
  r=Ours_seed$seed
  echo "=== eval $r $(date +%H:%M)"
  python scripts/predict_planner.py --run $r 2>&1 | grep --line-buffered -v Warn
  python scripts/follow_eval.py --run $r --sets D0 D1 val 2>&1 | grep --line-buffered -v -E "Warn|Loading"
  python scripts/score_planner.py --run $r --threads 28 > logs/step10/score_$r.log 2>&1
  grep "^$r " logs/step10/score_$r.log
done
echo "=== all done $(date +%H:%M)"
