#!/usr/bin/env bash
# 10단계 최종: CF⁻ 무게 0.25 (검증 세트로 고름, docs/step10_ours.md)로 Ours 시드 0, 1 학습 → 예측 → L1 판정 → 공식 D0 채점 → 기준값
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
for seed in 0 1; do
  r=Ours_w025_seed$seed
  echo "=== train $r $(date +%H:%M)"
  python scripts/train_planner.py --model Ours --steps 20000 --eval_every 5000 --seed $seed --neg_to_pos 0.25 --name $r \
    2>&1 | grep --line-buffered -v Warn > logs/step10/train_$r.log
  grep -E "eval" logs/step10/train_$r.log | tail -2
done
for seed in 0 1; do
  r=Ours_w025_seed$seed
  echo "=== eval $r $(date +%H:%M)"
  python scripts/predict_planner.py --run $r 2>&1 | grep --line-buffered -v Warn
  python scripts/follow_eval.py --run $r --sets D0 D1 val 2>&1 | grep --line-buffered -v -E "Warn|Loading"
  python scripts/score_planner.py --run $r --threads 28 > logs/step10/score_$r.log 2>&1
  grep "^$r " logs/step10/score_$r.log
  python scripts/flag_threshold.py --run $r 2>&1 | grep -v Warn > logs/step10/threshold_$r.json
done
echo "=== all done $(date +%H:%M)"
