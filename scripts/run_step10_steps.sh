#!/usr/bin/env bash
# 10단계 진단: (1) B-순응을 5만 step 학습해 검증 세트 CF⁺ 따르기가 2만 step보다 오르는지 (학습 부족인가)
#              (2) 2만 step B-순응이 학습 데이터의 CF⁺는 따르는지 (평균으로 끌림/학습 부족 대 일반화)
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
echo "=== train5k Bcomply_seed0 $(date +%H:%M)"
python scripts/predict_planner.py --run Bcomply_seed0 --sets train5k 2>&1 | grep --line-buffered -v Warn
python scripts/follow_eval.py --run Bcomply_seed0 --sets train5k 2>&1 | grep --line-buffered -v -E "Warn|Loading"
name=step10_Bcomply_50k_seed0
echo "=== $name $(date +%H:%M)"
python scripts/train_planner.py --model Bcomply --steps 50000 --eval_every 10000 --seed 0 --name $name \
  2>&1 | grep --line-buffered -v Warn > logs/step10/train_$name.log
grep "eval step" logs/step10/train_$name.log
python scripts/predict_planner.py --run $name --sets val train5k 2>&1 | grep --line-buffered -v Warn
python scripts/follow_eval.py --run $name --sets val train5k 2>&1 | grep --line-buffered -v -E "Warn|Loading"
echo "=== all done $(date +%H:%M)"
