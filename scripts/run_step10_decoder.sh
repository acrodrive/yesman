#!/usr/bin/env bash
# 10단계 디코더 시험: B-순응 시드 0, 2만 step, 검증 세트로만 판단 (navtest는 보지 않는다).
# 고르는 규칙 (결과를 보기 전에 정함): 검증 CF⁺ 따르기가 기존(x0 + L1, 0.620)보다 3%p 이상 오르고
# 원래 결정 따르기(0.955)가 1%p 넘게 떨어지지 않는 설정 중 가장 좋은 것.
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
for cfg in "v mse" "x0 mse"; do
  set -- $cfg
  name=step10_dec_${1}_${2}_Bcomply_seed0
  echo "=== $name $(date +%H:%M)"
  python scripts/train_planner.py --model Bcomply --steps 20000 --eval_every 10000 --seed 0 --pred $1 --loss $2 --name $name \
    2>&1 | grep --line-buffered -v Warn > logs/step10b/train_$name.log
  grep "eval step" logs/step10b/train_$name.log
  python scripts/predict_planner.py --run $name --sets val 2>&1 | grep --line-buffered -v Warn
  python scripts/follow_eval.py --run $name --sets val 2>&1 | grep --line-buffered -v -E "Warn|Loading"
done
echo "=== all done $(date +%H:%M)"
