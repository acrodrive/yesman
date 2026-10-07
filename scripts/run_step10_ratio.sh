#!/usr/bin/env bash
# 10단계: CF⁻ 무게(neg_to_pos) 고르기. 시드 0, 검증 세트로만 판단한다 (navtest는 보지 않는다).
# 고르는 규칙 (결과를 보기 전에 정함): 검증 세트 CF⁻ 거부율(τ 적용) 85% 이상인 설정 중, 경로만의 CF⁺ 따르기가 가장 높은 것.
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
for r in 0.5 0.25; do
  name=step10_ratio${r}_seed0
  echo "=== $name $(date +%H:%M)"
  python scripts/train_planner.py --model Ours --steps 20000 --eval_every 20000 --seed 0 --neg_to_pos $r --name $name \
    2>&1 | grep --line-buffered -v Warn > logs/step10/train_$name.log
  python scripts/predict_planner.py --run $name --sets val 2>&1 | grep --line-buffered -v Warn
  python scripts/follow_eval.py --run $name --sets val 2>&1 | grep --line-buffered -v -E "Warn|Loading"
done
for name in Ours_seed0 step10_ratio0.5_seed0 step10_ratio0.25_seed0; do
  python scripts/flag_threshold.py --run $name --val_only 2>&1 | grep -v Warn > logs/step10/val_$name.json
done
echo "=== all done $(date +%H:%M)"
