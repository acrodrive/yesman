#!/usr/bin/env bash
# 10단계 시험: 장면 토큰에 LTF BEV 지도 분할 조각 토큰을 더한 B-순응 (시드 0, 2만 step). 검증 세트로만 판단한다.
# 고르는 규칙 (결과를 보기 전에 정함): 검증 CF⁺ 따르기가 기존(0.620)보다 3%p 이상 오르고 원래 결정 따르기(0.955)가
# 1%p 넘게 떨어지지 않으면 채택한다.
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
name=step10_bev_Bcomply_seed0
echo "=== $name $(date +%H:%M)"
python scripts/train_planner.py --model Bcomply --steps 20000 --eval_every 10000 --seed 0 --bev_sem --name $name \
  2>&1 | grep --line-buffered -v Warn > logs/step10c/train_$name.log
grep -E "eval step|done" logs/step10c/train_$name.log
python scripts/predict_planner.py --run $name --sets val train5k 2>&1 | grep --line-buffered -v Warn
python scripts/follow_eval.py --run $name --sets val train5k 2>&1 | grep --line-buffered -v -E "Warn|Loading"
echo "=== all done $(date +%H:%M)"
