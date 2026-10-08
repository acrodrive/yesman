#!/usr/bin/env bash
# 12단계 (1) B3: Ours에서 판단 head만 뺀 모델. 같은 설정(일관된 CF⁺ 목표, CF⁻ 무게 0.25, 보이지 않는 원인 제외, 2만 step).
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
for seed in 0 1; do
  r=B3_rt_seed$seed
  echo "=== train $r $(date +%H:%M)"
  python scripts/train_planner.py --model B3 --steps 20000 --eval_every 10000 --seed $seed --bundle_suffix _retarget \
    --neg_to_pos 0.25 --name $r 2>&1 | grep --line-buffered -v Warn > logs/step12/train_$r.log
  python scripts/predict_planner.py --run $r 2>&1 | grep --line-buffered -v Warn
done
echo "=== GPU done $(date +%H:%M)"
for seed in 0 1; do
  r=B3_rt_seed$seed
  echo "=== eval $r $(date +%H:%M)"
  python scripts/follow_eval.py --run $r --sets D0 D1 D2 D3 val 2>&1 | grep --line-buffered -v -E "Warn|Loading"
  python scripts/score_rows.py --table exp/eval/$r/D2.parquet 2>&1 | grep -v Warn
  python scripts/score_planner.py --run $r --threads 28 > logs/step12/score_$r.log 2>&1
  grep "^$r " logs/step12/score_$r.log
done
echo "=== all done $(date +%H:%M)"
