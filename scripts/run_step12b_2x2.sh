#!/usr/bin/env bash
# 12단계 B 2x2: WV 묶음(세기 CF⁻ 2,198개)에서 세기 CF⁻ 무게 올림(--balance_strength) x 정답 위치·속도(--obj gtvel).
# 기준 칸(무게 그대로, 속도 없음)은 12단계 A의 Ours_WV. GPU 학습은 바로, CPU 평가는 앞 사다리 평가가 끝난 뒤.
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
L=/workspace/yesman/logs/step12b
RUNS=""
for v in bal gtvel balgtvel; do
  for seed in 0 1; do
    r=Ours_WV${v}_seed$seed; RUNS="$RUNS $r"
    extra=""; [[ $v == *bal* ]] && extra="$extra --balance_strength"; [[ $v == *gtvel* ]] && extra="$extra --obj gtvel"
    echo "=== train $r ($extra) $(date +%H:%M)"
    python scripts/train_planner.py --model Ours --steps 20000 --eval_every 10000 --seed $seed --bundle_suffix _WV \
      --neg_to_pos 0.25 $extra --name $r 2>&1 | grep --line-buffered -v Warn > $L/train_$r.log
    [ -f exp/train/$r/model.pt ] || { echo "학습 실패 $r: 중단"; exit 1; }
    python scripts/predict_planner.py --run $r --sets D0 D1 D2 D3 val D3x_gemma12b D3x_qwen8b D3x_gemma31b 2>&1 | grep -v Warn
  done
done
echo "=== GPU done $(date +%H:%M)"
until grep -q -E "all done|중단" $L/run.log; do sleep 60; done
for r in $RUNS; do
  echo "=== eval $r $(date +%H:%M)"
  python scripts/follow_eval.py --run $r --sets D0 D1 D2 val D3x_gemma12b D3x_qwen8b D3x_gemma31b 2>&1 | grep --line-buffered -v -E "Warn|Loading"
  python scripts/score_rows.py --table exp/eval/$r/D2.parquet 2>&1 | grep -v Warn
  python scripts/flag_threshold.py --run $r 2>&1 | grep -v Warn > $L/threshold_$r.json
  python scripts/score_planner.py --run $r --threads 28 > $L/score_$r.log 2>&1
  grep "^$r " $L/score_$r.log
done
python scripts/analyze_step12_d3safety.py --obj2 2>&1 | grep -v Warn
echo "=== all done $(date +%H:%M)"
