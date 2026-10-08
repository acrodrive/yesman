#!/usr/bin/env bash
# 12단계 (3) CF⁺ 목표 합성: 학습 묶음 합성 → (31B가 GPU를 비우면) B-순응, B3, Ours 시드 0, 1 다시 학습 → 평가.
# 설정은 10단계 최종과 같다(2만 step, CF⁻ 무게 0.25, 보이지 않는 원인 제외). 바뀌는 것은 CF⁺ 목표 경로뿐이다.
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
L=/workspace/yesman/logs/step12
echo "=== synth train $(date +%H:%M)"
[ -f exp/l3/train_bundle_train_synth.parquet ] || python scripts/l3_synth.py --bundle train 2>&1 | grep -v -E "Warn|Loading" | tail -4
until grep -q "VLM31 done" $L/vlm31b.log; do sleep 30; done
RUNS=""
for seed in 0 1; do
  for spec in "Bcomply Bcomply_syn" "B3 B3_syn" "Ours Ours_syn"; do
    set -- $spec; r=${2}_seed$seed; RUNS="$RUNS $r"
    extra=""; [ "$1" != "Bcomply" ] && extra="--neg_to_pos 0.25"
    echo "=== train $r $(date +%H:%M)"
    python scripts/train_planner.py --model $1 --steps 20000 --eval_every 10000 --seed $seed --bundle_suffix _synth $extra \
      --name $r 2>&1 | grep --line-buffered -v Warn > $L/train_$r.log
    python scripts/predict_planner.py --run $r 2>&1 | grep --line-buffered -v Warn
  done
done
echo "=== GPU done $(date +%H:%M)"
for r in $RUNS; do
  echo "=== eval $r $(date +%H:%M)"
  python scripts/follow_eval.py --run $r --sets D0 D1 D2 D3 val 2>&1 | grep --line-buffered -v -E "Warn|Loading"
  python scripts/score_rows.py --table exp/eval/$r/D2.parquet 2>&1 | grep -v Warn
  python scripts/score_planner.py --run $r --threads 28 > $L/score_$r.log 2>&1
  grep "^$r " $L/score_$r.log
  case $r in Ours*) python scripts/flag_threshold.py --run $r 2>&1 | grep -v Warn > $L/threshold_$r.json;; esac
done
echo "=== all done $(date +%H:%M)"
