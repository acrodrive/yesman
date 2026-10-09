#!/usr/bin/env bash
# 12단계 C: 그럴싸한 CF⁻ 생성(CPU) → 학습 묶음(합성 묶음 + 그럴싸한 CF⁻) → Ours x 시드 2 (CF⁻ 안에서 그럴싸한 것 85%) → 평가.
# 아키텍처와 나머지 설정은 최종 Ours(Ours_syn)와 같다. 비교 기준은 Ours_syn.
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
L=/workspace/yesman/logs/step12c
echo "=== plausible CF $(date +%H:%M)"
[ -f exp/l3/navtrain_sensor_cf_plausible.parquet ] || \
  python scripts/l3_plausible.py --workers 28 2>&1 | grep --line-buffered -v -E "Warn|Loading" > $L/plausible.log
[ -f exp/l3/navtrain_sensor_cf_plausible.parquet ] || { echo "생성 실패: 중단"; exit 1; }
tail -8 $L/plausible.log
python scripts/l3_build_extra.py --stage plausible 2>&1 | grep -v Warn
RUNS=""
for seed in 0 1; do
  r=Ours_plaus_seed$seed; RUNS="$RUNS $r"
  echo "=== train $r $(date +%H:%M)"
  python scripts/train_planner.py --model Ours --steps 20000 --eval_every 10000 --seed $seed --bundle_suffix _plaus \
    --neg_to_pos 0.25 --plausible_share 0.85 --name $r 2>&1 | grep --line-buffered -v Warn > $L/train_$r.log
  [ -f exp/train/$r/model.pt ] || { echo "학습 실패 $r: 중단"; exit 1; }
  python scripts/predict_planner.py --run $r --sets D0 D1 D2 D3 val D3x_gemma12b D3x_qwen8b D3x_gemma31b 2>&1 | grep -v Warn
done
for r in $RUNS; do
  echo "=== eval $r $(date +%H:%M)"
  python scripts/follow_eval.py --run $r --sets D0 D1 D2 val D3x_gemma12b D3x_qwen8b D3x_gemma31b 2>&1 | grep --line-buffered -v -E "Warn|Loading"
  python scripts/score_rows.py --table exp/eval/$r/D2.parquet 2>&1 | grep -v Warn
  python scripts/flag_threshold.py --run $r 2>&1 | grep -v Warn > $L/threshold_$r.json
  python scripts/score_planner.py --run $r --threads 28 > $L/score_$r.log 2>&1
  grep "^$r " $L/score_$r.log
done
python scripts/analyze_step12_d3safety.py --plaus 2>&1 | grep -v Warn
echo "=== all done $(date +%H:%M)"
