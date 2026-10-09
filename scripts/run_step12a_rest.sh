#!/usr/bin/env bash
# 12단계 A 후속: VLM 결정 판정 → 새 CF 묶음 → 새 CF⁺ 합성 → 변형 묶음 → Ours(W, V, WV) x 시드 2 학습 → 평가.
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
L=/workspace/yesman/logs/step12a
judge () {  # $1 = train_qwen8b / train_gemma12b
  [ -f exp/d3/$1.parquet ] && return 0
  python scripts/d3_judge.py --name $1 --scenes exp/d3/scenes_navtrain.parquet --split navtrain --cache navtrain_sensor \
    --workers 24 2>&1 | grep -v -E "Warn|Loading" | tail -6
}
until [ -f exp/d3/train_qwen8b_raw.parquet ] && grep -q "=== download gemma12b\|VLM done" $L/vlm.log; do sleep 60; done
echo "=== judge qwen $(date +%H:%M)"; judge train_qwen8b
until grep -q "VLM done\|중단" $L/vlm.log; do sleep 60; done
grep -q "VLM done" $L/vlm.log || { echo "VLM 실패: 중단"; exit 1; }
echo "=== judge gemma12b $(date +%H:%M)"; judge train_gemma12b
until grep -q "all done" $L/wide.log; do sleep 60; done
echo "=== build extra $(date +%H:%M)"
python scripts/l3_build_extra.py --stage extra 2>&1 | grep -v Warn
for b in train val; do
  python scripts/l3_synth.py --bundle $b --input exp/l3/extra_$b.parquet --out exp/l3/extra_${b}_synth.parquet 2>&1 | grep -v -E "Warn|Loading" | tail -4
done
python scripts/l3_build_extra.py --stage variants 2>&1 | grep -v Warn
RUNS=""
for seed in 0 1; do
  for v in W V WV; do
    r=Ours_${v}_seed$seed; RUNS="$RUNS $r"
    echo "=== train $r $(date +%H:%M)"
    python scripts/train_planner.py --model Ours --steps 20000 --eval_every 10000 --seed $seed --bundle_suffix _$v \
      --neg_to_pos 0.25 --name $r 2>&1 | grep --line-buffered -v Warn > $L/train_$r.log
    python scripts/predict_planner.py --run $r --sets D0 D1 D2 D3 val D3x_gemma12b D3x_qwen8b D3x_gemma31b 2>&1 | grep -v Warn
  done
done
echo "=== GPU done $(date +%H:%M)"
for r in $RUNS; do
  echo "=== eval $r $(date +%H:%M)"
  python scripts/follow_eval.py --run $r --sets D0 D1 D2 val D3x_gemma12b D3x_qwen8b D3x_gemma31b 2>&1 | grep --line-buffered -v -E "Warn|Loading"
  python scripts/score_rows.py --table exp/eval/$r/D2.parquet 2>&1 | grep -v Warn
  python scripts/flag_threshold.py --run $r 2>&1 | grep -v Warn > $L/threshold_$r.json
  python scripts/score_planner.py --run $r --threads 28 > $L/score_$r.log 2>&1
  grep "^$r " $L/score_$r.log
done
python scripts/analyze_step12_d3safety.py --variants 2>&1 | grep -v Warn
echo "=== all done $(date +%H:%M)"
