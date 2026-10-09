#!/usr/bin/env bash
# 12단계 B (진단): 물체 토큰 사다리. Ours(합성 목표, 최종 설정)에 물체 토큰 30개를 더해 학습하고 평가한다.
#   objltf = LTF 검출 결과(7단계에 저장), objgt = 정답 위치, objgtvel = 정답 위치 + 속도 (scripts/gt_objects.py)
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
L=/workspace/yesman/logs/step12b
until [ -f exp/features/ltf/navtrain/obj_gt.npy ] || [ "$1" = "ltf_only" ]; do sleep 30; done
RUNS=""
for v in ltf gt gtvel; do
  for seed in 0 1; do
    r=Ours_obj${v}_seed$seed; RUNS="$RUNS $r"
    echo "=== train $r $(date +%H:%M)"
    python scripts/train_planner.py --model Ours --steps 20000 --eval_every 10000 --seed $seed --bundle_suffix _synth \
      --neg_to_pos 0.25 --obj $v --name $r 2>&1 | grep --line-buffered -v Warn > $L/train_$r.log
    [ -f exp/train/$r/model.pt ] || { echo "학습 실패 $r: 중단"; exit 1; }
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
python scripts/analyze_step12_d3safety.py --obj 2>&1 | grep -v Warn
echo "=== all done $(date +%H:%M)"
