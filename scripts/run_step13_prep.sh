#!/usr/bin/env bash
# 13단계 준비: 최종 Ours(합성 목표)의 규칙 대체 경로 → D2 채점 → 최종 모델로 11단계 표와 그림 다시 만들기.
# GPU 충돌을 피하려고 12단계 재평가(run_step12_synth_d3x_fix.sh)가 끝난 뒤 시작한다.
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
until grep -q "all done" logs/step12/synth_d3x_fix.log; do sleep 20; done
for r in Ours_syn_seed0 Ours_syn_seed1; do
  python scripts/predict_rule.py --run $r 2>&1 | grep -v Warn
  python scripts/score_rows.py --table exp/eval/$r/D2_rule.parquet 2>&1 | grep -v Warn
done
python scripts/analyze_step11.py --final 2>&1 | grep -v Warn | tail -2
python scripts/viz_step11.py --final 2>&1 | grep -v -E "Warn|Loading"
echo "=== all done $(date +%H:%M)"
