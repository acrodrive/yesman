#!/usr/bin/env bash
# 12단계 (3) 이어서: 합성 목표 모델 6개를 D3 확장(실제 VLM 결정)으로 평가한다.
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
for r in Bcomply_syn_seed0 Bcomply_syn_seed1 B3_syn_seed0 B3_syn_seed1 Ours_syn_seed0 Ours_syn_seed1; do
  echo "=== $r $(date +%H:%M)"
  python scripts/predict_planner.py --run $r --sets D3x_gemma12b D3x_qwen8b D3x_gemma31b 2>&1 | grep --line-buffered -v Warn
  python scripts/follow_eval.py --run $r --sets D3x_gemma12b D3x_qwen8b D3x_gemma31b 2>&1 | grep --line-buffered -v -E "Warn|Loading"
done
python scripts/analyze_step12_d3safety.py --synth 2>&1 | grep -v Warn
echo "=== all done $(date +%H:%M)"
