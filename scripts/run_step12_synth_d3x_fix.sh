#!/usr/bin/env bash
# 12단계 (3): Ours_syn_seed1의 D3x 예측이 GPU 메모리 충돌(버전 2 VLM 시범과 겹침)로 실패해서 다시 돌린다.
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
r=Ours_syn_seed1
python scripts/predict_planner.py --run $r --sets D3x_gemma12b D3x_qwen8b D3x_gemma31b 2>&1 | grep --line-buffered -v Warn
python scripts/follow_eval.py --run $r --sets D3x_gemma12b D3x_qwen8b D3x_gemma31b 2>&1 | grep --line-buffered -v -E "Warn|Loading"
python scripts/analyze_step12_d3safety.py --synth 2>&1 | grep -v Warn
echo "=== all done $(date +%H:%M)"
