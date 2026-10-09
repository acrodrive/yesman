#!/usr/bin/env bash
# 12단계 D: 12단계 C(그럴싸한 CF⁻) 평가가 끝나면 probe 시험을 돌린다 (GPU, 모델 하나에 약 3분).
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
until grep -q -E "all done|중단" logs/step12c/run.log; do sleep 60; done
echo "=== probe $(date +%H:%M)"
python scripts/probe_dit.py --runs Ours_syn_seed0 Ours_syn_seed1 Ours_WV_seed0 Ours_WV_seed1 \
  Ours_WVbalgtvel_seed0 Ours_WVbalgtvel_seed1 Ours_plaus_seed0 Ours_plaus_seed1 2>&1 | grep --line-buffered -v Warn
echo "=== all done $(date +%H:%M)"
