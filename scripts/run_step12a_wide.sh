#!/usr/bin/env bash
# 12단계 A: 넓힌 CF 메뉴로 학습 장면 판정 (CPU). 6단계 학습용 판정과 같은 설정에 --wide만 더한다.
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
echo "=== wide CF $(date +%H:%M)"
python scripts/l3_run.py --split navtrain --cache navtrain_sensor --tokens data_lists/navtrain_sensor_tokens.txt \
  --skip_original --wide --workers 24 --name navtrain_sensor_cf_wide 2>&1 | grep --line-buffered -v -E "Warn|Loading" | tail -30
echo "=== all done $(date +%H:%M)"
