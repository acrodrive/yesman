#!/usr/bin/env bash
# 14단계 사후 분석 1번: T1 일부(무작위 1만 개 + 판단 모델이 거부한 결정 전부)를 모델 6개 x 시드 0·1로 채점한다.
# 입력은 scripts/step14_posthoc_cells.py prepare가 만든 exp/eval/<run>/posthoc/T1_sub.parquet (고르지 않은 행은 경로가 None).
# 기존 평가 파일은 건드리지 않는다. 끝난 것은 건너뛴다. 동시에 JOBS개(기본 2), 하나에 작업자 WORKERS개(기본 14).
#
# 사용법: nohup setsid bash scripts/run_step14_posthoc_t1.sh > logs/step14/posthoc_t1.log 2>&1 &
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
JOBS=${JOBS:-2}; WORKERS=${WORKERS:-14}
python scripts/step14_posthoc_cells.py prepare || exit 1
python - <<'EOF' | xargs -P $JOBS -I{} bash -c '
  d=exp/eval/{}/posthoc
  [ -f $d/T1_sub_scores.parquet ] && exit 0
  echo "=== $(date "+%m-%d %H:%M") 시작 {}"
  python scripts/score_rows.py --table $d/T1_sub.parquet --cache navtrain_step14_test --workers '$WORKERS' \
    > $d/score.log 2>&1 || { echo "=== 채점 실패 {}"; exit 1; }
  echo "=== $(date "+%m-%d %H:%M") 끝 {}: $(tail -1 $d/score.log)"'
import sys; sys.path.insert(0, "scripts")
from step14_posthoc_cells import runs
print("\n".join(runs()))
EOF
echo "=== $(date '+%m-%d %H:%M') 모두 끝 (실패 수: $(grep -c '채점 실패' logs/step14/posthoc_t1.log 2>/dev/null))"
