#!/usr/bin/env bash
# 14단계 사후 분석 (탐색): 테스트 세트 T2, T3_gemma31b 결정의 그럴싸함 점수를 내고, 결정 유형별 칸 JSON을 다시 만든다.
# 점수는 학습 묶음과 같은 방법(scripts/cf_plausibility.py, 닮은 장면 후보 = 학습 장면)이다. 테스트 결과는 다시 나누기만 한다.
# 출력: exp/step14/posthoc/{T2,T3_gemma31b}_plaus.parquet (git 밖), docs/step14_type_cells.json
#
# 사용법: bash scripts/run_step14_posthoc_plaus.sh
set -euo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
mkdir -p exp/step14/posthoc
for s in T2 T3_gemma31b; do
  python scripts/cf_plausibility.py --table data_lists/eval_step14/$s.parquet --query_split navtrain \
    --pool data_lists/step14/train_all_tokens.txt --out exp/step14/posthoc/${s}_plaus.parquet 2>&1 | grep -v Warn
done
python scripts/step14_type_cells.py
