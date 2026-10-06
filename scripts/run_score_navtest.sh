#!/usr/bin/env bash
# 3단계: NAVSIM v2 공식 채점 도구로 navtest를 채점한다.
# navtest는 one-stage 채점(run_pdm_score_one_stage.py, EPDMS, non-reactive 교통)을 쓴다.
# 결과 CSV: $NAVSIM_EXP_ROOT/step03/<agent>/<시각>/<시각>.csv (장면마다 한 줄 + 마지막 줄이 평균)
#
# 환경변수: THREADS(기본 16), METRIC_CACHE(기본 exp/metric_cache/navtest), EXP_NAME(기본 step03/<agent>)
# 사용법 (source scripts/setup/env.sh 후):
#   scripts/run_score_navtest.sh human_agent
#   scripts/run_score_navtest.sh constant_velocity_agent
# 먼저 metric cache가 있어야 한다:
#   python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
#       train_test_split=navtest metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/navtest worker.threads_per_node=16
set -euo pipefail
AGENT=${1:?agent config 이름 (human_agent, constant_velocity_agent, ...)}
shift
THREADS=${THREADS:-16}
METRIC_CACHE=${METRIC_CACHE:-$NAVSIM_EXP_ROOT/metric_cache/navtest}
EXP_NAME=${EXP_NAME:-step03/$AGENT}

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py \
    train_test_split=navtest \
    agent=$AGENT \
    experiment_name=$EXP_NAME \
    traffic_agents=non_reactive \
    metric_cache_path=$METRIC_CACHE \
    worker.threads_per_node=$THREADS \
    "$@"
