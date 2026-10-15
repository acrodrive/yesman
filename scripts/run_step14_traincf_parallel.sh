#!/usr/bin/env bash
# 14단계 데이터 준비 보조: 학습 CF 판정 중 확장된 규칙 기반 CF(traincf_wide)와 plausible CF(traincf_plausible)를
# 드라이버(run_step14_pipeline.sh)의 traincf_menu와 동시에 돌린다 (2026-10-15 사용자 결정. 세 판정은 서로 기다릴 필요가 없고,
# 테스트 CF 판정 때 작업자들이 디스크 읽기를 기다리느라 CPU가 절반쯤 놀았다).
#
# - 학습 추가 metric cache(cache_train)가 끝나기를 기다린 뒤 띄운다. 띄우기 전에 남은 메모리가 MIN_AVAIL_GB 이상인지 본다.
# - 드라이버가 같은 단계를 또 돌리지 않도록 state/<단계>.ok에 "따로 도는 중"을 적어 둔다. 드라이버가 먼저 묶음(bundle) 단계에
#   닿으면 입력 파일이 없어 bundle이 실패로 멈춘다(데이터는 망가지지 않는다). 두 판정이 끝나면 이 스크립트가 드라이버를 다시 띄운다
#   (끝난 단계는 건너뛰고, 돌고 있는 갈래는 자물쇠 때문에 다시 띄워지지 않는다).
# - 판정이 실패하면 .ok를 지우고 .fail을 남긴다. 드라이버가 그 단계에 닿으면 다시 돌린다.
#
# 사용법: nohup setsid bash scripts/run_step14_traincf_parallel.sh > logs/step14/traincf_parallel.log 2>&1 &
set -u
cd /workspace/yesman
ST=logs/step14/state
WORKERS_EACH=${WORKERS_EACH:-12}
MIN_AVAIL_GB=${MIN_AVAIL_GB:-30}
ev () { echo "$(date '+%m-%d %H:%M') $*" >> $ST/events.log; echo "=== $(date '+%m-%d %H:%M') $*"; }
avail_gb () { free -g | awk '/^Mem:/ {print $7}'; }

until [ -f $ST/cache_train.ok ]; do [ -f $ST/cache_train.fail ] && { ev "parallel: cache_train 실패, 그만둠"; exit 1; }; sleep 60; done

declare -A OUT=([traincf_wide]=exp/l3/step14_train_cf_wide.parquet [traincf_plausible]=exp/l3/step14_train_cf_plausible.parquet)
pids=()
for s in traincf_wide traincf_plausible; do
  if [ -f $ST/$s.ok ] && ! grep -q "따로 도는 중" $ST/$s.ok; then continue; fi
  while [ "$(avail_gb)" -lt $MIN_AVAIL_GB ]; do sleep 60; done
  touch $ST/$s.start; sleep 1
  echo "따로 도는 중 (run_step14_traincf_parallel.sh, $(date '+%m-%d %H:%M'))" > $ST/$s.ok
  ev "start $s (따로, 작업자 $WORKERS_EACH)"
  ( WORKERS=$WORKERS_EACH bash scripts/run_step14_data.sh $s >> logs/step14/pipeline_$s.log 2>&1
    if [ -f ${OUT[$s]} ] && [ ${OUT[$s]} -nt $ST/$s.start ]; then
      echo "끝남 (따로 돌림, $(date '+%m-%d %H:%M'))" > $ST/$s.ok; ev "ok $s (따로)"
    else
      rm -f $ST/$s.ok; touch $ST/$s.fail; ev "FAIL $s (따로, logs/step14/pipeline_$s.log)"; exit 1
    fi ) &
  pids+=($!)
  sleep 300  # 두 번째를 띄우기 전에 첫 번째의 메모리 사용이 자리 잡게 한다
done
fail=0; for p in "${pids[@]}"; do wait $p || fail=1; done
[ $fail = 0 ] || exit 1

# 드라이버의 CPU 갈래가 bundle에서 멈췄으면 다시 띄운다 (돌고 있는 갈래는 자물쇠 때문에 다시 띄워지지 않는다)
if [ -f $ST/bundle.fail ]; then
  rm -f $ST/bundle.fail
  ev "parallel: bundle이 먼저 실패해 있었다. 드라이버를 다시 띄운다"
  nohup setsid bash scripts/run_step14_pipeline.sh >> logs/step14/pipeline.log 2>&1 < /dev/null &
fi
ev "parallel: 끝"
