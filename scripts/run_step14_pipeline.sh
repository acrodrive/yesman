#!/usr/bin/env bash
# 14단계 데이터 준비를 사람 없이 끝까지 이어 돌리는 드라이버 (단계 내용은 scripts/run_step14_data.sh).
# CPU 갈래와 GPU 갈래를 따로 돌리고, 단계 사이의 순서를 지킨다.
#   CPU: download → shards → cache_test → testcf → cache_train → traincf_menu → traincf_wide → traincf_plausible
#        → bundle → (features를 기다림) plaus → cache_reserve
#   GPU: (shards를 기다림) features → (testcf를 기다림) vlm → freeze
# 단계마다 logs/step14/state/<단계>.{start,ok,fail}을 남기고, 사건은 logs/step14/state/events.log에 한 줄씩 적는다.
# 성공은 출력 파일로 확인한다(run_step14_data.sh는 파이프를 쓰므로 종료 코드를 믿지 않는다).
# 다시 띄우면 .ok인 단계는 건너뛰고, .fail인 단계부터 다시 한다. 같은 갈래가 이미 돌고 있으면 그 갈래는 띄우지 않는다.
#
# 사용법: nohup setsid bash scripts/run_step14_pipeline.sh > logs/step14/pipeline.log 2>&1 &
set -u
cd /workspace/yesman
ST=logs/step14/state; mkdir -p $ST
S=data_lists/step14
SHARDS="2 3 4 6 7 8 10 11 12 14 15 16 18 19 20 22 23 24 26 27 28 30 31 32"
MC=exp/metric_cache

ev () { echo "$(date '+%m-%d %H:%M') $*" >> $ST/events.log; }
newer () { [ -f "$1" ] && [ "$1" -nt "$ST/$2.start" ]; }
lines () { [ -f "$1" ] && wc -l < "$1" || echo 0; }
# 캐시 metadata의 장면 수가 목록의 95% 이상인가 (공식 caching은 몇 장면을 실패할 수 있다)
cache_ok () { local n m; n=$(( $(lines "$1") - 1 )); m=$(lines "$2"); [ "$n" -ge $(( m * 95 / 100 )) ] && [ "$m" -gt 0 ]; }

check () {
  case $1 in
    download) for i in $SHARDS; do [ -e /workspace/datasets/navsim/_done/navtrain_cur_$i ] || return 1; done ;;
    shards) newer $S/check.md shards ;;
    cache_test) newer $MC/navtrain_step14_test/metadata/navtrain_step14_test_metadata_node_0.csv cache_test \
                && cache_ok $MC/navtrain_step14_test/metadata/navtrain_step14_test_metadata_node_0.csv $S/test_tokens.txt ;;
    testcf) newer data_lists/eval_step14/T0.parquet testcf && newer data_lists/eval_step14/T2.parquet testcf ;;
    cache_train) newer $MC/navtrain_step14_train/metadata/navtrain_step14_train_metadata_node_0.csv cache_train \
                 && cache_ok $MC/navtrain_step14_train/metadata/navtrain_step14_train_metadata_node_0.csv $S/train_all_tokens.txt ;;
    traincf_menu) newer exp/l3/step14_train_cf.parquet traincf_menu ;;
    traincf_wide) newer exp/l3/step14_train_cf_wide.parquet traincf_wide ;;
    traincf_plausible) newer exp/l3/step14_train_cf_plausible.parquet traincf_plausible ;;
    bundle) newer exp/l3/train_bundle_train_step14.parquet bundle && newer exp/l3/train_bundle_val_step14.parquet bundle ;;
    plaus) newer exp/l3/train_bundle_train_step14_plaus.parquet plaus && newer exp/l3/train_bundle_val_step14_plaus.parquet plaus ;;
    cache_reserve) newer $MC/navtrain_step14_reserve/metadata/navtrain_step14_reserve_metadata_node_0.csv cache_reserve \
                   && cache_ok $MC/navtrain_step14_reserve/metadata/navtrain_step14_reserve_metadata_node_0.csv $S/reserve_tokens.txt ;;
    features) newer exp/features/ltf/navtrain/tokens.txt features \
              && [ "$(lines exp/features/ltf/navtrain/tokens.txt)" -eq "$(lines $S/camera_all_tokens.txt)" ] ;;
    vlm) newer data_lists/eval_step14/T3_gemma31b.parquet vlm ;;
    freeze) newer data_lists/eval_step14/SHA256SUMS freeze ;;
  esac
}

run () {  # 단계 하나. 성공하면 0
  local s=$1
  [ -f $ST/$s.ok ] && return 0
  rm -f $ST/$s.fail; touch $ST/$s.start; sleep 1
  ev "start $s"
  if [ $s = download ]; then
    # 처음 다운로드는 세 갈래로 따로 띄웠다(download_{a,b,c}). 그것이 끝나기를 기다린 뒤 빠진 조각만 다시 받는다
    until [ -f $ST/download_a.exit ] && [ -f $ST/download_b.exit ] && [ -f $ST/download_c.exit ]; do sleep 60; done
    check download || bash scripts/run_step14_data.sh download >> logs/step14/pipeline_download.log 2>&1
  else
    bash scripts/run_step14_data.sh $s >> logs/step14/pipeline_$s.log 2>&1
  fi
  if check $s; then touch $ST/$s.ok; ev "ok $s"; return 0; fi
  touch $ST/$s.fail; ev "FAIL $s (logs/step14/pipeline_$s.log)"; return 1
}

waitfor () {  # 다른 갈래의 단계가 끝나기를 기다린다. 그 단계가 실패하면 1
  until [ -f $ST/$1.ok ]; do
    [ -f $ST/$1.fail ] && { ev "stop: $1 failed, waiting lane gives up"; return 1; }
    sleep 60
  done
}

lane_cpu () {
  for s in download shards cache_test testcf cache_train traincf_menu traincf_wide traincf_plausible bundle; do
    run $s || return 1
  done
  waitfor features || return 1
  run plaus || return 1
  run cache_reserve || return 1
  ev "lane cpu done"
}

lane_gpu () {
  waitfor shards || return 1
  run features || return 1
  waitfor testcf || return 1
  run vlm || return 1
  run freeze || return 1
  ev "lane gpu done"
}

( flock -n 8 || { echo "cpu lane already running"; exit 0; }; lane_cpu ) 8> $ST/lane_cpu.lock &
( flock -n 9 || { echo "gpu lane already running"; exit 0; }; lane_gpu ) 9> $ST/lane_gpu.lock &
wait
ev "pipeline exit"
