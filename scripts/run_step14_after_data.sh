#!/usr/bin/env bash
# 14단계: 데이터 준비가 끝나면 학습을 자동으로 잇는다 (2026-10-15 사용자 결정).
# 1) 데이터 준비의 마지막 단계(plaus)가 끝나기를 기다린다 (실패 표시가 있어도 기다린다. 고친 뒤 드라이버가 다시 돌면 이어진다)
# 2) 흐름 확인: QUICK=1 run_step14_train.sh (200 step, 검증 300장면, 이름 앞에 q_. 결과는 쓰지 않는다)
# 3) 진짜 학습: run_step14_train.sh
# 사건은 logs/step14/state/events.log에 남긴다. 테스트 평가(run_step14_final.sh)는 여기서 돌리지 않는다(사용자 확인 뒤 한 번만).
#
# 사용법: nohup setsid bash scripts/run_step14_after_data.sh > logs/step14/after_data.log 2>&1 &
set -u
cd /workspace/yesman
ST=logs/step14/state
ev () { echo "$(date '+%m-%d %H:%M') $*" >> $ST/events.log; echo "=== $(date '+%m-%d %H:%M') $*"; }

until [ -f $ST/plaus.ok ]; do sleep 120; done
ev "after_data: 데이터 준비 끝. 학습 흐름 확인(QUICK) 시작"
if ! QUICK=1 bash scripts/run_step14_train.sh > logs/step14/train_quick.log 2>&1; then
  ev "FAIL train QUICK (logs/step14/train_quick.log). 진짜 학습은 띄우지 않음"; exit 1
fi
ev "ok train QUICK. 진짜 학습 시작"
if bash scripts/run_step14_train.sh > logs/step14/train.log 2>&1; then
  ev "ok train. 테스트 평가는 사용자 확인 뒤 (scripts/run_step14_final.sh, CONFIRM=1)"
else
  ev "FAIL train (logs/step14/train.log)"; exit 1
fi
