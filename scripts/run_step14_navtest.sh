#!/usr/bin/env bash
# 14단계 최종 모델을 navtest(13단계까지의 평가 세트, 14단계부터 개발 세트)에서 평가한다. 탐색 분석이다(사전 등록 8절:
# 새 모델들의 navtest 결과는 테스트 평가가 끝난 뒤에 본다. 2026-10-17 10:49 테스트 평가 끝).
# 세트: D0, D1, D2, D3x_{gemma31b,gemma12b,qwen8b}. 13단계 표와 같은 지표를 내기 위해 D0는 공식 채점(score_planner.py),
# D2는 전부, D3x는 실행 불가능한 결정만 행마다 채점한다. 비교 표는 scripts/step14_navtest_compare.py.
# 예측(GPU)은 하나씩, L1 판정·채점(CPU)은 자리 CPU_SLOTS개(기본 2)에서 겹쳐 돈다. 끝난 파일은 건너뛴다.
#
# 사용법: nohup setsid bash scripts/run_step14_navtest.sh > logs/step14/navtest.log 2>&1 &
# 다른 목록: LIST(줄마다 "이름 실행", 같은 실행은 한 번만), SETS, SEEDS, OUT 환경 변수로 바꾼다. SETS에 D0가 없으면 D0 공식 채점을 건너뛴다.
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
OUT=${OUT:-logs/step14/navtest}; mkdir -p $OUT/locks
LIST=${LIST:-exp/step14/select/final_models.txt}
SETS=${SETS:-"D0 D1 D2 D3x_gemma31b D3x_gemma12b D3x_qwen8b"}
SEEDS=${SEEDS:-"0 1"}
CPU_SLOTS=${CPU_SLOTS:-2}
note () { echo "=== $(date '+%m-%d %H:%M') $*"; }
cpu_run () {
  local i rc
  while true; do
    for i in $(seq 1 $CPU_SLOTS); do
      ( flock -n 9 || exit 99; "$@" ) 9> $OUT/locks/cpu$i; rc=$?
      [ $rc -ne 99 ] && return $rc
    done
    sleep 20
  done
}
eval_cpu () {
  local r=$1 d=exp/eval/$1 s
  for s in $SETS; do
    [ -f $d/${s}_l1.parquet ] || python scripts/follow_eval.py --run $r --sets $s >> $OUT/$r.log 2>&1 || { note "L1 판정 실패 $r $s"; return 1; }
  done
  [ -f $d/D2_scores.parquet ] || python scripts/score_rows.py --table $d/D2.parquet --out $d/D2_scores.parquet >> $OUT/$r.log 2>&1 \
    || { note "채점 실패 $r D2"; return 1; }
  for v in gemma31b gemma12b qwen8b; do
    [ -f $d/D3x_${v}_scores.parquet ] || python scripts/score_rows.py --table $d/D3x_${v}.parquet --out $d/D3x_${v}_scores.parquet \
      --only_status infeasible >> $OUT/$r.log 2>&1 || { note "채점 실패 $r D3x_$v"; return 1; }
  done
  case " $SETS " in *" D0 "*) ;; *) note "평가 끝 $r"; return 0 ;; esac
  [ -f $d/D0_score.json ] || python scripts/score_planner.py --run $r --threads 14 > $OUT/$r.score_planner.log 2>&1 \
    || { note "D0 채점 실패 $r"; return 1; }
  note "평가 끝 $r"
}
PIDS=""
for r in $(awk '$1 != "vote" && !seen[$2]++ {print $2}' $LIST); do
  for seed in $SEEDS; do
    run=${r}_seed$seed; d=exp/eval/$run; need=""
    for s in $SETS; do [ -f $d/$s.parquet ] || need="$need $s"; done
    if [ -n "$need" ]; then
      note "predict $run:$need"
      python scripts/predict_planner.py --run $run --sets $need >> $OUT/$run.log 2>&1 || { note "예측 실패 $run"; exit 1; }
    fi
    ( cpu_run eval_cpu $run ) & PIDS="$PIDS $!"
  done
done
fail=0; for p in $PIDS; do wait $p || fail=1; done
[ $fail = 0 ] || { note "평가 중 실패가 있다"; exit 1; }
note "끝"
