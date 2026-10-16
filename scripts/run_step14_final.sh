#!/usr/bin/env bash
# 14단계: 새 테스트 세트(T0~T2, T3_gemma31b)에서 최종 모델을 한 번 평가한다 (docs/step14_preregistration.md 7절, 8절, 9절 8번).
# 모델 목록은 scripts/run_step14_train.sh가 만든 exp/step14/select/final_models.txt (문서 이름, 코드 실행 이름 앞부분, 시드 0·1).
# 테스트 세트는 한 번만 평가한다. 그래서 CONFIRM=1이 없으면 돌지 않고, 한 번 돈 뒤(logs/step14/final/STARTED)에는 다시 돌지 않는다.
# 중간에 끊겼을 때만 RESUME=1로 이어 돈다(끝난 파일은 건너뛴다). 이어 돈 사실은 STARTED 파일에 남는다.
#
# 단계: 예측(predict_planner) → L1 일치 판정(follow_eval) → 행마다 채점(score_rows) → step14_eval.py(지표, 미리 정한 비교, 판정)
# 채점은 채점 결과를 쓰는 세트만 한다: T0(주행 점수), T2·T3(불가능한 결정의 안전율). T1(CF⁺)은 결정–경로 일치율, 과잉 거부,
# flag–경로 불일치만 쓰므로(L1과 판단) 채점하지 않는다.
# 판단 모델의 τ는 검증 세트에서 이미 정한 exp/eval/<run>/flag_threshold.json을 그대로 쓴다.
# 예측(GPU)은 하나씩 차례로, L1 판정과 채점(CPU)은 자리 CPU_SLOTS개(기본 2)에서 다음 예측과 겹쳐 돈다. 결과는 차례로 돌린 것과 같다.
#
# 사용법: CONFIRM=1 nohup setsid bash scripts/run_step14_final.sh > logs/step14/final.log 2>&1 &
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
OUT=logs/step14/final; mkdir -p $OUT
LIST=exp/step14/select/final_models.txt
SETS="T0 T1 T2 T3_gemma31b"
SCORE_SETS="T0 T2 T3_gemma31b"  # 채점 결과를 쓰는 세트
[ "${CONFIRM:-0}" = 1 ] || { echo "새 테스트 세트는 한 번만 평가한다. 확인했으면 CONFIRM=1을 붙여 다시 실행한다."; exit 1; }
[ -f $LIST ] || { echo "$LIST 없음 (run_step14_train.sh가 끝나야 한다)"; exit 1; }
sha256sum -c --quiet data_lists/eval_step14/SHA256SUMS 2>/dev/null \
  || (cd data_lists/eval_step14 && sha256sum -c --quiet SHA256SUMS) || { echo "테스트 세트 체크섬이 맞지 않는다"; exit 1; }
if [ -f $OUT/STARTED ]; then
  [ "${RESUME:-0}" = 1 ] || { echo "이미 평가를 시작했다($OUT/STARTED). 끊긴 것을 이어 돌릴 때만 RESUME=1"; exit 1; }
  echo "resume $(date '+%m-%d %H:%M')" >> $OUT/STARTED
else
  { echo "start $(date '+%m-%d %H:%M')"; cat $LIST; } > $OUT/STARTED
fi

run_of () { awk -v k=$1 '$1 == k {print $2}' $LIST; }
VOTE=$(run_of vote)
note () { echo "=== $(date '+%m-%d %H:%M') $*"; }

CPU_SLOTS=${CPU_SLOTS:-2}; LOCKD=$OUT/locks; mkdir -p $LOCKD
cpu_run () {  # cpu_run <명령...>  CPU 자리 하나를 잡고 돈다 (자리가 없으면 기다린다)
  local i rc
  while true; do
    for i in $(seq 1 $CPU_SLOTS); do
      ( flock -n 9 || exit 99; "$@" ) 9> $LOCKD/cpu$i; rc=$?
      [ $rc -ne 99 ] && return $rc
    done
    sleep 20
  done
}
eval_cpu () {  # eval_cpu <실행 이름> <접미사>  L1 판정(모든 세트) → 채점(SCORE_SETS)
  local r=$1 suffix=$2 d=exp/eval/$1 s
  for s in $SETS; do
    [ -f $d/${s}${suffix}_l1.parquet ] || python scripts/follow_eval.py --run $r --sets ${s}${suffix} >> $OUT/$r.log 2>&1 \
      || { note "L1 판정 실패 $r $s"; return 1; }
    [[ " $SCORE_SETS " == *" $s "* ]] || continue
    [ -f $d/${s}${suffix}_scores.parquet ] || python scripts/score_rows.py --table $d/${s}${suffix}.parquet \
      --out $d/${s}${suffix}_scores.parquet >> $OUT/$r.log 2>&1 || { note "채점 실패 $r $s"; return 1; }
  done
  note "평가 끝 $r$suffix"
}
PIDS=""
eval_run () {  # eval_run <실행 이름> [투표 인자]  예측(GPU, 지금 바로) → eval_cpu(뒤에서, CPU 자리)
  local r=$1 vote_args=${2:-} suffix=""
  [ -n "$vote_args" ] && suffix=_$VOTE
  local d=exp/eval/$r
  local need=""
  for s in $SETS; do [ -f $d/${s}${suffix}.parquet ] || need="$need $s"; done
  if [ -n "$need" ]; then
    note "predict $r$suffix:$need"
    python scripts/predict_planner.py --run $r --sets $need $vote_args >> $OUT/$r.log 2>&1 || { note "예측 실패 $r"; exit 1; }
  fi
  ( cpu_run eval_cpu $r "$suffix" ) &
  PIDS="$PIDS $!"
}

for k in B0 B1 B2 B3 OursPrev Ours; do
  for seed in 0 1; do eval_run $(run_of $k)_seed$seed; done
done
# Ours 노이즈 투표 (K = 10, m은 검증 세트로 고름). τ는 시드마다 그 실행의 값.
# 과잉 거부 5% 이하인 m이 없었으면(final_models.txt의 "vote none") 투표는 평가하지 않는다 (2026-10-16 사용자 결정 C)
if [ "$VOTE" != none ]; then
  M=${VOTE#vote10_}
  for seed in 0 1; do
    r=$(run_of Ours)_seed$seed
    TAU=$(python -c "import json; print(json.load(open('exp/eval/$r/flag_threshold.json'))['tau'])")
    eval_run $r "--vote_k 10 --vote_m $M --tau $TAU"
  done
fi
fail=0; for p in $PIDS; do wait $p || fail=1; done
[ $fail = 0 ] || { note "평가 중 실패가 있다. 고친 뒤 RESUME=1로 이어 돈다"; exit 1; }

note "step14_eval"
python scripts/step14_eval.py --models B0=$(run_of B0) B1=$(run_of B1) B2=$(run_of B2) B3=$(run_of B3) \
  OursPrev=$(run_of OursPrev) Ours=$(run_of Ours) $([ "$VOTE" != none ] && echo "--vote Ours=$VOTE") --out $OUT/eval.md 2>&1 | grep -v Warn
echo "done $(date '+%m-%d %H:%M')" >> $OUT/STARTED
note "끝. 결과 $OUT/eval.md"
