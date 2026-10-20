#!/usr/bin/env bash
# 14단계 사후 분석 (docs/TODO.md "아직 풀리지 않은 것")의 GPU 작업. 테스트 세트(T0~T3)는 쓰지 않는다.
#   5번: 결정 따르기를 나누는 데 필요한 예측. B3_s14_oursconf_seed0(Ours 구성의 B3, frz의 DiT 출처)의 val과 navtest D1,
#        OursDiT_s14_pl04_r1_jl2_frz_seed0(그 DiT를 고정하고 판단만 학습)의 navtest D1. L1 판정만 하고 채점은 하지 않는다
#   4번: B3_s14_menunl70_seed0 = menuonly(확장·plausible 모두 뺌)에 좌우만 바꾸지 않은 CF⁻의 무게 비중 0.705(기준 st80k와 같음)
#        → navtest D1, D2, D3x 3개 (scripts/run_step14_navtest.sh)
#
# 사용법: nohup setsid bash scripts/run_step14_posthoc_gpu.sh > logs/step14/posthoc_gpu.log 2>&1 < /dev/null &
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
LOG=logs/step14/posthoc; mkdir -p $LOG
note () { echo "=== $(date '+%m-%d %H:%M') $*"; }
gpu_wait () { while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; do sleep 60; done; }

# ---- 5번 예측 ----
pred () {  # pred <실행> <세트...>
  local r=$1; shift
  local s need=""
  for s in "$@"; do [ -f exp/eval/$r/$s.parquet ] || need="$need $s"; done
  if [ -n "$need" ]; then
    gpu_wait; note "predict $r:$need"
    python scripts/predict_planner.py --run $r --sets $need --val_bundle_suffix _step14_plaus >> $LOG/$r.log 2>&1 \
      || { note "예측 실패 $r"; exit 1; }
  fi
  for s in "$@"; do
    [ -f exp/eval/$r/${s}_l1.parquet ] || python scripts/follow_eval.py --run $r --sets $s >> $LOG/$r.log 2>&1 \
      || { note "L1 판정 실패 $r $s"; exit 1; }
  done
  note "5번 끝 $r"
}
pred B3_s14_oursconf_seed0 val D1 || exit 1
pred OursDiT_s14_pl04_r1_jl2_frz_seed0 D1 || exit 1

# ---- 4번 학습 ----
NAME=B3_s14_menunl70_seed0
if [ ! -f exp/train/$NAME/model.pt ]; then
  gpu_wait; note "train $NAME"
  python scripts/train_planner.py --model B3 --steps 80000 --seed 0 --bundle_suffix _step14_plaus --neg_to_pos 0.25 \
    --drop_source wide plausible --plausible_share 0 --nonlat_share 0.705 --name $NAME > $LOG/$NAME.train.log 2>&1
  [ -f exp/train/$NAME/model.pt ] || { note "학습 실패 $NAME"; exit 1; }
fi
note "학습 끝 $NAME: $(grep -m1 nonlat_share $LOG/$NAME.train.log)"
echo "menunl70_B3 B3_s14_menunl70" > exp/step14/contrib/runs3.txt
LIST=exp/step14/contrib/runs3.txt SETS="D1 D2 D3x_gemma31b D3x_gemma12b D3x_qwen8b" SEEDS=0 OUT=$LOG \
  bash scripts/run_step14_navtest.sh || { note "평가 실패"; exit 1; }
note "모두 끝"
