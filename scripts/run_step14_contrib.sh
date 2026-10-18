#!/usr/bin/env bash
# 14단계 기여 분석 2번 (탐색): 학습 묶음 안에서 무엇이 처음 보는 VLM 오답의 안전율을 올렸는지 가르는 B3 두 개를 학습하고
# navtest(개발 세트) D1, D2, D3x에서 평가한다. 기준은 같은 기본 구성(CF⁻ 무게 0.25, plausible 비중 0.6, 8만 step)의
# B3_s14_st80k_seed0이다. 테스트 세트는 쓰지 않는다.
#   B3_s14_nowide_seed0   확장 CF(source = wide)를 CF⁺·CF⁻ 모두 뺀다
#   B3_s14_neg026_seed0   CF⁻를 무작위로 26%만 남긴다(장면당 CF⁻ 수가 13단계 묶음 수준, 종류 구성은 그대로)
#   (2차) 출처 2x2와 batch 구성. 14단계 메뉴 CF⁻의 바뀐 종류 구성은 13단계와 같고, 새 종류(세로 방향·세기·여러 개)는
#   확장과 plausible에서만 나온다. plausible 비중 0.6이면 CF⁻ 무게의 60%가 plausible이다
#   B3_s14_noplaus_seed0  plausible CF(CF⁺·CF⁻)를 빼고 plausible 비중 0
#   B3_s14_menuonly_seed0 확장·plausible을 모두 빼고 메뉴 CF만 (CF⁻ 종류 구성이 13단계와 거의 같다)
#   B3_s14_menuneg_seed0  (3차) 확장·plausible의 CF⁻만 빼고 CF⁺는 남긴다 (menuonly의 효과가 CF⁻에서 오는지 CF⁺에서 오는지)
#   B3_s14_pos024_seed0   CF⁺를 무작위로 23.55%만 남긴다(batch 구성 원래 결정 33% / CF⁺ 53% / CF⁻ 13%가 13단계와 같다)
#
# 사용법: nohup setsid bash scripts/run_step14_contrib.sh > logs/step14/contrib2.log 2>&1 < /dev/null &
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
LOG=logs/step14/contrib; mkdir -p $LOG
note () { echo "=== $(date '+%m-%d %H:%M') $*"; }
BASE="--model B3 --steps 80000 --seed 0 --bundle_suffix _step14_plaus --neg_to_pos 0.25 --plausible_share 0.6"
train () {  # train <이름> [인자...]
  local name=$1; shift
  [ -f exp/train/$name/model.pt ] && return 0
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; do sleep 60; done  # 다른 GPU 작업과 겹치지 않게
  note "train $name ($*)"
  python scripts/train_planner.py $BASE --name $name "$@" > $LOG/$name.train.log 2>&1
  [ -f exp/train/$name/model.pt ] || { note "학습 실패 $name"; exit 1; }
}
train B3_s14_nowide_seed0 --drop_source wide
train B3_s14_neg026_seed0 --neg_keep_frac 0.26
train B3_s14_noplaus_seed0 --drop_source plausible --plausible_share 0
train B3_s14_menuonly_seed0 --drop_source wide plausible --plausible_share 0
train B3_s14_pos024_seed0 --pos_keep_frac 0.2355
train B3_s14_menuneg_seed0 --drop_source wide plausible --drop_neg_only --plausible_share 0
printf '%s\n' "nowide_B3 B3_s14_nowide" "neg026_B3 B3_s14_neg026" "noplaus_B3 B3_s14_noplaus" "menuonly_B3 B3_s14_menuonly" \
  "pos024_B3 B3_s14_pos024" "menuneg_B3 B3_s14_menuneg" > exp/step14/contrib/runs2.txt
LIST=exp/step14/contrib/runs2.txt SETS="D1 D2 D3x_gemma31b D3x_gemma12b D3x_qwen8b" SEEDS=0 OUT=$LOG \
  bash scripts/run_step14_navtest.sh || { note "평가 실패"; exit 1; }
note "끝"
