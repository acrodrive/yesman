#!/usr/bin/env bash
# 14단계: 학습 step 수·크기 고르기, 장면 수 단계, CF 학습 구성 고르기, Ours 설정 고르기, 최종 학습 (docs/step14_preregistration.md 9절 5~8번).
# 고르는 일은 모두 검증 세트(val, 학습 묶음 train_bundle_val_step14_plaus)로 하고, 고를 때는 시드 0만 학습한다.
# 새 테스트 세트(T*)는 이 스크립트에서 쓰지 않는다. 테스트 평가는 scripts/run_step14_final.sh (한 번만).
#
# 다시 돌리면 끝난 학습(exp/train/<이름>/model.pt)과 평가(exp/eval/<이름>/val_*)는 건너뛴다. 고른 결과는 exp/step14/select/*.json,
# 고른 설정은 exp/step14/select/conf_*.txt, 최종 모델 목록은 exp/step14/final_models.txt (문서 이름, 코드 실행 이름 앞부분).
#
# 실행 이름 (코드 이름 + _s14_ + 설정): 예 B3_s14_st40k_seed0, OursDiT_s14_pl06_bs_seed0
# 기본 구성(2단계 이전): CF⁻ 무게 0.25(13단계 값), plausible CF⁻ 비중은 묶음 그대로(옵션 끔), 나머지 장치 끔
#
# 사용법: nohup setsid bash scripts/run_step14_train.sh > logs/step14/train.log 2>&1 &
#         QUICK=1 이면 step 수를 200으로, 검증 묶음을 300장면으로 줄이고 이름 앞에 q_를 붙여 흐름만 확인한다(결과는 쓰지 않는다)
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
SEL=exp/step14/select; LOG=logs/step14/train; mkdir -p $SEL $LOG
BUNDLE="--bundle_suffix _step14_plaus"
BASE_CF="--neg_to_pos 0.25"
VALSUF=_step14_plaus  # 검증 세트(val)로 쓸 묶음
Q=""
if [ "${QUICK:-0}" = 1 ]; then
  Q="q_"; SEL=$SEL/quick; VALSUF=_step14_plaus_quick; mkdir -p $SEL
  [ -f exp/l3/train_bundle_val${VALSUF}.parquet ] || python -c "
import pandas as pd, numpy as np
d = pd.read_parquet('exp/l3/train_bundle_val_step14_plaus.parquet')
t = np.random.default_rng(0).choice(sorted(d.token.unique()), 300, replace=False)
d[d.token.isin(set(t))].to_parquet('exp/l3/train_bundle_val${VALSUF}.parquet', index=False)"
fi
note () { echo "=== $(date '+%m-%d %H:%M') $*"; }
die () { note "중단: $*"; exit 1; }

gpu_free () {  # 다른 GPU 작업(데이터 준비의 VLM 등)과 겹치지 않게 기다린다
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; do sleep 60; done
}

train () {  # train <실행 이름> <코드 모델> [train_planner 인자...]
  local name=$1 model=$2; shift 2
  [ -f exp/train/$name/model.pt ] && return 0
  gpu_free
  note "train $name ($model $*)"
  local steps=${STEPS:-20000}; [ -n "$Q" ] && steps=200
  python scripts/train_planner.py --model $model --name $name --steps $steps ${SIZE_ARGS:-} $BUNDLE "$@" \
    > $LOG/$name.log 2>&1
  [ -f exp/train/$name/model.pt ] || die "학습 실패 $name ($LOG/$name.log)"
}

evalval () {  # evalval <실행 이름> [judge]  검증 세트 예측 → L1 일치 판정 → 행마다 채점 → (판단 모델) τ
  local name=$1 judge=${2:-}
  local d=exp/eval/$name
  if [ ! -f $d/val.parquet ]; then gpu_free; python scripts/predict_planner.py --run $name --sets val --val_bundle_suffix $VALSUF \
      > $LOG/$name.eval.log 2>&1 || die "예측 실패 $name"; fi
  [ -f $d/val_l1.parquet ] || python scripts/follow_eval.py --run $name --sets val >> $LOG/$name.eval.log 2>&1 || die "L1 판정 실패 $name"
  [ -f $d/val_scores.parquet ] || python scripts/score_rows.py --table $d/val.parquet --out $d/val_scores.parquet \
      >> $LOG/$name.eval.log 2>&1 || die "채점 실패 $name"
  if [ -n "$judge" ] && [ ! -f $d/flag_threshold.json ]; then
    python scripts/flag_threshold.py --run $name --val_only >> $LOG/$name.eval.log 2>&1 || die "τ 실패 $name"
  fi
}

best () {  # best <select json> → 고른 실행 이름
  python -c "import json,sys; b=json.load(open(sys.argv[1]))['best']; print(b.split(':')[0] if b else '')" $1
}
select_cfg () {  # select_cfg <이름> <select_config 인자...>  표는 로그(stderr)로, 고른 실행 이름은 stdout으로
  local out=$SEL/$1.json; shift
  [ -f $out ] || python scripts/select_config.py --out $out "$@" > $SEL/$(basename $out .json).txt 2>&1 \
    || { echo "고르기 실패 $out" >&2; return 1; }
  tail -12 $SEL/$(basename $out .json).txt >&2
  local b; b=$(best $out); [ -n "$b" ] || { echo "고른 후보 없음 $out" >&2; return 1; }
  echo $b
}
upper_args () { echo "--upper_follow ${Q}Bcomply_s14_seed0 --upper_safe ${Q}B1_s14_seed0"; }

# ---- 4.2 학습 step 수 (B3, 기본 구성, 지금 크기, 시드 0) ----
note "4.2 step 수"
for st in 20 40 80 120; do
  STEPS=$((st * 1000)) train ${Q}B3_s14_st${st}k_seed0 B3 $BASE_CF
  evalval ${Q}B3_s14_st${st}k_seed0
done
b=$(select_cfg steps --rule steps --cands ${Q}B3_s14_st20k_seed0 ${Q}B3_s14_st40k_seed0 ${Q}B3_s14_st80k_seed0 ${Q}B3_s14_st120k_seed0) || die "step 수"
STEPS=${b#${Q}B3_s14_st}; STEPS=$(( ${STEPS%k_seed0} * 1000 )); export STEPS
note "고른 step 수 $STEPS"

# ---- 4.2 모델 크기 (고른 step 수에서) ----
note "4.2 모델 크기"
STEPTAG=st$((STEPS / 1000))k
train ${Q}B3_s14_${STEPTAG}_L6_seed0 B3 $BASE_CF --n_layers 6;    evalval ${Q}B3_s14_${STEPTAG}_L6_seed0
train ${Q}B3_s14_${STEPTAG}_W384_seed0 B3 $BASE_CF --d_model 384; evalval ${Q}B3_s14_${STEPTAG}_W384_seed0
b=$(select_cfg size --rule size --base ${Q}B3_s14_${STEPTAG}_seed0 --cands ${Q}B3_s14_${STEPTAG}_L6_seed0 ${Q}B3_s14_${STEPTAG}_W384_seed0) || die "크기"
case $b in *_L6_seed0) export SIZE_ARGS="--n_layers 6";; *_W384_seed0) export SIZE_ARGS="--d_model 384";; *) export SIZE_ARGS="";; esac
echo "STEPS=$STEPS SIZE_ARGS=$SIZE_ARGS" > $SEL/conf_train.txt
note "고른 크기: '${SIZE_ARGS}'"
# 이후 모든 모델은 STEPS, SIZE_ARGS로 학습한다. 기본 구성의 B3(고른 step 수·크기) = B3_s14_base
BASE_B3=${Q}B3_s14_${STEPTAG}_seed0
case "$SIZE_ARGS" in "--n_layers 6") BASE_B3=${Q}B3_s14_${STEPTAG}_L6_seed0;; "--d_model 384") BASE_B3=${Q}B3_s14_${STEPTAG}_W384_seed0;; esac

# ---- 4.1 장면 수 단계 (보고만, B3 기본 구성) ----
note "4.1 장면 수 단계 (보고만)"
train ${Q}B3_s14_sh0_seed0 B3 $BASE_CF --extra_shards;                                   evalval ${Q}B3_s14_sh0_seed0
train ${Q}B3_s14_sh4_seed0 B3 $BASE_CF --extra_shards 2 10 18 26;                       evalval ${Q}B3_s14_sh4_seed0
train ${Q}B3_s14_sh8_seed0 B3 $BASE_CF --extra_shards 2 10 18 26 6 14 22 30;            evalval ${Q}B3_s14_sh8_seed0
[ -f $SEL/ladder.txt ] || python scripts/select_config.py --rule steps --cands ${Q}B3_s14_sh0_seed0 ${Q}B3_s14_sh4_seed0 \
  ${Q}B3_s14_sh8_seed0 $BASE_B3 > $SEL/ladder.txt 2>&1   # 고르지 않는다. 표만 남긴다 (+16 = 기본 구성 B3)
cat $SEL/ladder.txt | head -6

# ---- 상한선 모델 (B0 = 코드 B1, B2 = 코드 Bcomply) ----
note "상한선 모델"
train ${Q}B1_s14_seed0 B1;      evalval ${Q}B1_s14_seed0
train ${Q}Bcomply_s14_seed0 Bcomply; evalval ${Q}Bcomply_s14_seed0

# ---- 4.3~4.5 CF 학습 구성 고르기 (B3, Ours 각자) ----
choose_cf () {  # choose_cf <코드 모델> → $SEL/conf_<모델>.txt
  local M=$1 J=""; [ $M = OursDiT ] && J=judge
  local conf_file=$SEL/conf_$M.txt
  [ -f $conf_file ] && { note "$M 구성 이미 고름: $(cat $conf_file)"; return 0; }
  note "4.3 $M: plausible CF⁻ 비중"
  for p in 04 06 085; do
    train ${Q}${M}_s14_pl${p}_seed0 $M $BASE_CF --plausible_share 0.${p#0}; evalval ${Q}${M}_s14_pl${p}_seed0 $J
  done
  local cur; cur=$(select_cfg ${M}_2a --rule upper $(upper_args) --cands ${Q}${M}_s14_pl04_seed0 ${Q}${M}_s14_pl06_seed0 ${Q}${M}_s14_pl085_seed0) || die "$M 비중"
  local pl=${cur#${Q}${M}_s14_pl}; pl=${pl%_seed0}
  local tag="pl${pl}" conf="$BASE_CF --plausible_share 0.${pl#0}"
  for dev in bs:--balance_strength bt:--balance_by_type pw:"--plaus_weight strength"; do
    local k=${dev%%:*} a=${dev#*:}
    note "4.3 $M: $tag + $k"
    train ${Q}${M}_s14_${tag}_${k}_seed0 $M $conf $a; evalval ${Q}${M}_s14_${tag}_${k}_seed0 $J
    local b; b=$(select_cfg ${M}_2_${tag}_${k} --rule upper $(upper_args) --cands $cur ${Q}${M}_s14_${tag}_${k}_seed0) || die "$M $k"
    if [ "$b" != "$cur" ]; then cur=$b; tag=${tag}_${k}; conf="$conf $a"; fi   # 점수가 오르면 켠 채로 다음 장치로
  done
  note "4.4 $M: CF⁻ 무게 (0.25 = $cur)"
  local c2; c2=$(echo $conf | sed 's/--neg_to_pos 0.25//')
  train ${Q}${M}_s14_${tag}_r05_seed0 $M $c2 --neg_to_pos 0.5; evalval ${Q}${M}_s14_${tag}_r05_seed0 $J
  train ${Q}${M}_s14_${tag}_r1_seed0 $M $c2 --neg_to_pos 1;    evalval ${Q}${M}_s14_${tag}_r1_seed0 $J
  local b; b=$(select_cfg ${M}_3 --rule upper $(upper_args) --cands $cur ${Q}${M}_s14_${tag}_r05_seed0 ${Q}${M}_s14_${tag}_r1_seed0) || die "$M 비율"
  case $b in *_r05_seed0) conf="$c2 --neg_to_pos 0.5"; tag=${tag}_r05;; *_r1_seed0) conf="$c2 --neg_to_pos 1"; tag=${tag}_r1;; esac
  echo "$tag|$conf" > $conf_file
  note "$M 구성: $tag ($conf)"
}
choose_cf B3
choose_cf OursDiT
B3_TAG=$(cut -d'|' -f1 $SEL/conf_B3.txt); B3_CONF=$(cut -d'|' -f2 $SEL/conf_B3.txt)
O_TAG=$(cut -d'|' -f1 $SEL/conf_OursDiT.txt); O_CONF=$(cut -d'|' -f2 $SEL/conf_OursDiT.txt)
O_BASE=${Q}OursDiT_s14_${O_TAG}_seed0

# ---- 이전 Ours (ablation): Ours의 구성 그대로 ----
note "이전 Ours"
train ${Q}Ours_s14_${O_TAG}_seed0 Ours $O_CONF; evalval ${Q}Ours_s14_${O_TAG}_seed0 judge

# ---- 5.2 Ours 설정 ① 분기 위치, ② 학습 방식 ----
auc_of () { python -c "
import sys; sys.argv=['x']; sys.path.insert(0,'scripts'); import select_config as s
print(s.measure('$1','val')['auc_kind'])"; }
O_CUR=$O_BASE; O_ARGS=""
if [ ! -f $SEL/ours_branch.txt ]; then
  a_new=$(auc_of $O_BASE); a_old=$(auc_of ${Q}Ours_s14_${O_TAG}_seed0)
  note "① 결정 종류 안 AUC: Ours $a_new, 이전 Ours $a_old"
  if python -c "import sys; sys.exit(0 if float('$a_new') < float('$a_old') else 1)"; then
    train ${Q}OursDiT_s14_${O_TAG}_jl2_seed0 OursDiT $O_CONF --judge_layer 2; evalval ${Q}OursDiT_s14_${O_TAG}_jl2_seed0 judge
    b=$(select_cfg ours_branch --rule auc $(upper_args) --base $O_BASE --cands ${Q}OursDiT_s14_${O_TAG}_jl2_seed0) || die "①"
    [ "$b" != "$O_BASE" ] && { O_CUR=$b; O_ARGS="--judge_layer 2"; }
  fi
  echo "$O_CUR|$O_ARGS" > $SEL/ours_branch.txt
fi
O_CUR=$(cut -d'|' -f1 $SEL/ours_branch.txt); O_ARGS=$(cut -d'|' -f2 $SEL/ours_branch.txt)
if [ ! -f $SEL/ours_train_mode.txt ]; then
  note "② 학습 방식: DiT를 먼저 학습(B3, Ours 구성)하고 고정"
  train ${Q}B3_s14_oursconf_seed0 B3 $O_CONF
  FZ=${Q}OursDiT_s14_${O_TAG}${O_ARGS:+_jl2}_frz_seed0
  train $FZ OursDiT $O_CONF $O_ARGS --freeze_dit --init_from ${Q}B3_s14_oursconf_seed0; evalval $FZ judge
  b=$(select_cfg ours_train_mode --rule auc $(upper_args) --base $O_CUR --cands $FZ) || die "②"
  if [ "$b" = "$FZ" ]; then echo "$FZ|$O_ARGS --freeze_dit --init_from ${Q}B3_s14_oursconf_seed{S}" > $SEL/ours_train_mode.txt
  else echo "$O_CUR|$O_ARGS" > $SEL/ours_train_mode.txt; fi
fi
O_FINAL=$(cut -d'|' -f1 $SEL/ours_train_mode.txt); O_FARGS=$(cut -d'|' -f2 $SEL/ours_train_mode.txt)
note "Ours 설정: $O_FINAL ($O_FARGS)"

# ---- 5.2 ③ 노이즈 투표 m (K = 10) ----
TAU=$(python -c "import json; print(json.load(open('exp/eval/$O_FINAL/flag_threshold.json'))['tau'])")
for m in 1 2 3 5; do
  d=exp/eval/$O_FINAL
  [ -f $d/val_vote10_${m}.parquet ] || { gpu_free; python scripts/predict_planner.py --run $O_FINAL --sets val --val_bundle_suffix $VALSUF \
      --vote_k 10 --vote_m $m --tau $TAU >> $LOG/$O_FINAL.vote.log 2>&1 || die "투표 예측 실패 m=$m"; }
  [ -f $d/val_vote10_${m}_l1.parquet ] || python scripts/follow_eval.py --run $O_FINAL --sets val_vote10_${m} >> $LOG/$O_FINAL.vote.log 2>&1
  [ -f $d/val_vote10_${m}_scores.parquet ] || python scripts/score_rows.py --table $d/val_vote10_${m}.parquet \
      --out $d/val_vote10_${m}_scores.parquet >> $LOG/$O_FINAL.vote.log 2>&1
done
b=$(select_cfg ours_vote_m --rule frog5 --cands $O_FINAL:val_vote10_1 $O_FINAL:val_vote10_2 $O_FINAL:val_vote10_3 $O_FINAL:val_vote10_5) || die "투표 m"
VOTE_M=$(python -c "import json; print(json.load(open('$SEL/ours_vote_m.json'))['best'].split('_')[-1])")
note "투표 m = $VOTE_M"

# ---- 최종 모델 목록과 시드 1 학습 ----
B3_FINAL=${Q}B3_s14_${B3_TAG}_seed0
cat > $SEL/final_models.txt <<EOF
B0 ${Q}B1_s14
B1 ${Q}B2_s14
B2 ${Q}Bcomply_s14
B3 ${B3_FINAL%_seed0}
OursPrev ${Q}Ours_s14_${O_TAG}
Ours ${O_FINAL%_seed0}
vote vote10_${VOTE_M}
EOF
note "최종 모델"; cat $SEL/final_models.txt
train ${Q}B2_s14_seed0 B2; evalval ${Q}B2_s14_seed0
train ${Q}B1_s14_seed1 B1 --seed 1;           evalval ${Q}B1_s14_seed1
train ${Q}B2_s14_seed1 B2 --seed 1;           evalval ${Q}B2_s14_seed1
train ${Q}Bcomply_s14_seed1 Bcomply --seed 1; evalval ${Q}Bcomply_s14_seed1
train ${B3_FINAL%_seed0}_seed1 B3 $B3_CONF --seed 1; evalval ${B3_FINAL%_seed0}_seed1
train ${Q}Ours_s14_${O_TAG}_seed1 Ours $O_CONF --seed 1; evalval ${Q}Ours_s14_${O_TAG}_seed1 judge
if echo "$O_FARGS" | grep -q freeze_dit; then
  train ${Q}B3_s14_oursconf_seed1 B3 $O_CONF --seed 1
  train ${O_FINAL%_seed0}_seed1 OursDiT $O_CONF ${O_FARGS//\{S\}/1} --seed 1
else
  train ${O_FINAL%_seed0}_seed1 OursDiT $O_CONF $O_FARGS --seed 1
fi
evalval ${O_FINAL%_seed0}_seed1 judge
note "학습 끝. 테스트 평가는 scripts/run_step14_final.sh (사용자 확인 뒤 한 번만)"
