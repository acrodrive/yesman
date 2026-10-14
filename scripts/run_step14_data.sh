#!/usr/bin/env bash
# 14단계 데이터 준비 (docs/step14_preregistration.md 2절, 4.1절, 6절). 단계 이름을 인자로 준다. 끊기면 같은 단계부터 다시 돌린다.
#   bash scripts/run_step14_data.sh download shards cache features testcf vlm freeze traincf bundle plaus
# download   남은 navtrain 카메라 24조각 (scripts/data/download_navsim.sh, 조각마다 _done 표시로 이어 받기)
# shards     용도별 장면 목록 (data_lists/step14/). EXTRA_VAL_FRAC 환경 변수로 학습 추가 로그의 검증 비율(기본 0.1, step14 문서 2.1절)
# cache      metric cache: 학습 추가 → navtrain_sensor에 더함, 테스트 → navtrain_step14_test, 예비 → navtrain_step14_reserve
# features   LTF 특징 (GPU). 카메라가 있는 navtrain 장면 전부를 navtrain_step14로 만든 뒤 navtrain과 바꿔 끼운다
# testcf     테스트 장면의 원래 결정 + 규칙 기반 CF 메뉴 전부를 L2로 판정 → T0~T2  (모델 결과는 보지 않는다)
# vlm        테스트 장면 전부에 Gemma 4 31B (GPU, VLM 환경 /root/vlm) → L2 판정 → T3_gemma31b
# freeze     T0~T3 체크섬과 README (이후 바꾸지 않는다)
# traincf    학습 + 검증 장면: 규칙 기반 CF 메뉴 전부, 확장된 규칙 기반 CF, plausible CF (모두 L2 판정)
# bundle     학습 묶음 (CF⁺ target = 사람 경로를 고친 경로, 실패하면 사람 경로에 가장 가까운 후보) → train_bundle_{train,val}_step14
# plaus      그럴싸함 점수를 붙인 묶음 → train_bundle_{train,val}_step14_plaus
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
L=logs/step14; mkdir -p $L exp/step14
S=data_lists/step14
W=${WORKERS:-24}
SHARDS="2 3 4 6 7 8 10 11 12 14 15 16 18 19 20 22 23 24 26 27 28 30 31 32"
VLM="env VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=/root/hf/cache YESMAN_ROOT=/workspace/yesman /root/vlm/bin/python"
G31=/root/hf/gemma-4-31B-it-qat-w4a16-ct
step () { echo "=== $1 $(date +%m-%d\ %H:%M)"; }

for stage in "$@"; do
  case $stage in
  download)
    step download
    du -sh /workspace 2>/dev/null | tail -1
    bash scripts/data/download_navsim.sh "navtrain $SHARDS" 2>&1 | tee -a $L/download.log | grep -E "done|FAILED"
    du -sh /workspace 2>/dev/null | tail -1 ;;
  shards)
    step shards
    python scripts/step14_shards.py --extra_val_frac ${EXTRA_VAL_FRAC:-0.1} 2>&1 | tee $L/shards.log | grep -v Warn ;;
  cache)
    step cache
    python scripts/run_metric_caching_tokens.py --split navtrain --tokens $S/train_extra_tokens.txt \
      --out exp/metric_cache/navtrain_sensor --workers 16 > $L/cache_train_extra.log 2>&1
    python scripts/run_metric_caching_tokens.py --split navtrain --tokens $S/test_tokens.txt \
      --out exp/metric_cache/navtrain_step14_test --workers 16 > $L/cache_test.log 2>&1
    python scripts/run_metric_caching_tokens.py --split navtrain --tokens $S/reserve_tokens.txt \
      --out exp/metric_cache/navtrain_step14_reserve --workers 16 > $L/cache_reserve.log 2>&1
    tail -2 $L/cache_*.log ;;
  features)
    step features
    python scripts/ltf_features.py --split navtrain --tokens $S/camera_all_tokens.txt --logs $S/camera_all_logs.txt \
      --name navtrain_step14 2>&1 | grep -v Warn | tee $L/features.log | tail -3
    [ -f exp/features/ltf/navtrain_step14/keyval.npy ] || { echo "특징 계산 실패: 중단"; exit 1; }
    # 기존 특징(2.3만 장면)은 navtrain_step13으로 남기고, 새 특징을 navtrain 이름으로 쓴다 (yesman/train_data.py가 이 이름을 읽는다)
    [ -d exp/features/ltf/navtrain_step13 ] || mv exp/features/ltf/navtrain exp/features/ltf/navtrain_step13
    mv exp/features/ltf/navtrain_step14 exp/features/ltf/navtrain ;;
  testcf)
    step testcf
    python scripts/l3_run.py --split navtrain --cache navtrain_step14_test --tokens $S/test_tokens.txt \
      --workers $W --name step14_test_cf 2>&1 | grep -v -E "Warn|Loading" | tee $L/testcf.log | tail -15
    python scripts/step14_testset.py --stage sets 2>&1 | grep -v Warn ;;
  vlm)
    step vlm
    python scripts/d3_prepare.py --from_table data_lists/eval_step14/T0.parquet --split navtrain \
      --out exp/d3/scenes_step14_test.parquet 2>&1 | grep -v Warn | tail -2
    [ -d /root/vlm ] || bash scripts/setup/install_vlm.sh
    [ -d $G31 ] || (cd /root && uvx --from "huggingface_hub[hf_xet,cli]" hf download google/gemma-4-31B-it-qat-w4a16-ct --local-dir $G31 > /workspace/yesman/$L/dl_gemma31b.log 2>&1)
    $VLM scripts/vlm_d3.py --scenes exp/d3/scenes_step14_test.parquet --name step14_test_gemma31b --model $G31 \
      --max_model_len 4096 > $L/vlm_gemma31b.log 2>&1
    grep -E "saved|Error" $L/vlm_gemma31b.log | tail -2
    python scripts/d3_judge.py --name step14_test_gemma31b --scenes exp/d3/scenes_step14_test.parquet --split navtrain \
      --cache navtrain_step14_test --workers $W --out data_lists/eval_step14/T3_gemma31b.parquet 2>&1 \
      | grep -v -E "Warn|Loading" | tail -6 ;;
  freeze)
    step freeze
    python scripts/step14_testset.py --stage freeze ;;
  traincf)
    step traincf
    python scripts/l3_run.py --split navtrain --cache navtrain_sensor --tokens $S/train_all_tokens.txt --skip_original \
      --workers $W --name step14_train_cf 2>&1 | grep -v -E "Warn|Loading" | tee $L/traincf_menu.log | tail -15
    python scripts/l3_run.py --split navtrain --cache navtrain_sensor --tokens $S/train_all_tokens.txt --skip_original \
      --wide --workers $W --name step14_train_cf_wide 2>&1 | grep -v -E "Warn|Loading" | tee $L/traincf_wide.log | tail -15
    python scripts/l3_plausible.py --targets $S/train_all_tokens.txt --pool $S/train_all_tokens.txt --cache navtrain_sensor \
      --workers $W --out exp/l3/step14_train_cf_plausible.parquet 2>&1 | grep -v -E "Warn|Loading" \
      | tee $L/traincf_plausible.log | tail -15 ;;
  bundle)
    step bundle
    python scripts/step14_bundle.py 2>&1 | grep -v Warn | tee $L/bundle.log
    for b in train val; do
      python scripts/l3_synth.py --bundle $b --input exp/l3/step14_bundle_${b}_raw.parquet \
        --out exp/l3/train_bundle_${b}_step14.parquet --workers $W 2>&1 | grep -v -E "Warn|Loading" | tee -a $L/synth.log | tail -5
    done ;;
  plaus)
    step plaus
    for b in train val; do
      python scripts/cf_plausibility.py --table exp/l3/train_bundle_${b}_step14.parquet --pool $S/train_all_tokens.txt \
        --out exp/l3/train_bundle_${b}_step14_plaus.parquet 2>&1 | grep -v Warn | tee -a $L/plaus.log
    done
    # 확인: 실제 VLM의 틀린 결정이 규칙 기반 CF⁻보다 그럴싸한가 (navtest 개발 세트, Gemma 4 31B)
    python scripts/cf_plausibility.py --table data_lists/eval/D3x_gemma31b.parquet --query_split navtest \
      --pool $S/train_all_tokens.txt --out exp/step14/D3x_gemma31b_plaus.parquet 2>&1 | grep -v Warn | tee -a $L/plaus.log ;;
  *) echo "모르는 단계: $stage"; exit 1 ;;
  esac
done
echo "=== all done $(date +%m-%d\ %H:%M)"
