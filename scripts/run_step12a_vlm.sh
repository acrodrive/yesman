#!/usr/bin/env bash
# 12단계 A: 학습 장면(navtrain 22,700개)에 Qwen3-VL 8B와 Gemma 4 12B를 돌려 실제 VLM 결정을 모은다 (CF 만들기용).
# Gemma 4 31B는 평가 전용(학습에 쓰지 않음)이므로 지운다. 이어 하기 가능(중간 저장).
set -uo pipefail
cd /workspace/yesman
L=/workspace/yesman/logs/step12a
VLM="env VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=/root/hf/cache YESMAN_ROOT=/workspace/yesman /root/vlm/bin/python"
echo "=== qwen8b navtrain $(date +%H:%M)"
$VLM scripts/vlm_d3.py --scenes exp/d3/scenes_navtrain.parquet --name train_qwen8b --model /root/hf/Qwen3-VL-8B-Instruct > $L/vlm_qwen8b.log 2>&1
grep -E "saved|Error" $L/vlm_qwen8b.log | tail -2
[ -f exp/d3/train_qwen8b_raw.parquet ] || { echo "qwen 실패: 중단"; exit 1; }
rm -rf /root/hf/gemma-4-31B-it-qat-w4a16-ct /root/hf/Qwen3-VL-8B-Instruct
echo "=== download gemma12b $(date +%H:%M)"
[ -d /root/hf/gemma-4-12B-it ] || (cd /root && uvx --from "huggingface_hub[hf_xet,cli]" hf download google/gemma-4-12B-it --local-dir /root/hf/gemma-4-12B-it > $L/dl_gemma12b.log 2>&1)
echo "=== gemma12b navtrain $(date +%H:%M)"
$VLM scripts/vlm_d3.py --scenes exp/d3/scenes_navtrain.parquet --name train_gemma12b --model /root/hf/gemma-4-12B-it > $L/vlm_gemma12b.log 2>&1
grep -E "saved|Error" $L/vlm_gemma12b.log | tail -2
[ -f exp/d3/train_gemma12b_raw.parquet ] || { echo "gemma12b 실패: 중단"; exit 1; }
echo "=== VLM done $(date +%H:%M)"
