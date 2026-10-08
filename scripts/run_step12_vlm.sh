#!/usr/bin/env bash
# 12단계 (2) D3 확장: VLM 3개 x 3,000장면 (기존 D3 1,000 + 새 2,000, 시드 1). 프롬프트, 스키마, 판정은 7단계와 같다.
# 컨테이너 디스크가 작으므로 모델마다 쓰고 지운다.
set -uo pipefail
cd /workspace/yesman
L=logs/step12
VLM="env VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=/root/hf/cache YESMAN_ROOT=/workspace/yesman /root/vlm/bin/python"
until grep -q "GPU done" $L/b3.log; do sleep 30; done
echo "=== gemma12b ext $(date +%H:%M)"
$VLM scripts/vlm_d3.py --scenes exp/d3/scenes_ext.parquet --name d3x_gemma12b_ext --model /root/hf/gemma-4-12B-it > $L/vlm_gemma12b.log 2>&1
grep -E "saved" $L/vlm_gemma12b.log
rm -rf /root/hf/gemma-4-12B-it
echo "=== qwen8b $(date +%H:%M)"
$VLM scripts/vlm_d3.py --scenes exp/d3/scenes_3k.parquet --name d3x_qwen8b --model /root/hf/Qwen3-VL-8B-Instruct > $L/vlm_qwen8b.log 2>&1
grep -E "saved" $L/vlm_qwen8b.log
rm -rf /root/hf/Qwen3-VL-8B-Instruct
echo "=== download gemma31b $(date +%H:%M)"
(cd /root && uvx --from "huggingface_hub[hf_xet,cli]" hf download google/gemma-4-31B-it-qat-w4a16-ct --local-dir /root/hf/gemma-4-31B-it-qat-w4a16-ct > $L/dl_gemma31b.log 2>&1)
echo "=== gemma31b $(date +%H:%M)"
$VLM scripts/vlm_d3.py --scenes exp/d3/scenes_3k.parquet --name d3x_gemma31b --model /root/hf/gemma-4-31B-it-qat-w4a16-ct > $L/vlm_gemma31b.log 2>&1
grep -E "saved" $L/vlm_gemma31b.log
echo "=== VLM done $(date +%H:%M)"
