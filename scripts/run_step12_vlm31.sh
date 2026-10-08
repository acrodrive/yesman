#!/usr/bin/env bash
# 12단계 (2): Gemma 4 31B (run_step12_vlm.sh에서 다운로드 로그 경로 실수로 건너뛴 것을 다시 돌림)
set -uo pipefail
cd /workspace/yesman
L=/workspace/yesman/logs/step12
VLM="env VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=/root/hf/cache YESMAN_ROOT=/workspace/yesman /root/vlm/bin/python"
echo "=== download gemma31b $(date +%H:%M)"
(cd /root && uvx --from "huggingface_hub[hf_xet,cli]" hf download google/gemma-4-31B-it-qat-w4a16-ct --local-dir /root/hf/gemma-4-31B-it-qat-w4a16-ct > $L/dl_gemma31b.log 2>&1)
tail -1 $L/dl_gemma31b.log
echo "=== gemma31b $(date +%H:%M)"
$VLM scripts/vlm_d3.py --scenes exp/d3/scenes_3k.parquet --name d3x_gemma31b --model /root/hf/gemma-4-31B-it-qat-w4a16-ct > $L/vlm_gemma31b.log 2>&1
grep -E "saved|Error" $L/vlm_gemma31b.log | tail -3
echo "=== VLM31 done $(date +%H:%M)"
