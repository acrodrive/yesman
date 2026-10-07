#!/usr/bin/env bash
# 7단계 VLM(D3)용 환경. vLLM 0.31이 torch 2.13을 쓰므로 navsim 환경과 따로 둔다.
# 컨테이너 디스크(/root)에 둔다: 한 번만 쓰는 환경이고(약 10 GB), 가중치(24 GB)도 다시 받을 수 있다. 볼륨 용량을 아낀다.
set -euo pipefail
uv venv -q --python 3.11 /root/vlm
VIRTUAL_ENV=/root/vlm uv pip install -q vllm==0.31.0 "huggingface_hub[hf_xet]" pillow pandas pyarrow
uvx --from "huggingface_hub[hf_xet,cli]" hf download google/gemma-4-12B-it --local-dir /root/hf/gemma-4-12B-it
# 실행: FlashInfer 샘플러는 nvcc가 필요하므로 끈다 (greedy 디코딩이라 쓰지 않는다)
#   VLLM_USE_FLASHINFER_SAMPLER=0 YESMAN_ROOT=/workspace/yesman /root/vlm/bin/python scripts/vlm_d3.py --name d3
