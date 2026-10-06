# 모든 작업 전에 `source /workspace/yesman/scripts/setup/env.sh` 로 불러온다.
# pod를 다시 띄우면 컨테이너 디스크(~/.bashrc 등)는 초기화되므로, 환경변수는 이 파일에 둔다.

export WORKSPACE=/workspace
export YESMAN_ROOT=$WORKSPACE/yesman
export CONDA_ROOT=$WORKSPACE/miniforge3

# NAVSIM (docs/install.md 기준). 데이터셋은 /workspace/datasets/navsim 아래에 NAVSIM 공식 구조로 둔다.
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export OPENSCENE_DATA_ROOT=$WORKSPACE/datasets/navsim
export NUPLAN_MAPS_ROOT=$OPENSCENE_DATA_ROOT/maps
export NAVSIM_DEVKIT_ROOT=$YESMAN_ROOT/third_party/navsim
export NAVSIM_EXP_ROOT=$YESMAN_ROOT/exp

# 이 저장소의 패키지(yesman/)를 import할 수 있게 한다
export PYTHONPATH=$YESMAN_ROOT${PYTHONPATH:+:$PYTHONPATH}

# pip / torch / HF 캐시도 볼륨에 둔다 (컨테이너 디스크는 pod 종료 시 지워진다)
export PIP_CACHE_DIR=$WORKSPACE/.cache/pip
export TORCH_HOME=$WORKSPACE/.cache/torch
export HF_HOME=$WORKSPACE/.cache/huggingface

source $CONDA_ROOT/etc/profile.d/conda.sh
conda activate navsim
