#!/usr/bin/env bash
# 1단계: NAVSIM devkit과 conda 환경을 네트워크 볼륨(/workspace)에 설치한다.
# 새 pod에서도 /workspace가 그대로 붙어 있으면 다시 실행할 필요가 없다.
set -euo pipefail

WORKSPACE=/workspace
CONDA_ROOT=$WORKSPACE/miniforge3
NAVSIM_DIR=$WORKSPACE/yesman/third_party/navsim
NAVSIM_COMMIT=0a380a9063d7162ec93d0f51e9990ebac585f720   # main, 2025-10-27 (v2.2 이후)
export PIP_CACHE_DIR=$WORKSPACE/.cache/pip

# 1) Miniforge
if [ ! -x $CONDA_ROOT/bin/conda ]; then
  wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh
  bash /tmp/miniforge.sh -b -p $CONDA_ROOT
fi
source $CONDA_ROOT/etc/profile.d/conda.sh

# 2) NAVSIM devkit (yesman repo에서는 gitignore. 버전은 위 커밋으로 고정)
[ -d $NAVSIM_DIR ] || git clone https://github.com/autonomousvision/navsim.git $NAVSIM_DIR
git -C $NAVSIM_DIR checkout -q $NAVSIM_COMMIT

# 3) conda 환경: environment.yml과 같은 구성(python 3.9, pip 23.3.1, nb_conda_kernels).
#    단 torch 2.0.1/torchvision 0.15.2는 RTX 5090(sm_120)을 지원하지 않으므로
#    torch 2.7.1/torchvision 0.22.1 (CUDA 12.8)로 바꾼다. 나머지 requirements는 그대로 쓴다.
if ! conda env list | grep -q "^navsim "; then
  conda create -y -n navsim python=3.9 pip=23.3.1 nb_conda_kernels -c conda-forge
fi
conda activate navsim

REQ=$(mktemp)
grep -v -E '^(torch|torchvision)==' $NAVSIM_DIR/requirements.txt > $REQ
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r $REQ
pip install --no-deps -e $NAVSIM_DIR
rm -f $REQ

python -c "import torch, navsim, nuplan; print('torch', torch.__version__, '| cuda build', torch.version.cuda)"
