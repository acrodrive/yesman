"""평가에 쓰는 세트의 이름 규칙 (14단계에서 정리).

세트 이름 = <기본 이름>[_noise<seed>][_vote<K>_<M>]. 예: val, D2, T2, val_vote10_3, T3_gemma31b_vote10_3
- val, train5k: navtrain 검증·학습 묶음 (장면 특징 navtrain, metric cache navtrain_sensor. 14단계 학습 장면 목록
  navtrain_step14_train이 있으면 그것을 쓴다. 기존 장면도 모두 들어 있는 목록이라 기존 실행의 결과는 같다)
- D0~D3, D3x_*: navtest 평가 세트 (13단계까지의 테스트, 14단계부터 개발 세트. 장면 특징 navtest, metric cache navtest)
- T0~T3_*: 14단계 새 테스트 세트 (navtrain의 받지 않았던 카메라 조각. 장면 특징 navtrain, metric cache navtrain_step14_test)
"""

import os
import re
from pathlib import Path

TEST_CACHE = "navtrain_step14_test"
TRAIN14_CACHE = "navtrain_step14_train"  # 기존 navtrain_sensor + 학습 추가 장면 (scripts/run_step14_data.sh cache_train)


def base_name(name: str) -> str:
    """접미사(_noise, _vote)를 뗀 이름."""
    return re.sub(r"(_noise\d+)?(_vote\d+_\d+)?$", "", name)


def split_of(name: str) -> str:
    """장면 특징과 장면 로더의 split."""
    b = base_name(name)
    return "navtrain" if b in ("val", "train5k") or b.startswith("T") else "navtest"


def cache_of(name: str) -> str:
    """metric cache 폴더 이름 (exp/metric_cache/<이름>)."""
    b = base_name(name)
    if b in ("val", "train5k"):
        m = Path(os.environ.get("NAVSIM_EXP_ROOT", "")) / f"metric_cache/{TRAIN14_CACHE}/metadata"
        return TRAIN14_CACHE if m.is_dir() else "navtrain_sensor"
    return TEST_CACHE if b.startswith("T") else "navtest"
