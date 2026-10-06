"""NAVSIM 장면 불러오기 도우미."""

import os
from pathlib import Path
from typing import Iterable, Optional

from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.common.dataclasses import Scene, SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader

DATA_ROOT = Path(os.environ["OPENSCENE_DATA_ROOT"])
DEVKIT = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
SPLITS = {"navtest": "test", "navtrain": "trainval"}  # scene filter 이름 → 로그 폴더


def scene_filter(split: str, log_names: Optional[Iterable[str]] = None,
                 tokens: Optional[Iterable[str]] = None) -> SceneFilter:
    cfg = OmegaConf.load(DEVKIT / f"navsim/planning/script/config/common/train_test_split/scene_filter/{split}.yaml")
    f: SceneFilter = instantiate(cfg)
    if log_names is not None:
        f.log_names = sorted(set(log_names) & set(f.log_names)) if f.log_names else sorted(set(log_names))
    if tokens is not None:
        f.tokens = sorted(set(tokens) & set(f.tokens)) if f.tokens else sorted(set(tokens))
    return f


def scene_loader(split: str, log_names: Optional[Iterable[str]] = None, tokens: Optional[Iterable[str]] = None,
                 sensor_config: Optional[SensorConfig] = None) -> SceneLoader:
    """split(navtest, navtrain)의 장면 로더. log_names를 주면 그 로그만 읽는다(빠름)."""
    sub = SPLITS[split]
    return SceneLoader(DATA_ROOT / "navsim_logs" / sub, DATA_ROOT / "sensor_blobs" / sub,
                       scene_filter(split, log_names, tokens),
                       sensor_config=sensor_config or SensorConfig.build_no_sensors())


def scene_without_sensors(loader: SceneLoader, token: str) -> Scene:
    f = loader._scene_filter
    return Scene.from_scene_dict_list(loader.scene_frames_dicts[token], None, f.num_history_frames,
                                      f.num_future_frames, SensorConfig.build_no_sensors())
