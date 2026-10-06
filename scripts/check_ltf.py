"""1단계 확인: LTF 공식 체크포인트를 현재 NAVSIM 코드로 불러오고, navtest 장면 하나에서 특징을 뽑는다.

사용법 (source scripts/setup/env.sh 후):
    python scripts/check_ltf.py [--ckpt PATH] [--num-scenes N]

확인하는 것
  1. 공식 경로(TransfuserAgent.initialize)로 체크포인트를 불러올 수 있는가
  2. state_dict 키가 모델과 정확히 맞는가 (strict load, 누락/잉여 0개)
  3. sensor_blobs/test 에 이미지가 있는 navtest 장면에서 특징이 나오는가
  4. LTF가 예측한 경로가 사람 경로와 가까운가 (가중치가 제대로 들어갔다는 간접 확인)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader

DATA_ROOT = Path(os.environ["OPENSCENE_DATA_ROOT"])
DEVKIT = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
NAVTEST_CFG = DEVKIT / "navsim/planning/script/config/common/train_test_split/scene_filter/navtest.yaml"


def load_agent(ckpt: Path) -> TransfuserAgent:
    agent = TransfuserAgent(config=TransfuserConfig(latent=True), lr=1e-4, checkpoint_path=str(ckpt))
    try:
        agent.initialize()  # 공식 경로 (strict=True)
        print("[load] TransfuserAgent.initialize(): OK")
    except Exception as e:
        # torch>=2.6 은 torch.load 기본값이 weights_only=True 라서 Lightning 체크포인트가 막힐 수 있다.
        print(f"[load] TransfuserAgent.initialize() 실패: {type(e).__name__}: {str(e).splitlines()[0]}")
        print("[load] weights_only=False 로 다시 불러온다 (공식 체크포인트이므로 신뢰한다)")
        state_dict = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        agent.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()}, strict=True)
    return agent.eval()


def check_keys(agent: TransfuserAgent, ckpt: Path) -> None:
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = {k.replace("agent.", ""): v for k, v in raw["state_dict"].items()}
    model_keys = set(agent.state_dict().keys())
    missing, unexpected = model_keys - set(sd), set(sd) - model_keys
    print(f"[keys] ckpt top-level: {sorted(raw.keys())}")
    print(f"[keys] params: ckpt {len(sd)}, model {len(model_keys)}, missing {len(missing)}, unexpected {len(unexpected)}")
    assert not missing and not unexpected, (sorted(missing)[:5], sorted(unexpected)[:5])
    assert "_transfuser_model._backbone.lidar_latent" in sd, "latent=True 용 파라미터가 없다"


def scenes_with_images(loader: SceneLoader, n: int):
    """sensor_blobs/test 에 3캠 이미지가 실제로 있는 장면만 고른다 (1단계에서는 첫 조각만 받았다)."""
    out = []
    for token in loader.tokens:
        frame = loader.scene_frames_dicts[token][loader._scene_filter.num_history_frames - 1]
        paths = [DATA_ROOT / "sensor_blobs/test" / frame["cams"][c]["data_path"] for c in ("CAM_L0", "CAM_F0", "CAM_R0")]
        if all(p.exists() for p in paths):
            out.append(token)
            if len(out) == n:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path(os.environ["YESMAN_ROOT"]) / "checkpoints/ltf/ltf_seed_0.ckpt")
    ap.add_argument("--num-scenes", type=int, default=5)
    args = ap.parse_args()

    print(f"torch {torch.__version__} (cuda build {torch.version.cuda}, cuda available {torch.cuda.is_available()})")
    agent = load_agent(args.ckpt)
    check_keys(agent, args.ckpt)

    scene_filter: SceneFilter = instantiate(OmegaConf.load(NAVTEST_CFG))
    loader = SceneLoader(
        data_path=DATA_ROOT / "navsim_logs/test",
        original_sensor_path=DATA_ROOT / "sensor_blobs/test",
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    tokens = scenes_with_images(loader, args.num_scenes)
    print(f"[data] navtest scenes in logs: {len(loader.tokens)}, checked with images: {len(tokens)}")
    assert tokens, "이미지가 있는 navtest 장면이 없다"

    # 백본 출력(장면 특징 후보)을 hook으로 잡는다
    captured = {}
    model = agent._transfuser_model
    model._backbone.register_forward_hook(lambda m, i, o: captured.update(bev_upscale=o[0], bev=o[1]))
    model._tf_decoder.register_forward_hook(lambda m, i, o: captured.update(query_out=o))

    builder = agent.get_feature_builders()[0]
    ade, fde = [], []
    for token in tokens:
        scene = loader.get_scene_from_token(token)
        feats = {k: v[None] for k, v in builder.compute_features(scene.get_agent_input()).items()}
        with torch.no_grad():
            out = agent.forward(feats)
        pred = out["trajectory"][0].numpy()  # (8, 3) x, y, heading
        gt = scene.get_future_trajectory(num_trajectory_frames=8).poses  # (8, 3)
        err = np.linalg.norm(pred[:, :2] - gt[:, :2], axis=1)
        ade.append(err.mean()), fde.append(err[-1])
        if token == tokens[0]:
            print(f"[feat] token {token}  log {scene.scene_metadata.log_name}")
            for k, v in {**feats, **captured}.items():
                print(f"[feat]   {k:15s} {tuple(v.shape)}  mean {v.float().mean():+.4f}  std {v.float().std():.4f}")
            print(f"[traj]   pred end {pred[-1].round(2)}  gt end {gt[-1].round(2)}")
        assert all(torch.isfinite(v).all() for v in captured.values())
    print(f"[traj] {len(tokens)} scenes: L2 ADE {np.mean(ade):.2f} m, FDE(4s) {np.mean(fde):.2f} m")
    print("OK")


if __name__ == "__main__":
    main()
