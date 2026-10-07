"""7단계: 고정 인코더(LTF)로 학습 장면과 평가 장면의 특징을 한 번 계산하여 저장한다.

사용법 (source scripts/setup/env.sh 후, GPU pod):
    python scripts/ltf_features.py --split navtest
    python scripts/ltf_features.py --split navtrain   # 센서를 받은 장면(data_lists/navtrain_sensor_tokens.txt)만

저장하는 것 (exp/features/ltf/<split>/, 장면 순서는 tokens.txt와 같다)
    keyval       (N, 65, 256) fp16  LTF 트랜스포머 디코더가 보는 토큰 (BEV 8x8=64개 + 차량 상태 1개, 위치 임베딩 포함)
    query_out    (N, 31, 256) fp16  LTF 디코더 출력 (경로 쿼리 1개 + 물체 쿼리 30개)
    status       (N, 8)       fp32  입력 차량 상태 (내비게이션 명령 4 + 속도 2 + 가속도 2)
    ltf_traj     (N, 8, 3)    fp32  LTF가 예측한 경로 (검증, 참고용)
    agent_states (N, 30, 5)   fp32  LTF가 예측한 BEV 박스 (x, y, heading, length, width). 9.4 인식/판단 분리 실험용
    agent_logits (N, 30)      fp32  BEV 박스 존재 logit
    bev_sem      (N, 128, 256) uint8 LTF가 예측한 BEV 지도 분할 (argmax). 9.4 실험용
    cam_sum      (N,)         fp64  입력 이미지(3x256x1024) 합. 장면-특징 짝 확인용

pod가 꺼져도 이어서 할 수 있도록 shard(기본 2000장면)마다 _shards/에 저장하고, 끝나면 하나로 합친다.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.agents.transfuser.transfuser_config import TransfuserConfig
from navsim.common.dataclasses import AgentInput

from yesman.data import scene_loader

ROOT = Path(os.environ["YESMAN_ROOT"])
CKPT = ROOT / "checkpoints/ltf/ltf_seed_0.ckpt"
FIELDS = {  # 이름: (모양, dtype)
    "keyval": ((65, 256), np.float16),
    "query_out": ((31, 256), np.float16),
    "status": ((8,), np.float32),
    "ltf_traj": ((8, 3), np.float32),
    "agent_states": ((30, 5), np.float32),
    "agent_logits": ((30,), np.float32),
    "bev_sem": ((128, 256), np.uint8),
    "cam_sum": ((), np.float64),
}


def load_agent() -> TransfuserAgent:
    agent = TransfuserAgent(config=TransfuserConfig(latent=True), lr=1e-4, checkpoint_path=str(CKPT))
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    agent.load_state_dict({k.replace("agent.", ""): v for k, v in sd.items()}, strict=True)
    return agent.eval()


def split_tokens(split: str):
    if split == "navtrain":
        return sorted(open(ROOT / "data_lists/navtrain_sensor_tokens.txt").read().split())
    return None  # navtest: scene filter 전체


class FeatureDataset(Dataset):
    def __init__(self, loader, tokens, builder):
        self.loader, self.tokens, self.builder = loader, tokens, builder

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        ld = self.loader
        agent_input = AgentInput.from_scene_dict_list(ld.scene_frames_dicts[self.tokens[i]], ld._original_sensor_path,
                                                      num_history_frames=ld._scene_filter.num_history_frames,
                                                      sensor_config=ld._sensor_config)
        f = self.builder.compute_features(agent_input)
        return i, f["camera_feature"], f["status_feature"]


@torch.no_grad()
def run_batch(agent, cam, status):
    model = agent._transfuser_model
    captured = {}
    h = model._tf_decoder.register_forward_hook(lambda m, inp, out: captured.update(keyval=inp[1], query_out=out))
    out = agent.forward({"camera_feature": cam, "status_feature": status})
    h.remove()
    return {
        "keyval": captured["keyval"].half(),
        "query_out": captured["query_out"].half(),
        "status": status,
        "ltf_traj": out["trajectory"],
        "agent_states": out["agent_states"],
        "agent_logits": out["agent_labels"],
        "bev_sem": out["bev_semantic_map"].argmax(1).to(torch.uint8),
        "cam_sum": cam.double().sum((1, 2, 3)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["navtest", "navtrain"], required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "exp/features/ltf")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--shard", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N장면만 (시험용)")
    args = ap.parse_args()

    out = args.out / args.split
    (out / "_shards").mkdir(parents=True, exist_ok=True)
    agent = load_agent().cuda()
    t0 = time.time()
    tokens = split_tokens(args.split)
    logs = None
    if tokens is not None:
        logs = open(ROOT / "data_lists/navtrain_sensor_logs.txt").read().split()
    loader = scene_loader(args.split, log_names=logs, tokens=tokens, sensor_config=agent.get_sensor_config())
    tokens = sorted(loader.tokens)
    if args.limit:
        tokens = tokens[: args.limit]
    print(f"[{args.split}] {len(tokens)} scenes (loader {time.time() - t0:.0f}s)", flush=True)
    (out / "tokens.txt").write_text("\n".join(tokens) + "\n")

    builder = agent.get_feature_builders()[0]
    n_shards = (len(tokens) + args.shard - 1) // args.shard
    for s in range(n_shards):
        path = out / "_shards" / f"{s:03d}.npz"
        if path.exists():
            continue
        lo, hi = s * args.shard, min(len(tokens), (s + 1) * args.shard)
        ds = FeatureDataset(loader, tokens[lo:hi], builder)
        dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers, pin_memory=True,
                        persistent_workers=False, prefetch_factor=4)
        buf = {k: np.zeros((hi - lo, *shape), dt) for k, (shape, dt) in FIELDS.items()}
        ts = time.time()
        for idx, cam, status in dl:
            res = run_batch(agent, cam.cuda(non_blocking=True), status.cuda(non_blocking=True))
            for k, v in res.items():
                buf[k][idx.numpy()] = v.cpu().numpy()
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **buf)
        tmp.rename(path)
        el = time.time() - ts
        print(f"  shard {s + 1}/{n_shards} [{lo}:{hi}] {el:.0f}s ({(hi - lo) / el:.1f} scenes/s)", flush=True)

    # 합치기: 필드마다 .npy 하나 (np.load(mmap_mode='r')로 빠르게 읽는다)
    shards = [np.load(out / "_shards" / f"{s:03d}.npz") for s in range(n_shards)]
    for k in FIELDS:
        np.save(out / f"{k}.npy", np.concatenate([sh[k] for sh in shards]))
    print(f"[{args.split}] done in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
