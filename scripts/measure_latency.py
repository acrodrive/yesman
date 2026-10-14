"""14단계: planner 추론 지연을 잰다 (docs/step14_preregistration.md 5.2절 노이즈 투표의 K를 정하기 전에).

장면 하나(batch 1)를 넣었을 때 경로 + 판단을 내는 시간을 잰다. 장면 특징(LTF)은 미리 계산한 값과 같은 모양의 무작위 텐서를
쓴다(지연은 값과 무관하다). LTF 인코더 자체의 지연은 --ltf로 따로 잰다(카메라 3장 + 차량 상태, 무작위 입력).
  sample        노이즈 1개 (B0~B3, 이전 Ours, Ours 노이즈 1개)
  vote K        노이즈 K개를 한 batch로 (Ours 노이즈 투표, Planner.sample_vote)
GPU에서는 torch.cuda.synchronize로 끝을 맞추고, 준비 실행(warmup) 뒤 반복 실행의 중앙값과 p95를 낸다.

사용법: python scripts/measure_latency.py --run OursDiT_seed0 [--ks 1 5 10] [--ltf] [--out logs/step14/latency.json]
        python scripts/measure_latency.py --random_init --device cpu   # 가중치 없이 구조만 (로컬 확인용)
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from yesman.model import ModelConfig, Planner, encode_decisions


def timeit(fn, device, n=100, warmup=10):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="")
    ap.add_argument("--random_init", action="store_true", help="가중치 없이 새 Ours 구조로 잰다")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    ap.add_argument("--n_steps", type=int, default=10)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--ltf", action="store_true", help="LTF 인코더(카메라 → 장면 특징)의 지연도 잰다 (GPU, NAVSIM 환경)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dev = args.device

    if args.random_init:
        model = Planner(ModelConfig(judge=True, judge_from="dit"), torch.zeros(8, 3), torch.ones(8, 3))
    else:
        ck = torch.load(Path(os.environ["YESMAN_ROOT"]) / f"exp/train/{args.run}/model.pt", map_location="cpu",
                        weights_only=False)
        model = Planner(ModelConfig(**ck["cfg"]), ck["model"]["pose_mean"], ck["model"]["pose_std"])
        model.load_state_dict(ck["model"])
    model = model.to(dev).eval()
    kv, qo = torch.randn(1, 65, 256, device=dev), torch.randn(1, 31, 256, device=dev)
    dec = encode_decisions(*(torch.zeros(1, 2, device=dev) for _ in range(4))) if model.cfg.use_decision else None
    res = {"run": args.run or "random_init", "device": dev,
           "gpu": torch.cuda.get_device_name() if dev == "cuda" else "", "n_steps": args.n_steps,
           "params_M": sum(p.numel() for p in model.parameters()) / 1e6}
    with torch.no_grad():
        res["sample"] = timeit(lambda: model.sample(kv, qo, dec, args.n_steps), dev, args.n)
        if model.cfg.judge and model.cfg.judge_from == "dit":
            for k in args.ks:
                res[f"vote_{k}"] = timeit(lambda: model.sample_vote(kv, qo, dec, k=k, m=1, n_steps=args.n_steps), dev, args.n)
        if args.ltf:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from ltf_features import load_agent
            agent = load_agent().to(dev)
            feats = {"camera_feature": torch.randn(1, 3, 256, 1024, device=dev), "status_feature": torch.randn(1, 8, device=dev)}
            res["ltf"] = timeit(lambda: agent._transfuser_model(feats), dev, args.n)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
