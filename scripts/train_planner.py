"""planner 학습 (8단계 과적합 확인, 9~10단계 비교 모델과 제안 방법).

모델 (yesman.md 9.3, 같은 디코더와 같은 학습 설정. 차이는 결정 입력, 학습 샘플 종류, 판단 head뿐이다)
  B1      결정 입력 없음, 원래 결정의 샘플(= 사람 경로)만
  B2      결정 입력, 원래 결정만
  Bcomply 결정 입력, 원래 결정 + CF⁺
  Ours    결정 입력, 원래 결정 + CF⁺ + CF⁻, 판단 head

사용법
  python scripts/train_planner.py --model Bcomply --overfit_scenes 100 --steps 3000 --name step08_overfit_Bcomply
  python scripts/train_planner.py --model B2 --steps 50000 --name B2_seed0

출력: exp/train/<name>/ (model.pt, log.jsonl, eval.json)
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from yesman.model import ModelConfig, Planner
from yesman.train_data import SAMPLE_TYPES, load_bundle, load_features, pose_stats

ROOT = Path(os.environ["YESMAN_ROOT"])
MODELS = {
    "B1": dict(use_decision=False, judge=False, sample_types=("original",)),
    "B2": dict(use_decision=True, judge=False, sample_types=("original",)),
    "Bcomply": dict(use_decision=True, judge=False, sample_types=("original", "cf_pos")),
    "Ours": dict(use_decision=True, judge=True, sample_types=SAMPLE_TYPES),
}


def batch_inputs(feats, s, idx):
    r = s.feat_row[idx]
    return feats["keyval"][r], feats["query_out"][r], s.dec[idx]


@torch.no_grad()
def evaluate(model, feats, s, seeds=(0, 1, 2), n_steps=10, bs=1024):
    """DDIM으로 샘플링한 경로와 목표 경로의 오차 (샘플 종류별). seed마다 시작 노이즈를 바꾼다."""
    model.eval()
    preds = []
    for seed in seeds:
        g = torch.Generator(device="cuda").manual_seed(seed)
        out = []
        for lo in range(0, len(s), bs):
            idx = torch.arange(lo, min(len(s), lo + bs), device="cuda")
            kv, qo, dec = batch_inputs(feats, s, idx)
            out.append(model.sample(kv, qo, dec if model.cfg.use_decision else None, n_steps, g)["poses"])
        preds.append(torch.cat(out))
    model.train()
    P = torch.stack(preds)  # (n_seed, N, 8, 3)
    err = (P[..., :2] - s.target[None, ..., :2]).norm(dim=-1)  # (n_seed, N, 8)
    ade, fde = err.mean(-1), err[..., -1]
    spread = (P[..., :2] - P.mean(0, keepdim=True)[..., :2]).norm(dim=-1).mean(-1)  # seed끼리 퍼짐
    res = {}
    for i, st in enumerate(SAMPLE_TYPES):
        m = s.stype == i
        if m.any():
            res[st] = {"n": int(m.sum()), "ade": float(ade[:, m].mean()), "fde": float(fde[:, m].mean()),
                       "ade_p95": float(ade[:, m].quantile(0.95)), "seed_spread": float(spread[:, m].mean())}
    return res, P[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--overfit_scenes", type=int, default=0, help="학습 묶음에서 장면 N개만 쓰고 그 장면으로 평가한다")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--no_query_out", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = ROOT / "exp/train" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = MODELS[args.model]
    feats = load_features("navtrain")
    train = load_bundle("train", feats, spec["sample_types"])
    if args.overfit_scenes:
        rng = np.random.default_rng(args.seed)
        scenes = rng.choice(sorted(train.df.token.unique()), size=args.overfit_scenes, replace=False)
        train = train.subset(train.df.token.isin(set(scenes)).to_numpy())
        val = train
    else:
        val = load_bundle("val", feats, spec["sample_types"])
    print(f"[{args.name}] model {args.model}: train {len(train):,} samples "
          f"({train.df.sample_type.value_counts().to_dict()}), val {len(val):,}", flush=True)

    cfg = ModelConfig(use_decision=spec["use_decision"], judge=spec["judge"], n_layers=args.n_layers,
                      use_query_out=not args.no_query_out)
    mean, std = pose_stats(train)
    model = Planner(cfg, mean, std).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params {n_params / 1e6:.2f} M, cfg {cfg.to_dict()}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / args.warmup) * 0.5 *
                                              (1 + math.cos(math.pi * min(i, args.steps) / args.steps)))
    log = open(out_dir / "log.jsonl", "w")
    t0, run = time.time(), {}
    for step in range(1, args.steps + 1):
        idx = torch.randint(len(train), (args.batch,), device="cuda")
        kv, qo, dec = batch_inputs(feats, train, idx)
        losses = model.loss(kv, qo, dec, train.target[idx], train.flag[idx], train.reason[idx])
        loss = sum(losses.values())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        for k, v in losses.items():
            run[k] = run.get(k, 0.0) + float(v)
        if step % 250 == 0:
            rec = {"step": step, **{k: v / 250 for k, v in run.items()}, "grad_norm": float(gn),
                   "lr": sched.get_last_lr()[0], "sec": round(time.time() - t0, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            run = {}
            if step % 1000 == 0:
                print("  " + " ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()),
                      flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            res, _ = evaluate(model, feats, val)
            print(f"  [eval step {step}] " + "  ".join(
                f"{k}: ADE {v['ade']:.3f} FDE {v['fde']:.3f} p95 {v['ade_p95']:.3f} spread {v['seed_spread']:.3f}"
                for k, v in res.items()), flush=True)
            log.write(json.dumps({"step": step, "eval": res}) + "\n")
    torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "args": vars(args), "argv": sys.argv}, out_dir / "model.pt")
    res, pred = evaluate(model, feats, val)
    json.dump({"eval": res, "args": vars(args), "cfg": cfg.to_dict(), "n_params": n_params,
               "train_counts": train.df.sample_type.value_counts().to_dict(), "sec": time.time() - t0},
              open(out_dir / "eval.json", "w"), indent=1)
    # 그림용: 평가 샘플의 예측 경로 (seed 0)
    pd_out = val.df[["token", "log_name", "sample_type", "cf_name", "decision"]].copy()
    pd_out["pred_poses"] = list(pred.cpu().numpy().reshape(len(val), -1))
    pd_out["target_poses"] = list(val.target.cpu().numpy().reshape(len(val), -1))
    pd_out.to_parquet(out_dir / "pred.parquet", index=False)
    print(f"[{args.name}] done in {time.time() - t0:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
