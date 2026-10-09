"""planner 학습 (8단계 과적합 확인, 9~10단계 비교 모델과 제안 방법).

모델 (yesman.md 9.3, 같은 디코더와 같은 학습 설정. 차이는 결정 입력, 학습 샘플 종류, 판단 head뿐이다)
  B1      결정 입력 없음, 원래 결정의 샘플(= 사람 경로)만
  B2      결정 입력, 원래 결정만
  Bcomply 결정 입력, 원래 결정 + CF⁺
  Ours    결정 입력, 원래 결정 + CF⁺ + CF⁻, 판단 head
  B3      결정 입력, 원래 결정 + CF⁺ + CF⁻, 판단 head 없음 (12단계)

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
    "B3": dict(use_decision=True, judge=False, sample_types=SAMPLE_TYPES),  # Ours에서 판단 head만 뺀 것 (12단계)
}


def batch_inputs(feats, s, idx):
    r = s.feat_row[idx]
    return (feats["keyval"][r], feats["query_out"][r], s.dec[idx], (feats["bev_sem"][r] if "bev_sem" in feats else None),
            (feats["obj"][r] if "obj" in feats else None))


@torch.no_grad()
def evaluate(model, feats, s, seeds=(0, 1, 2), n_steps=10, bs=1024):
    """DDIM으로 샘플링한 경로와 목표 경로의 오차 (샘플 종류별). seed마다 시작 노이즈를 바꾼다."""
    model.eval()
    preds, flags = [], []
    for seed in seeds:
        g = torch.Generator(device="cuda").manual_seed(seed)
        out = []
        for lo in range(0, len(s), bs):
            idx = torch.arange(lo, min(len(s), lo + bs), device="cuda")
            kv, qo, dec, bev, obj = batch_inputs(feats, s, idx)
            o = model.sample(kv, qo, dec if model.cfg.use_decision else None, n_steps, g, bev=bev, obj=obj)
            out.append(o["poses"])
            if seed == seeds[0] and model.cfg.judge:
                flags.append(o["flag_logit"])
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
    if flags:  # 판단 head: p = sigmoid(flag_logit), 1 = ACCEPT. 판단 토큰은 노이즈와 무관하므로 seed 하나로 충분하다
        p = torch.sigmoid(torch.cat(flags))
        res["judge"] = judge_metrics(p, s.flag, s.stype)
    return res, P[0]


def auc(score, label):
    """ROC AUC (label 1을 양성으로, score가 클수록 양성)."""
    order = torch.argsort(score)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(score) + 1, device=score.device, dtype=torch.float64)
    n1, n0 = float(label.sum()), float((1 - label).sum())
    return float((ranks[label > 0.5].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def judge_metrics(p, flag, stype, thr=0.5):
    out = {"auc": auc(p, flag), "acc@0.5": float(((p >= thr).float() == flag).float().mean())}
    for i, st in enumerate(SAMPLE_TYPES):
        m = stype == i
        if m.any():
            out[f"reject@0.5_{st}"] = float((p[m] < thr).float().mean())
    return out


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
    ap.add_argument("--neg_to_pos", type=float, default=1.0, help="CF⁻ 전체 무게 / CF⁺ 전체 무게 (14절: 1:1에서 시작)")
    ap.add_argument("--w_flag", type=float, default=1.0)
    ap.add_argument("--w_reason", type=float, default=1.0)
    ap.add_argument("--bundle_suffix", default="", help="학습 묶음 파일 이름 뒤에 붙인다 (예: _retarget, 10단계 시험)")
    ap.add_argument("--bev_sem", action="store_true", help="LTF BEV 지도 분할 조각 토큰을 장면 토큰에 더한다 (10단계 시험)")
    ap.add_argument("--obj", choices=["", "ltf", "gt", "gtvel"], default="",
                    help="물체 토큰 30개를 더한다 (12단계 B 진단): LTF 검출 결과 / 정답 위치 / 정답 위치 + 속도")
    ap.add_argument("--pred", choices=["x0", "v"], default="x0")
    ap.add_argument("--loss", choices=["l1", "mse"], default="l1")
    ap.add_argument("--balance_strength", action="store_true",
                    help="CF⁻ 안에서 세기 기준 CF⁻의 전체 무게를 도로 기준과 같게 한다 (CF⁻ 전체 무게는 그대로, 12단계 B)")
    ap.add_argument("--plausible_share", type=float, default=0.0,
                    help="CF⁻ 안에서 그럴싸한 CF⁻(source = plausible)의 무게 비중 (CF⁻ 전체 무게는 그대로, 12단계 C)")
    ap.add_argument("--keep_unseen", action="store_true", help="보이지 않는 원인 CF⁻도 학습에 쓴다 (기본: 뺀다, 10단계 결정)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = ROOT / "exp/train" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = MODELS[args.model]
    feats = load_features("navtrain", bev_sem=args.bev_sem, obj=args.obj)
    train = load_bundle("train" + args.bundle_suffix, feats, spec["sample_types"])
    if args.overfit_scenes:
        rng = np.random.default_rng(args.seed)
        scenes = rng.choice(sorted(train.df.token.unique()), size=args.overfit_scenes, replace=False)
        train = train.subset(train.df.token.isin(set(scenes)).to_numpy())
        val = train
    else:
        val = load_bundle("val" + args.bundle_suffix, feats, spec["sample_types"])
    if not args.keep_unseen:  # 카메라 시야 밖 원인의 CF⁻는 학습과 검증에서 뺀다 (10단계 사용자 결정)
        train = train.subset((train.df.visibility != "unseen").to_numpy())
        if val is not train:
            val = val.subset((val.df.visibility != "unseen").to_numpy())
    # 샘플 무게: 원래 결정과 CF⁺는 1씩(B-순응과 같은 비율), CF⁻는 전체 무게가 CF⁺의 neg_to_pos배가 되게 한다
    st = train.df.sample_type.to_numpy()
    w = np.ones(len(train))
    if (st == "cf_neg").any() and (st == "cf_pos").any():
        w[st == "cf_neg"] = args.neg_to_pos * (st == "cf_pos").sum() / (st == "cf_neg").sum()
        if args.balance_strength:
            cat = train.df.category.fillna("").to_numpy()
            neg, strength, road = st == "cf_neg", (st == "cf_neg") & (cat == "strength"), (st == "cf_neg") & (cat == "road")
            w[strength] *= road.sum() / strength.sum()
            w[neg] *= args.neg_to_pos * (st == "cf_pos").sum() / w[neg].sum()
            print(f"  balance_strength: 세기 CF⁻ {strength.sum():,}개, 무게 {w[strength].sum() / w[neg].sum():.2f} (CF⁻ 안), "
                  f"도로 {w[road].sum() / w[neg].sum():.2f}", flush=True)
        if args.plausible_share:
            neg = st == "cf_neg"
            src = train.df["source"].fillna("").to_numpy() if "source" in train.df else np.full(len(train), "")
            pl = neg & (src == "plausible")
            w[pl] *= args.plausible_share / (1 - args.plausible_share) * w[neg & ~pl].sum() / w[pl].sum()
            w[neg] *= args.neg_to_pos * (st == "cf_pos").sum() / w[neg].sum()
            print(f"  plausible_share: 그럴싸한 CF⁻ {pl.sum():,}개, 무게 {w[pl].sum() / w[neg].sum():.2f} (CF⁻ 안)", flush=True)
    weights = torch.as_tensor(w, device="cuda", dtype=torch.float)
    mix = {k: round(float(w[st == k].sum() / w.sum()), 3) for k in SAMPLE_TYPES if (st == k).any()}
    print(f"[{args.name}] model {args.model}: train {len(train):,} samples, batch mix {mix} "
          f"({train.df.sample_type.value_counts().to_dict()}), val {len(val):,}", flush=True)

    cfg = ModelConfig(use_decision=spec["use_decision"], judge=spec["judge"], n_layers=args.n_layers,
                      use_query_out=not args.no_query_out, pred=args.pred, loss=args.loss, use_bev_sem=args.bev_sem,
                      obj=args.obj)
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
        idx = torch.multinomial(weights, args.batch, replacement=True)
        kv, qo, dec, bev, obj = batch_inputs(feats, train, idx)
        losses = model.loss(kv, qo, dec, train.target[idx], train.flag[idx], train.reason[idx], bev=bev, obj=obj)
        loss = losses["traj"] + args.w_flag * losses.get("flag", 0) + args.w_reason * losses.get("reason", 0)
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
                for k, v in res.items() if k != "judge"), flush=True)
            if "judge" in res:
                print("  [eval judge] " + " ".join(f"{k} {v:.3f}" for k, v in res["judge"].items()), flush=True)
            log.write(json.dumps({"step": step, "eval": res}) + "\n")
    torch.save({"model": model.state_dict(), "cfg": cfg.to_dict(), "args": vars(args), "argv": sys.argv}, out_dir / "model.pt")
    res, pred = evaluate(model, feats, val)
    json.dump({"eval": res, "args": vars(args), "cfg": cfg.to_dict(), "n_params": n_params,
               "train_counts": train.df.sample_type.value_counts().to_dict(), "batch_mix": mix,
               "sec": time.time() - t0},
              open(out_dir / "eval.json", "w"), indent=1)
    # 그림용: 평가 샘플의 예측 경로 (seed 0)
    pd_out = val.df[["token", "log_name", "sample_type", "cf_name", "decision"]].copy()
    pd_out["pred_poses"] = list(pred.cpu().numpy().reshape(len(val), -1))
    pd_out["target_poses"] = list(val.target.cpu().numpy().reshape(len(val), -1))
    pd_out.to_parquet(out_dir / "pred.parquet", index=False)
    print(f"[{args.name}] done in {time.time() - t0:.0f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
