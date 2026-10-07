"""학습한 planner로 평가 세트(D0~D3)와 검증 세트의 경로를 만든다 (9~11단계).

- 시작 노이즈는 장면 token마다 고정한다(crc32(token)을 시드로 한 정규분포). 같은 장면이면 모델과 결정이 달라도
  같은 노이즈에서 시작하므로, 모델끼리, 결정끼리 짝지어 비교할 수 있다. --noise_seed로 바꾸면 시드 안정성을 잰다.
- 출력: exp/eval/<run>/<set>.parquet (token, log_name, cf_name, sample_type 등 + pred_poses(24), Ours는 p, 근거 logit)

사용법: python scripts/predict_planner.py --run B2_seed0 [--sets D0 D1 D2 D3 val]
"""

import argparse
import os
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from yesman.model import ModelConfig, Planner, REASONS, encode_decisions
from yesman.train_data import decision_arrays, load_features

ROOT = Path(os.environ["YESMAN_ROOT"])
KEEP = ["token", "log_name", "cf_name", "sample_type", "decision", "flag", "reason", "category", "visibility",
        "status"]


def load_model(run: str) -> Planner:
    ck = torch.load(ROOT / f"exp/train/{run}/model.pt", map_location="cuda", weights_only=False)
    cfg = ModelConfig(**ck["cfg"])
    sd = ck["model"]
    m = Planner(cfg, sd["pose_mean"], sd["pose_std"]).cuda()
    m.load_state_dict(sd)
    return m.eval()


def set_table(name: str) -> pd.DataFrame:
    if name == "val":
        df = pd.read_parquet(ROOT / "exp/l3/train_bundle_val.parquet")
    elif name == "train5k":  # 학습 묶음에서 무작위 5천 개 (학습 데이터를 따르는지 보는 진단용, 시드 0)
        df = pd.read_parquet(ROOT / "exp/l3/train_bundle_train.parquet").sample(n=5000, random_state=0)
    else:
        df = pd.read_parquet(ROOT / f"data_lists/eval/{name}.parquet")
        if name == "D3":
            df = df[df.parse_status == "ok"]
    return df.reset_index(drop=True)


def token_noise(tokens, seed: int) -> torch.Tensor:
    z = np.stack([np.random.default_rng([zlib.crc32(t.encode()), seed]).standard_normal((8, 3)) for t in tokens])
    return torch.as_tensor(z, dtype=torch.float32, device="cuda")


@torch.no_grad()
def predict(model, feats, df, noise_seed=0, bs=2048):
    d = decision_arrays(df)
    t = lambda a: torch.as_tensor(a, device="cuda")  # noqa: E731
    dec = encode_decisions(t(d["lon"]), t(d["lon_s"]), t(d["lat"]), t(d["lat_s"]))
    rows = t(df.token.map(feats["row"]).to_numpy())
    z0 = token_noise(df.token.tolist(), noise_seed)
    out = {"poses": [], "flag_logit": [], "reason_logit": []}
    for lo in range(0, len(df), bs):
        sl = slice(lo, lo + bs)
        r = rows[sl]
        o = model.sample(feats["keyval"][r], feats["query_out"][r], dec[sl] if model.cfg.use_decision else None,
                         z0=z0[sl], bev=feats["bev_sem"][r] if model.cfg.use_bev_sem else None)
        for k in out:
            if k in o:
                out[k].append(o[k].cpu())
    res = df[[c for c in KEEP if c in df.columns]].copy()
    res["pred_poses"] = list(torch.cat(out["poses"]).numpy().reshape(len(df), -1))
    if out["flag_logit"]:
        res["p"] = torch.sigmoid(torch.cat(out["flag_logit"])).numpy()
        rl = torch.cat(out["reason_logit"]).numpy()
        for i, r_ in enumerate(REASONS):
            res[f"reason_logit_{r_}"] = rl[:, i]
    for c in [c for c in df.columns if c.startswith("d_seg")]:
        res[c] = df[c].to_numpy()  # 결정 (follow 판정에 쓴다)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--sets", nargs="+", default=["D0", "D1", "D2", "D3", "val"])
    ap.add_argument("--noise_seed", type=int, default=0)
    args = ap.parse_args()
    model = load_model(args.run)
    out = ROOT / "exp/eval" / args.run
    out.mkdir(parents=True, exist_ok=True)
    bev = model.cfg.use_bev_sem
    feats = {"navtest": load_features("navtest", bev_sem=bev), "navtrain": load_features("navtrain", bev_sem=bev)}
    for s in args.sets:
        t0 = time.time()
        df = set_table(s)
        res = predict(model, feats["navtrain" if s in ("val", "train5k") else "navtest"], df, args.noise_seed)
        suffix = "" if args.noise_seed == 0 else f"_noise{args.noise_seed}"
        res.to_parquet(out / f"{s}{suffix}.parquet", index=False)
        print(f"[{args.run}] {s}: {len(res):,} rows ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
