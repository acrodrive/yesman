"""학습한 planner로 평가 세트(D0~D3)와 검증 세트의 경로를 만든다 (9~11단계).

- 시작 노이즈는 장면 token마다 고정한다(crc32(token)을 시드로 한 정규분포). 같은 장면이면 모델과 결정이 달라도
  같은 노이즈에서 시작하므로, 모델끼리, 결정끼리 짝지어 비교할 수 있다. --noise_seed로 바꾸면 시드 안정성을 잰다.
- 출력: exp/eval/<run>/<set>.parquet (token, log_name, cf_name, sample_type 등 + pred_poses(24), Ours는 p, 근거 logit)

- 노이즈 투표 (14단계, DiT 내부에서 판단하는 OursDiT만): --vote_k K --vote_m M. 장면마다 시작 노이즈 K개로 한 배치에
  샘플링하고, REJECT(p < tau)가 M개 이상이면 거부한다. 최종 경로는 최종 판단과 같은 판단을 낸 샘플 중 확신이 가장 큰
  샘플의 경로이다(Planner.sample_vote). 노이즈 k = 0은 투표하지 않을 때의 노이즈와 같다.
  출력 열: p(고른 샘플의 p), vote_accept, vote_n_reject, vote_p_mean. 파일 이름 뒤에 _voteK_M이 붙는다.

사용법: python scripts/predict_planner.py --run B2_seed0 [--sets D0 D1 D2 D3 val]
        python scripts/predict_planner.py --run OursDiT_seed0 --vote_k 10 --vote_m 3
"""

import argparse
import os
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from yesman.eval_sets import split_of
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


def set_table(name: str, bundle_suffix: str = "") -> pd.DataFrame:
    if name == "val":
        df = pd.read_parquet(ROOT / f"exp/l3/train_bundle_val{bundle_suffix}.parquet")
    elif name == "train5k":  # 학습 묶음에서 무작위 5천 개 (학습 데이터를 따르는지 보는 진단용, 시드 0)
        df = pd.read_parquet(ROOT / "exp/l3/train_bundle_train.parquet").sample(n=5000, random_state=0)
    else:  # D*: navtest 평가 세트 (data_lists/eval), T*: 14단계 새 테스트 세트 (data_lists/eval_step14)
        folder = "eval_step14" if name.startswith("T") else "eval"
        df = pd.read_parquet(ROOT / f"data_lists/{folder}/{name}.parquet")
        if name.startswith(("D3", "T3")):  # VLM 결정: 파싱에 성공한 것만
            df = df[df.parse_status == "ok"]
    return df.reset_index(drop=True)


def token_noise(tokens, seed: int, k: int = 0) -> torch.Tensor:
    """장면 token마다 고정한 시작 노이즈 (N, 8, 3). k는 노이즈 투표의 몇 번째 노이즈인지(0은 투표하지 않을 때와 같다)."""
    key = lambda t: [zlib.crc32(t.encode()), seed] + ([k] if k else [])  # noqa: E731
    z = np.stack([np.random.default_rng(key(t)).standard_normal((8, 3)) for t in tokens])
    return torch.as_tensor(z, dtype=torch.float32, device="cuda")


@torch.no_grad()
def predict(model, feats, df, noise_seed=0, bs=2048, vote_k=0, vote_m=0, tau=0.5):
    d = decision_arrays(df)
    t = lambda a: torch.as_tensor(a, device="cuda")  # noqa: E731
    dec = encode_decisions(t(d["lon"]), t(d["lon_s"]), t(d["lat"]), t(d["lat_s"]))
    rows = t(df.token.map(feats["row"]).to_numpy())
    if vote_k:
        z0 = torch.stack([token_noise(df.token.tolist(), noise_seed, k) for k in range(vote_k)], 1)  # (N, K, 8, 3)
        bs = max(1, bs // vote_k)
    else:
        z0 = token_noise(df.token.tolist(), noise_seed)
    out = {"poses": [], "flag_logit": [], "reason_logit": [], "accept": [], "n_reject": [], "p": []}
    for lo in range(0, len(df), bs):
        sl = slice(lo, lo + bs)
        r = rows[sl]
        args = (feats["keyval"][r], feats["query_out"][r], dec[sl] if model.cfg.use_decision else None)
        kw = dict(z0=z0[sl], bev=feats["bev_sem"][r] if model.cfg.use_bev_sem else None,
                  obj=feats["obj"][r] if model.cfg.obj else None)
        o = model.sample_vote(*args, k=vote_k, m=vote_m, tau=tau, **kw) if vote_k else model.sample(*args, **kw)
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
    if out["accept"]:
        res["vote_accept"] = torch.cat(out["accept"]).numpy()
        res["vote_n_reject"] = torch.cat(out["n_reject"]).numpy()
        res["vote_p_mean"] = torch.cat(out["p"]).mean(1).numpy()
    for c in [c for c in df.columns if c.startswith("d_seg")]:
        res[c] = df[c].to_numpy()  # 결정 (follow 판정에 쓴다)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--sets", nargs="+", default=["D0", "D1", "D2", "D3", "val"])
    ap.add_argument("--noise_seed", type=int, default=0)
    ap.add_argument("--val_bundle_suffix", default="", help="val 세트로 쓸 검증 묶음 이름 뒤에 붙인다 (예: _step14)")
    ap.add_argument("--vote_k", type=int, default=0, help="노이즈 투표: 장면마다 시작 노이즈 개수 (0 = 투표 안 함)")
    ap.add_argument("--vote_m", type=int, default=0, help="노이즈 투표: REJECT가 이만큼 이상이면 거부")
    ap.add_argument("--tau", type=float, default=0.5, help="노이즈 하나의 판단 기준값: p < tau면 REJECT")
    args = ap.parse_args()
    assert not args.vote_k or args.vote_m >= 1, "--vote_k를 쓰면 --vote_m(1 이상)도 정해야 한다"
    model = load_model(args.run)
    out = ROOT / "exp/eval" / args.run
    out.mkdir(parents=True, exist_ok=True)
    bev = model.cfg.use_bev_sem
    obj = model.cfg.obj
    splits = {split_of(s) for s in args.sets}
    feats = {sp: load_features(sp, bev_sem=bev, obj=obj) for sp in splits}
    for s in args.sets:
        t0 = time.time()
        df = set_table(s, args.val_bundle_suffix)
        res = predict(model, feats[split_of(s)], df, args.noise_seed,
                      vote_k=args.vote_k, vote_m=args.vote_m, tau=args.tau)
        suffix = ("" if args.noise_seed == 0 else f"_noise{args.noise_seed}") + \
            (f"_vote{args.vote_k}_{args.vote_m}" if args.vote_k else "")
        res.to_parquet(out / f"{s}{suffix}.parquet", index=False)
        print(f"[{args.run}] {s}: {len(res):,} rows ({time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
