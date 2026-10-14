"""14단계: CF의 그럴싸함 점수 (docs/step14_preregistration.md 6절).

장면 S의 결정 d에 대해, 다른 로그에서 내비게이션 명령이 같고 속도 차이가 2 m/s 이내인 장면 중 LTF 장면 특징
(keyval 평균, 길이 1로 정규화)의 cosine 유사도가 가장 높은 K개 장면(닮은 장면)을 찾고, 그 장면들에서 사람이 한 결정이
d와 같은 비율을 낸다.
  plaus_kind      결정 종류(구간마다 앞뒤·좌우 행동, 강도 무시)가 같은 비율
  plaus_strength  결정 종류가 같고 강도 차이가 ±0.2 안인 비율 (6.2의 ≈에서 이웃 라벨 허용 없음과 같음. keep lane 강도는 보지 않음)
  plaus_n         찾은 닮은 장면 수 (조건을 만족하는 장면이 K보다 적으면 K보다 작다)
닮은 장면 후보는 --pool 장면이다(테스트·예비 장면을 빼기 위해). plausible CF는 결정을 빌려 온 장면(src_token)을 이웃에서
뺀다(순환 방지). 사람 결정은 L1 라벨(exp/l1/navtrain_labels.parquet)이다.

사용법:
  python scripts/cf_plausibility.py --table exp/l3/train_bundle_train_step14.parquet --pool data_lists/step14/train_all_tokens.txt
  python scripts/cf_plausibility.py --table data_lists/eval/D3x_gemma31b.parquet --query_split navtest --pool ...   # 확인용
출력: 기본 <table>_plaus.parquet (원래 행 순서, 위 세 열 추가)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from yesman.decision import LAT_ACTIONS, LON_ACTIONS
from yesman.train_data import decision_arrays

ROOT = Path(os.environ["YESMAN_ROOT"])
EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
FEAT = ROOT / "exp/features/ltf"
DV, TOL = 2.0, 0.2
KEEP = LAT_ACTIONS.index("keep_lane")


def scene_emb(split: str, device):
    tokens = (FEAT / split / "tokens.txt").read_text().split()
    kv = torch.from_numpy(np.load(FEAT / split / "keyval.npy")).to(device).float().mean(1)
    st = torch.from_numpy(np.load(FEAT / split / "status.npy")).to(device).float()
    return tokens, torch.nn.functional.normalize(kv, dim=1), st[:, :4].argmax(1), st[:, 4:6].norm(dim=1)


def label_arrays(labels: pd.DataFrame, tokens, device):
    """사람 결정(L1 라벨)을 (N, 2) 배열로. 라벨이 없거나 invalid면 valid = False."""
    lab = labels.reindex(tokens)
    valid = lab.valid.fillna(False).astype(bool).to_numpy().copy()
    lab = lab.fillna({f"seg{k}_{c}": v for k in (1, 2) for c, v in (("lon", LON_ACTIONS[0]), ("lat", LAT_ACTIONS[0]),
                                                                       ("lon_strength", 0.0), ("lat_strength", 0.0))})
    a = decision_arrays(lab, prefix="")
    t = {k: torch.as_tensor(v, device=device) for k, v in a.items()}
    t["valid"] = torch.as_tensor(valid, device=device)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True, help="token, log_name, d_seg* 열이 있는 표 (src_token이 있으면 뺀다)")
    ap.add_argument("--pool", type=Path, required=True, help="닮은 장면 후보 (줄마다 token, navtrain)")
    ap.add_argument("--query_split", default="navtrain", help="표 장면의 특징 폴더 (navtrain / navtest)")
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_parquet(args.table)
    labels = pd.read_parquet(EXP / "l1/navtrain_labels.parquet").drop_duplicates("token").set_index("token")
    p_tok, p_emb, p_cmd, p_v = scene_emb("navtrain", dev)
    in_pool = set(args.pool.read_text().split())
    pm = torch.as_tensor([t in in_pool for t in p_tok], device=dev)
    p_tok, p_emb, p_cmd, p_v = [p_tok[i] for i in pm.nonzero()[:, 0].tolist()], p_emb[pm], p_cmd[pm], p_v[pm]
    p_log = labels.log_name.reindex(p_tok).to_numpy()
    p_lab = label_arrays(labels, p_tok, dev)
    q_tok, q_emb, q_cmd, q_v = scene_emb(args.query_split, dev)
    q_row = {t: i for i, t in enumerate(q_tok)}
    logs = pd.factorize(pd.Series(np.concatenate([p_log, df.log_name.to_numpy()])))[0]
    p_lid, q_lid = torch.as_tensor(logs[:len(p_log)], device=dev), torch.as_tensor(logs[len(p_log):], device=dev)
    src = df.src_token.fillna("").tolist() if "src_token" in df else [""] * len(df)
    p_idx = {t: i for i, t in enumerate(p_tok)}
    src_i = torch.as_tensor([p_idx.get(s, -1) for s in src], device=dev)
    d = {k: torch.as_tensor(v, device=dev) for k, v in decision_arrays(df).items()}
    qi = torch.as_tensor(df.token.map(q_row).to_numpy(), device=dev)

    kind, strength, n = [], [], []
    for lo in range(0, len(df), 1024):
        sl = slice(lo, lo + 1024)
        r = qi[sl]
        sim = q_emb[r] @ p_emb.T
        bad = (q_lid[sl, None] == p_lid[None]) | (q_cmd[r, None] != p_cmd[None]) | ((q_v[r, None] - p_v[None]).abs() > DV)
        bad |= ~p_lab["valid"][None]
        s = src_i[sl]
        bad[torch.arange(len(s), device=dev)[s >= 0], s[s >= 0]] = True
        sim[bad] = -2
        top = sim.topk(min(args.k, sim.shape[1]), dim=1)
        ok = top.values > -2
        nb = top.indices
        same_kind = ((p_lab["lon"][nb] == d["lon"][sl, None]) & (p_lab["lat"][nb] == d["lat"][sl, None])).all(-1)
        close = ((p_lab["lon_s"][nb] - d["lon_s"][sl, None]).abs() <= TOL + 1e-6) & \
            ((d["lat"][sl, None] == KEEP) | ((p_lab["lat_s"][nb] - d["lat_s"][sl, None]).abs() <= TOL + 1e-6))
        same_str = same_kind & close.all(-1)
        cnt = ok.sum(1).clamp(min=1)
        kind.append(((same_kind & ok).sum(1) / cnt).cpu())
        strength.append(((same_str & ok).sum(1) / cnt).cpu())
        n.append(ok.sum(1).cpu())
    df["plaus_kind"] = torch.cat(kind).numpy()
    df["plaus_strength"] = torch.cat(strength).numpy()
    df["plaus_n"] = torch.cat(n).numpy()
    out = args.out or args.table.with_name(args.table.stem + "_plaus.parquet")
    df.to_parquet(out, index=False)
    by = "source" if "source" in df else "cf_name"
    sub = df[df.sample_type == "cf_neg"] if "sample_type" in df else df
    print(sub.groupby(by)[["plaus_kind", "plaus_strength", "plaus_n"]].mean().round(3).to_string())
    print(f"saved {out}")


if __name__ == "__main__":
    main()
