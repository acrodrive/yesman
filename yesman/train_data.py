"""학습 데이터: 학습 묶음(exp/l3/train_bundle_*.parquet) + LTF 특징(exp/features/ltf/) → GPU 텐서.

특징은 작으므로(navtrain 1.15 GB, fp16) GPU에 통째로 올리고, batch는 GPU에서 행 번호로 꺼낸다.
샘플 한 줄 = (장면 특징 행 번호, 결정, 목표 경로, flag, 판단 근거, 샘플 종류).
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch

from yesman.decision import LAT_ACTIONS, LON_ACTIONS
from yesman.model import N_SEG, REASONS, encode_decisions

ROOT = Path(os.environ["YESMAN_ROOT"])
FEAT = ROOT / "exp/features/ltf"
SAMPLE_TYPES = ("original", "cf_pos", "cf_neg")


def load_features(split: str, device="cuda") -> Dict[str, object]:
    tokens = FEAT.joinpath(split, "tokens.txt").read_text().split()
    out = {"tokens": tokens, "row": {t: i for i, t in enumerate(tokens)}}
    for k in ("keyval", "query_out"):
        out[k] = torch.from_numpy(np.load(FEAT / split / f"{k}.npy")).to(device)
    return out


def decision_arrays(df: pd.DataFrame, prefix: str = "d_") -> Dict[str, np.ndarray]:
    """결정 열(d_seg{k}_lon ...) → 행동 번호와 세기 배열 (N, N_SEG)."""
    lon_id, lat_id = {a: i for i, a in enumerate(LON_ACTIONS)}, {a: i for i, a in enumerate(LAT_ACTIONS)}
    k = range(1, N_SEG + 1)
    return {
        "lon": np.stack([df[f"{prefix}seg{i}_lon"].map(lon_id).to_numpy() for i in k], 1).astype(np.int64),
        "lon_s": np.stack([df[f"{prefix}seg{i}_lon_strength"].to_numpy() for i in k], 1).astype(np.float32),
        "lat": np.stack([df[f"{prefix}seg{i}_lat"].map(lat_id).to_numpy() for i in k], 1).astype(np.int64),
        "lat_s": np.stack([df[f"{prefix}seg{i}_lat_strength"].to_numpy() for i in k], 1).astype(np.float32),
    }


@dataclass
class Samples:
    feat_row: torch.Tensor  # (N,) 특징 행 번호
    dec: torch.Tensor  # (N, N_SEG, DEC_IN)
    target: torch.Tensor  # (N, 8, 3)
    flag: torch.Tensor  # (N,) 1 = ACCEPT, 0 = REJECT
    reason: torch.Tensor  # (N, 3)
    stype: torch.Tensor  # (N,) SAMPLE_TYPES 번호
    df: pd.DataFrame  # 원래 표 (같은 순서)

    def __len__(self):
        return len(self.feat_row)

    def subset(self, mask) -> "Samples":
        m = torch.as_tensor(np.asarray(mask), device=self.feat_row.device)
        idx = m.nonzero()[:, 0] if m.dtype == torch.bool else m
        return Samples(self.feat_row[idx], self.dec[idx], self.target[idx], self.flag[idx], self.reason[idx],
                       self.stype[idx], self.df.iloc[idx.cpu().numpy()].reset_index(drop=True))


def load_bundle(name: str, feats: Dict, sample_types: Iterable[str] = SAMPLE_TYPES,
                tokens: Optional[Iterable[str]] = None, device="cuda") -> Samples:
    """name: train / val. sample_types: 쓸 샘플 종류(B2: original, B-순응: original + cf_pos, Ours: 전부)."""
    df = pd.read_parquet(ROOT / f"exp/l3/train_bundle_{name}.parquet")
    df = df[df.sample_type.isin(list(sample_types))]
    if tokens is not None:
        df = df[df.token.isin(set(tokens))]
    df = df.reset_index(drop=True)
    d = decision_arrays(df)
    t = lambda a, dt=None: torch.as_tensor(a, device=device, dtype=dt)  # noqa: E731
    dec = encode_decisions(t(d["lon"]), t(d["lon_s"]), t(d["lat"]), t(d["lat_s"]))
    reason = np.stack([df.reason.fillna("").str.contains(r).to_numpy() for r in REASONS], 1)
    return Samples(
        feat_row=t(df.token.map(feats["row"]).to_numpy(), torch.long),
        dec=dec,
        target=t(np.stack(df.target_poses.to_numpy()).reshape(-1, 8, 3), torch.float32),
        flag=t((df.flag == "ACCEPT").to_numpy(), torch.float32),
        reason=t(reason, torch.float32),
        stype=t(df.sample_type.map({s: i for i, s in enumerate(SAMPLE_TYPES)}).to_numpy(), torch.long),
        df=df,
    )


def pose_stats(s: Samples):
    """목표 경로의 (시점, 축)별 평균과 표준편차 (정규화용, 학습 세트에서만 계산한다)."""
    return s.target.mean(0), s.target.std(0).clamp(min=1e-3)
