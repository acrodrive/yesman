"""5단계: L2 경로 모음을 만든다 (navtrain 전체 1,192개 로그의 사람 경로 + 원래 장면에서의 L1 라벨).

입력: exp/l1/navtrain_features.parquet (4단계 l1_extract.py), exp/l1/navtrain_labels.parquet (l1_label.py)
출력: exp/l2/path_bank.parquet (valid 장면만. 열: token, log_name, map_name, v0, poses, seg*_ 라벨,
      pt_s / pt_d / pt_hrel = 원래 차선 기준의 진행 거리, 옆 거리, 차선 대비 방향 (점 9개, 차선 기준으로 옮길 때 씀))

사용법: python scripts/l2_build_bank.py
"""

import os
from pathlib import Path

import pandas as pd

EXP = Path(os.environ["NAVSIM_EXP_ROOT"])


def main():
    f = pd.read_parquet(EXP / "l1/navtrain_features.parquet", columns=["token", "log_name", "map_name", "valid", "v0",
                                                                       "poses", "pt_s", "pt_d", "pt_hrel"])
    lab = pd.read_parquet(EXP / "l1/navtrain_labels.parquet")
    lab = lab.drop(columns=["log_name"]).rename(columns={"valid": "label_valid"})
    bank = f[f.valid == 1].merge(lab, on="token")
    bank = bank[bank.label_valid].drop(columns=["valid", "label_valid"]).reset_index(drop=True)
    out = EXP / "l2/path_bank.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    bank.to_parquet(out, index=False)
    print(f"saved {out}: {len(bank):,} paths from {bank.log_name.nunique():,} logs, "
          f"{bank.map_name.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
