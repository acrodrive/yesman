"""11단계 RQ5 "규칙 대체" (yesman.md 9.3): 제안 방법이 REJECT를 내면 결정을 "keep lane + stop"으로 바꾸어 같은 모델로
경로를 다시 만든다. 여기서는 모든 행에 대해 그 경로를 만들어 두고, 11단계 분석에서 REJECT인 행에만 쓴다.

규칙 결정 (6단계 stop CF와 같다): [stop s, keep lane] + [stop 0, keep lane], s = 지금 속도 / 2초 / a_max (최대 1).
지금 속도는 LTF 입력 차량 상태(status_feature의 속도 2개)의 크기이다(l3.cf_menu와 같은 ego_velocity).
시작 노이즈는 predict_planner.py와 같다(장면 token마다 고정).

사용법: python scripts/predict_rule.py --run Ours_w025_rt_seed0 [--sets D0 D1 D2 D3]
출력: exp/eval/<run>/<set>_rule.parquet (원래 행 순서, pred_poses = 규칙 결정으로 만든 경로, d_seg* = 규칙 결정)
"""

import argparse
import os
from pathlib import Path

import numpy as np

sys_path = Path(__file__).parent
import sys  # noqa: E402

sys.path.insert(0, str(sys_path))
from predict_planner import load_model, predict, set_table  # noqa: E402

from yesman.l1 import load_thresholds  # noqa: E402
from yesman.train_data import load_features  # noqa: E402

ROOT = Path(os.environ["YESMAN_ROOT"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--sets", nargs="+", default=["D0", "D1", "D2", "D3"])
    args = ap.parse_args()
    model = load_model(args.run)
    feats = load_features("navtest", bev_sem=model.cfg.use_bev_sem)
    status = np.load(ROOT / "exp/features/ltf/navtest/status.npy")
    speed = dict(zip(feats["tokens"], np.linalg.norm(status[:, 4:6], axis=1)))
    a_max = load_thresholds()["a_max"]
    for s in args.sets:
        df = set_table(s).copy()
        v0 = df.token.map(speed).to_numpy()
        df["d_seg1_lon"], df["d_seg2_lon"] = "stop", "stop"
        df["d_seg1_lon_strength"] = np.clip(v0 / 2.0 / a_max, 0.0, 1.0)
        df["d_seg2_lon_strength"] = 0.0
        for k in (1, 2):
            df[f"d_seg{k}_lat"], df[f"d_seg{k}_lat_strength"] = "keep_lane", 0.0
            for c in ("ambiguous_lon", "ambiguous_lat", "structural"):
                df[f"d_seg{k}_{c}"] = False
            df[f"d_seg{k}_alt_lon"], df[f"d_seg{k}_alt_lat"] = "", ""
        res = predict(model, feats, df)
        res.to_parquet(ROOT / f"exp/eval/{args.run}/{s}_rule.parquet", index=False)
        print(f"[{args.run}] {s}_rule: {len(res):,} rows", flush=True)


if __name__ == "__main__":
    main()
