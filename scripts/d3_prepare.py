"""7단계: D3(VLM 결정)에 쓸 navtest 장면을 고르고, VLM 입력(이미지 경로, 차량 상태, 내비게이션 명령)을 저장한다.

장면: D0(구조적 애매함이 없는 navtest 장면 11,751개)에서 무작위로 n개 (시드 0). 무작위로 고르는 이유는
RQ2가 "실제 VLM 결정의 오류 분포"를 재기 때문이다(장면 종류를 일부러 맞추면 분포가 바뀐다).

사용법: python scripts/d3_prepare.py [--n 1000]
출력: exp/d3/scenes.parquet (token, log_name, cam_l0/f0/r0 경로, speed, accel, command)
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from yesman.data import DATA_ROOT, scene_loader

ROOT = Path(os.environ["YESMAN_ROOT"])
COMMANDS = ("left", "straight", "right", "unknown")  # driving_command one-hot 순서 (navtest 회전 라벨로 확인)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", type=Path, default=None, help="이 표의 장면은 빼고 고른다 (12단계 D3 확장)")
    ap.add_argument("--out", type=Path, default=ROOT / "exp/d3/scenes.parquet")
    ap.add_argument("--split", default="navtest", help="navtrain: 학습 묶음의 모든 장면 (12단계 A, 실제 VLM 결정으로 CF 만들기)")
    args = ap.parse_args()

    if args.split == "navtrain":
        pick = pd.concat([pd.read_parquet(ROOT / f"exp/l3/train_bundle_{b}.parquet", columns=["token", "log_name", "sample_type"])
                          for b in ("train", "val")])
        pick = pick[pick.sample_type == "original"].drop_duplicates("token").sort_values("token")
    else:
        d0 = pd.read_parquet(ROOT / "data_lists/eval/D0.parquet")
        if args.exclude:
            d0 = d0[~d0.token.isin(set(pd.read_parquet(args.exclude, columns=["token"]).token))]
        pick = d0.sample(n=args.n, random_state=args.seed).sort_values("token")
    loader = scene_loader(args.split, log_names=pick.log_name.unique(), tokens=pick.token)
    rows = []
    for r in pick.itertuples():
        frames = loader.scene_frames_dicts[r.token]
        cur = frames[loader._scene_filter.num_history_frames - 1]
        assert cur["token"] == r.token
        # 차량 상태는 NAVSIM AgentInput과 같은 ego 좌표계 값 (LTF status_feature와 같다)
        ego = loader.get_agent_input_from_token(r.token).ego_statuses[-1]
        rows.append(dict(
            token=r.token, log_name=r.log_name,
            **{f"cam_{c}": str(DATA_ROOT / "sensor_blobs" / ("trainval" if args.split == "navtrain" else "test")
                               / cur["cams"][f"CAM_{c.upper()}"]["data_path"])
               for c in ("l0", "f0", "r0")},
            speed=float(np.linalg.norm(ego.ego_velocity)), vx=float(ego.ego_velocity[0]),
            ax=float(ego.ego_acceleration[0]), command=COMMANDS[int(np.argmax(ego.driving_command))]))
    df = pd.DataFrame(rows)
    assert all(Path(p).exists() for c in ("l0", "f0", "r0") for p in df[f"cam_{c}"])
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"{len(df)} scenes in {df.log_name.nunique()} logs -> {out}")
    print(df.command.value_counts().to_string())
    print(df.speed.describe().round(2).to_string())


if __name__ == "__main__":
    main()
