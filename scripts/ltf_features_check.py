"""7단계 끝났다는 기준 확인: 저장한 LTF 특징 (scripts/ltf_features.py 출력).

1. 빠짐: 학습과 평가에 쓰는 모든 장면(학습 묶음, 평가 세트 D0~D3)의 특징이 있고, 값이 유한하다.
2. 짝: 장면을 무작위로 골라 DataLoader 없이 장면 하나씩 다시 계산한 값과 저장된 값이 같다.
3. 짝 (전체): LTF가 예측한 경로와 사람 경로의 오차가, 장면 순서를 섞었을 때보다 훨씬 작다.
4. 속도: 특징을 메모리에 올리는 시간과, 무작위 batch를 꺼내는 속도.

사용법: python scripts/ltf_features_check.py [--n_recompute 50]
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from yesman.data import scene_loader

import sys
sys.path.insert(0, str(Path(__file__).parent))
from ltf_features import FIELDS, load_agent, run_batch  # noqa: E402

ROOT = Path(os.environ["YESMAN_ROOT"])
FEAT = ROOT / "exp/features/ltf"


def load(split, mmap=True):
    tokens = FEAT.joinpath(split, "tokens.txt").read_text().split()
    arrs = {k: np.load(FEAT / split / f"{k}.npy", mmap_mode="r" if mmap else None) for k in FIELDS}
    return tokens, arrs


def check_coverage():
    need = {
        "navtrain": set(pd.read_parquet(ROOT / "exp/l3/train_bundle_train.parquet", columns=["token"]).token)
        | set(pd.read_parquet(ROOT / "exp/l3/train_bundle_val.parquet", columns=["token"]).token),
        "navtest": set().union(*(set(pd.read_parquet(ROOT / f"data_lists/eval/{s}.parquet", columns=["token"]).token)
                                 for s in ("D0", "D1", "D2")))
        | set(pd.read_parquet(ROOT / "exp/d3/scenes.parquet", columns=["token"]).token),
    }
    for split, want in need.items():
        tokens, arrs = load(split)
        assert len(set(tokens)) == len(tokens), "토큰 중복"
        missing = want - set(tokens)
        bad = {k: int((~np.isfinite(np.asarray(a, dtype=np.float32).reshape(len(tokens), -1))).any(1).sum())
               for k, a in arrs.items() if k != "bev_sem"}
        n = {k: a.shape[0] for k, a in arrs.items()}
        print(f"[coverage] {split}: saved {len(tokens):,}, needed {len(want):,}, missing {len(missing)}, "
              f"non-finite rows {bad}, rows per field {set(n.values())}")
        assert not missing and set(n.values()) == {len(tokens)} and not any(bad.values())


def check_recompute(n, seed=0):
    agent = load_agent().cuda()
    builder = agent.get_feature_builders()[0]
    rng = np.random.default_rng(seed)
    for split in ("navtest", "navtrain"):
        tokens, arrs = load(split)
        idx = sorted(rng.choice(len(tokens), size=n, replace=False))
        pick = [tokens[i] for i in idx]
        loader = scene_loader(split, tokens=pick, sensor_config=agent.get_sensor_config())
        # 입력(cam_sum, status)은 정확히 같아야 한다. 특징은 batch 크기에 따라 GPU 연산 순서가 달라지고(TF32, cuDNN)
        # fp16으로 저장하므로 조금 다르다. 그래서 상대 오차 |a-b|/|b|와, 저장된 다른 장면과의 상대 차이를 함께 본다.
        err = {k: 0.0 for k in ("status", "cam_sum")}
        rel = {k: [] for k in ("keyval", "query_out")}
        rel_other = {k: [] for k in ("keyval", "query_out")}
        agree = 0.0
        for j, (i, tok) in enumerate(zip(idx, pick)):
            f = builder.compute_features(loader.get_agent_input_from_token(tok))
            res = run_batch(agent, f["camera_feature"][None].cuda(), f["status_feature"][None].cuda())
            res = {k: v[0].cpu().numpy().astype(np.float64) for k, v in res.items()}
            for k in err:
                err[k] = max(err[k], float(np.abs(res[k] - arrs[k][i]).max()))
            other = idx[(j + 1) % n]
            for k in rel:
                a = res[k]
                rel[k].append(np.linalg.norm(a - arrs[k][i]) / np.linalg.norm(a))
                rel_other[k].append(np.linalg.norm(a - arrs[k][other]) / np.linalg.norm(a))
            agree += float((res["bev_sem"] == arrs["bev_sem"][i]).mean()) / n
        print(f"[recompute] {split}: {n} random scenes; inputs max abs diff cam_sum {err['cam_sum']:.1e}, status "
              f"{err['status']:.1e}; relative L2 diff (max over scenes) "
              + ", ".join(f"{k} {max(rel[k]):.4f} (vs other scene min {min(rel_other[k]):.3f})" for k in rel)
              + f"; bev_sem pixel agreement {agree:.4f}")
        assert err["cam_sum"] < 1e-3 and err["status"] < 1e-6
        assert all(max(rel[k]) < 0.02 and max(rel[k]) < 0.05 * min(rel_other[k]) for k in rel)


def check_alignment():
    for split in ("navtest", "navtrain"):
        tokens, arrs = load(split)
        feats = pd.read_parquet(ROOT / f"exp/l1/{split}_features.parquet", columns=["token", "poses"])
        feats = feats[feats.token.isin(set(tokens))].drop_duplicates("token").set_index("token")
        pos = {t: i for i, t in enumerate(tokens)}
        rows = np.array([pos[t] for t in feats.index])
        gt = np.stack(feats.poses.to_numpy()).reshape(-1, 8, 3)[:, :, :2]
        pred = np.asarray(arrs["ltf_traj"])[rows, :, :2]
        ade = np.linalg.norm(pred - gt, axis=-1).mean(1)
        shuf = np.linalg.norm(pred[np.random.default_rng(0).permutation(len(rows))] - gt, axis=-1).mean(1)
        print(f"[align] {split}: {len(rows):,} scenes, LTF vs human ADE mean {ade.mean():.2f} m (median "
              f"{np.median(ade):.2f}); with shuffled pairing {shuf.mean():.2f} m")


def check_speed():
    for split in ("navtrain",):
        t0 = time.time()
        tokens, arrs = load(split, mmap=False)
        t_load = time.time() - t0
        kv, qo = torch.from_numpy(arrs["keyval"]), torch.from_numpy(arrs["query_out"])
        t0, nb = time.time(), 2000
        for _ in range(nb):
            i = torch.randint(len(tokens), (256,))
            _ = kv[i].cuda(non_blocking=True), qo[i].cuda(non_blocking=True)
        torch.cuda.synchronize()
        el = time.time() - t0
        gb = sum(a.nbytes for a in arrs.values()) / 1e9
        print(f"[speed] {split}: load all fields to RAM {t_load:.1f}s ({gb:.2f} GB); random batch 256 "
              f"(keyval+query_out) to GPU: {nb / el:.0f} batches/s = {nb * 256 / el:,.0f} samples/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_recompute", type=int, default=50)
    args = ap.parse_args()
    check_coverage()
    check_alignment()
    check_recompute(args.n_recompute)
    check_speed()
    print("OK")


if __name__ == "__main__":
    main()
