"""사후 분석: 세기 기준 실패가 "앞차가 멈췄는지 달리는지 모름" 때문인가 (2026-10-09, 사용자 질문).

planner 입력은 현재 영상 한 장과 ego 상태뿐이라 앞차의 속도를 알기 어렵다. 앞차 상태별로 결과를 나눠 본다.
앞차 (결과를 보기 전에 정함): 현재 시점에 ego 앞 0~50 m, 옆 |y| < 2 m인 차량 중 가장 가까운 것.
  서 있음 = 속도 < 0.5 m/s, 달림 = 0.5 m/s 이상, 앞차 없음 = 해당 차량 없음. 속도는 정답 어노테이션(분석에만 씀).
대상: D2 세기 기준(주 결과), D1의 같은 결정 종류(faster, slower, stop).
추가 (결과를 본 뒤 더함): 실패 종류(충돌/도로 이탈/역주행)별 비율, 그리고 앞차가 있는 faster에서 D2(불가능)와 D1(가능)을
앞차 거리·ego 속도만으로, 앞차 속도까지 넣어 가를 수 있는 정도(정답 정보로 학습한 분류기, 로그 단위 5겹 교차검증 AUC)와 Ours 판단 head의 AUC.
모델: 최종 모델(B1, B2, B-순응, B3, Ours; 합성 목표), 시드 평균.

사용법: python scripts/analyze_lead_vehicle.py
출력: logs/step12a/lead_vehicle.md
"""

import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from analyze_step11 import tau  # noqa: E402

from navsim.common.dataclasses import SensorConfig  # noqa: E402
from yesman.data import scene_loader, scene_without_sensors  # noqa: E402

ROOT = Path(os.environ["YESMAN_ROOT"])
E = ROOT / "exp/eval"
RUNS = {"B1": "B1", "B2": "B2", "B-순응": "Bcomply_syn", "B3": "B3_syn", "Ours": "Ours_syn"}
SAFE = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance"]


def lead_of_log(args):
    log, tokens = args
    loader = scene_loader("navtest", [log], tokens, sensor_config=SensorConfig.build_no_sensors())
    out = {}
    for t in tokens:
        sc = scene_without_sensors(loader, t)
        fr = sc.frames[sc.scene_metadata.num_history_frames - 1]
        a = fr.annotations
        ego_v = float(np.linalg.norm(fr.ego_status.ego_velocity[:2]))
        best = None
        for b, name, v in zip(a.boxes, a.names, a.velocity_3d):
            if name != "vehicle":
                continue
            x, y = float(b[0]), float(b[1])
            if 0 < x < 50 and abs(y) < 2.0 and (best is None or x < best[0]):
                best = (x, float(np.linalg.norm(v[:2])))
        out[t] = {"ego_v": ego_v, "lead_x": best[0] if best else np.nan, "lead_v": best[1] if best else np.nan}
    return out


def leads(tokens_by_log):
    res = {}
    with Pool(24) as pool:
        for r in pool.imap_unordered(lead_of_log, list(tokens_by_log.items())):
            res.update(r)
    return pd.DataFrame.from_dict(res, orient="index")


def state(row):
    if np.isnan(row.lead_x):
        return "앞차 없음"
    return "서 있음" if row.lead_v < 0.5 else "달림"


def model_cols(df, setname, idx):
    """idx: 세트 안의 행 번호. 모델마다 실행(ACCEPT·결정대로), 거부, 결과 안전(D2만)."""
    out = {}
    for m, r in RUNS.items():
        ex, rej, safe = [], [], []
        for s in (0, 1):
            run = f"{r}_seed{s}"
            f = pd.read_parquet(E / f"{run}/{setname}_l1.parquet").follow.to_numpy()[idx]
            t = tau(run)
            acc = np.ones(len(idx), bool) if t is None else pd.read_parquet(E / f"{run}/{setname}.parquet").p.to_numpy()[idx] >= t
            ex.append(acc & f), rej.append(~acc)
            if setname == "D2":
                sc = pd.read_parquet(E / f"{run}/D2_scores.parquet")
                safe.append((sc[SAFE].min(axis=1) >= 1).to_numpy()[idx])
        out[f"{m}_exec"] = np.mean(ex, 0)
        out[f"{m}_reject"] = np.mean(rej, 0)
        if safe:
            out[f"{m}_safe"] = np.mean(safe, 0)
    return pd.DataFrame(out)


def separability(D2, D1, lead):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict

    a = D2[(D2.visibility != "unseen") & (D2.category == "strength") & (D2.cf_name == "faster")][["token", "log_name"]].assign(y=1, src="D2")
    b = D1[D1.cf_name == "faster"][["token", "log_name"]].assign(y=0, src="D1")
    df = pd.concat([a.assign(i=a.index), b.assign(i=b.index)], ignore_index=True)
    df = pd.concat([df, lead.reindex(df.token).reset_index(drop=True)], axis=1)
    df = df[df.lead_x.notna()].reset_index(drop=True)
    L = ["## 앞차가 있는 faster: 불가능(D2)과 가능(D1)을 가를 수 있는 정도 (AUC)", "",
         f"행: 불가능 {int(df.y.sum())}, 가능 {int((1 - df.y).sum())}", ""]
    for name, cols in (("앞차 거리 + ego 속도 (정답 정보)", ["lead_x", "ego_v"]),
                       ("앞차 거리 + ego 속도 + 앞차 속도 (정답 정보)", ["lead_x", "ego_v", "lead_v"])):
        p = cross_val_predict(HistGradientBoostingClassifier(max_iter=200), df[cols].to_numpy(), df.y,
                              groups=df.log_name, cv=GroupKFold(5), method="predict_proba")[:, 1]
        L.append(f"- {name}: {roc_auc_score(df.y, p):.3f}")
    for s in (0, 1):
        run = f"{RUNS['Ours']}_seed{s}"
        p = np.empty(len(df))
        for src in ("D2", "D1"):
            m = (df.src == src).to_numpy()
            p[m] = pd.read_parquet(E / f"{run}/{src}.parquet").p.to_numpy()[df.i[m].astype(int)]
        L.append(f"- Ours 판단 head ({run}, 1 - p): {roc_auc_score(df.y, 1 - p):.3f}")
    return L + [""]


def main():
    D2 = pd.read_parquet(ROOT / "data_lists/eval/D2.parquet")
    D1 = pd.read_parquet(ROOT / "data_lists/eval/D1.parquet", columns=["token", "log_name", "cf_name"])
    i2 = np.flatnonzero(((D2.visibility != "unseen") & (D2.category == "strength")).to_numpy())
    i1 = np.flatnonzero(D1.cf_name.isin(["faster", "slower", "stop"]).to_numpy())
    toks = pd.concat([D2.iloc[i2][["token", "log_name"]], D1.iloc[i1][["token", "log_name"]]]).drop_duplicates("token")
    lead = leads(toks.groupby("log_name").token.apply(list).to_dict())
    L = ["# 앞차 상태별 결과 (세기 기준과 속도 결정, 시드 평균)", ""]
    # 속도 필드 확인: 앞차가 가까이(< 20 m) 있고 ego가 달릴 때 앞차 속도와 ego 속도의 관계
    near = lead[(lead.lead_x < 20) & (lead.ego_v > 2)]
    L.append(f"속도 필드 확인: 앞차 < 20 m, ego > 2 m/s인 {len(near)}장면에서 ego 속도와 앞차 속도의 상관 "
             f"{np.corrcoef(near.ego_v, near.lead_v)[0, 1]:.2f}, 중앙값 ego {near.ego_v.median():.1f} / 앞차 {near.lead_v.median():.1f} m/s "
             "(비슷하면 절대 속도)\n")
    for name, df, idx, setname in (("D2 세기 기준 (실행할 수 없음)", D2, i2, "D2"),
                                   ("D1 faster·slower·stop (실행 가능)", D1, i1, "D1")):
        sub = df.iloc[idx].reset_index(drop=True)
        sub = pd.concat([sub, lead.reindex(sub.token).reset_index(drop=True), model_cols(df, setname, idx)], axis=1)
        sub["state"] = sub.apply(state, axis=1)
        cols = [c for c in sub.columns if c.endswith(("_exec", "_reject", "_safe"))]
        g = sub.groupby("state")
        tab = g[cols].mean()
        tab.insert(0, "n", g.size())
        tab.insert(1, "앞차 거리 중앙값", g.lead_x.median().round(1))
        tab.insert(2, "ego 속도 중앙값", g.ego_v.median().round(1))
        L += [f"## {name}", "", tab.round(3).to_string(), ""]
        if setname == "D2":
            L += ["실패 종류 (시드 평균, 행 비율)", ""]
            for m in ("Ours", "B1"):
                f = []
                for s in (0, 1):
                    sc = pd.read_parquet(E / f"{RUNS[m]}_seed{s}/D2_scores.parquet").iloc[idx].reset_index(drop=True)
                    f.append(pd.DataFrame({"충돌": sc[SAFE[0]] < 1, "도로 이탈": sc[SAFE[1]] < 1, "역주행": sc[SAFE[2]] < 1}).astype(float))
                L += [m, ((f[0] + f[1]) / 2).groupby(sub.state).mean().round(3).to_string(), ""]
    L += separability(D2, D1, lead)
    (ROOT / "logs/step12a").mkdir(parents=True, exist_ok=True)
    (ROOT / "logs/step12a/lead_vehicle.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
