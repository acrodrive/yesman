"""14단계 사후 분석 3번 (docs/TODO.md): 강도 기준이 약한 이유. 새 테스트 세트의 결과를 앞차 상태별로 다시 나눈다.

이미 한 번 평가한 결과(exp/eval/<run>_seed<s>/)를 다시 나누기만 한다. 이 결과로 모델이나 τ를 바꾸지 않는다.
앞차는 12단계 scripts/analyze_lead_vehicle.py와 같은 정의: 현재 시점에 ego 앞 0~50 m, 옆 |y| < 2 m인 차량 중 가장 가까운 것.
서 있음 = 속도 < 0.5 m/s. 속도·위치는 정답 어노테이션(분석에만 쓴다).

대상
  T3 강도 기준 (Gemma 4 31B의 실행 불가능한 결정 중 category = strength, 206개)
  T2 강도 기준 (규칙 기반 메뉴 CF⁻, 보이지 않는 원인 제외, 1,533개)
  대조: T3 도로·다른 차 기준, T1의 faster / slower / stop (실행 가능)
모델: B3, 이전 Ours, Ours (시드 0·1 평균). 지표: 안전율(NC·DAC·DDC 모두 통과), 거부율, 실패 종류, T1은 결정–경로 일치율

사용법: python scripts/step14_posthoc_strength.py
출력: docs/step14_posthoc_strength.md, 앞차 표 exp/step14/posthoc/lead_test.parquet (테스트 장면 전부, 다시 쓰지 않음)
"""

import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from step14_posthoc_judge import delta_strength  # noqa: E402

from navsim.common.dataclasses import SensorConfig  # noqa: E402
from yesman.data import scene_loader, scene_without_sensors  # noqa: E402

ROOT = Path(os.environ["YESMAN_ROOT"])
EXP = Path(os.environ["NAVSIM_EXP_ROOT"])
E = EXP / "eval"
D = ROOT / "data_lists/eval_step14"
OUT = ROOT / "docs/step14_posthoc_strength.md"
LEAD = EXP / "step14/posthoc/lead_test.parquet"
MODELS = {"B3": "B3_s14_pl04_bs_r05", "OursPrev": "Ours_s14_pl04_r1", "Ours": "OursDiT_s14_pl04_r1_jl2"}
LABEL = {"B3": "B3", "OursPrev": "이전 Ours", "Ours": "Ours"}
SEEDS = (0, 1)
SAFE = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance"]
FAIL = {"no_at_fault_collisions": "충돌", "drivable_area_compliance": "도로 이탈", "driving_direction_compliance": "역주행"}
WORKERS = 6  # T1 채점(scripts/run_step14_posthoc_t1.sh)과 CPU를 나눠 쓴다


def lead_of_log(args):
    """scripts/analyze_lead_vehicle.py의 lead_of_log와 같고, 테스트 장면이 navtrain 로그에 있어서 split만 다르다."""
    log, tokens = args
    loader = scene_loader("navtrain", [log], tokens, sensor_config=SensorConfig.build_no_sensors())
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


def leads():
    if LEAD.exists():
        return pd.read_parquet(LEAD)
    t0 = pd.read_parquet(D / "T0.parquet", columns=["token", "log_name"])
    res = {}
    with Pool(WORKERS) as pool:
        for r in pool.imap_unordered(lead_of_log, list(t0.groupby("log_name").token.apply(list).items())):
            res.update(r)
    df = pd.DataFrame.from_dict(res, orient="index").rename_axis("token").reset_index()
    LEAD.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LEAD, index=False)
    return df


def add_lead(t, lead):
    t = t.merge(lead, on="token", how="left", validate="m:1")
    t["state"] = np.where(t.lead_x.isna(), "앞차 없음", np.where(t.lead_v < 0.5, "앞차 서 있음", "앞차 달림"))
    t["dist"] = pd.cut(t.lead_x, [0, 10, 20, 50], right=False, labels=["< 10 m", "10~20 m", "20~50 m"]).astype(str) \
        .replace("nan", "앞차 없음")
    t["ego"] = pd.cut(t.ego_v, [0, 1, 5, 10, 99], right=False, labels=["< 1 m/s", "1~5 m/s", "5~10 m/s", "≥ 10 m/s"]) \
        .astype(str)
    return t


def lon_dir(src, t0):
    """앞뒤 강도를 사람 결정보다 세게(↑) / 약하게(↓) 했나: 두 구간 중 앞뒤 행동이 같고 차이가 가장 큰 구간의 부호."""
    h = t0.set_index("token").loc[src.token]
    best, sign = np.zeros(len(src)), np.zeros(len(src))
    for k in (1, 2):
        c = f"d_seg{k}_lon"
        same = src[c].to_numpy() == h[c].to_numpy()
        d = src[f"{c}_strength"].to_numpy(float) - h[f"{c}_strength"].to_numpy(float)
        d = np.where(same & np.isfinite(d), d, 0)
        upd = np.abs(d) > np.abs(best)
        best, sign = np.where(upd, d, best), np.where(upd, np.sign(d), sign)
    lon = src.d_seg1_lon.to_numpy()
    out = np.where(sign > 0, "앞뒤 강도 ↑", np.where(sign < 0, "앞뒤 강도 ↓", "앞뒤 강도 같음"))
    return np.char.add(np.char.add(lon.astype(str), " / "), out.astype(str))


def model_cols(t, setname, keep):
    """keep: 세트 안의 행 마스크. 모델마다 시드 평균 거부, 안전, 실패 종류, 결정–경로 일치."""
    for m, r in MODELS.items():
        acc, safe, fol, fails = [], [], [], {k: [] for k in FAIL}
        for s in SEEDS:
            run = f"{r}_seed{s}"
            f = pd.read_parquet(E / run / f"{setname}_l1.parquet")
            th = E / run / "flag_threshold.json"
            a = (f.p >= json.load(open(th))["tau"]).to_numpy() if "p" in f and th.exists() else np.ones(len(f), bool)
            acc.append(a[keep])
            fol.append((a & f.follow.to_numpy())[keep])
            sc_path = E / run / f"{setname}_scores.parquet"
            if sc_path.exists():
                sc = pd.read_parquet(sc_path)[keep]
                safe.append((sc[SAFE].min(axis=1) >= 1).to_numpy())
                for k in FAIL:
                    fails[k].append((sc[k] < 1).to_numpy())
        t[f"{m}_reject"] = 1 - np.mean(acc, 0)
        t[f"{m}_follow"] = np.mean(fol, 0)
        if safe:
            t[f"{m}_safe"] = np.mean(safe, 0)
            for k, v in FAIL.items():
                t[f"{m}_fail_{v}"] = np.mean(fails[k], 0)
    return t


def pct(x):
    return f"{100 * x:.1f}%" if pd.notna(x) else "—"


def group_table(t, by, cols, order=None):
    g = t.groupby(by)
    keys = [k for k in (order or sorted(g.groups)) if k in g.groups]
    head = [by, "n"] + [c for _, c in cols]
    rows = []
    for k in keys:
        x = g.get_group(k)
        rows.append([k, f"{len(x):,}"] + [pct(x[c].mean()) for c, _ in cols])
    x = t
    rows.append(["전체", f"{len(x):,}"] + [pct(x[c].mean()) for c, _ in cols])
    lines = ["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]
    return "\n".join(lines + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def main():
    lead = leads()
    t0 = pd.read_parquet(D / "T0.parquet")
    t3 = pd.read_parquet(D / "T3_gemma31b.parquet")
    k3 = (t3.status == "infeasible").to_numpy()
    T3 = model_cols(t3[k3].reset_index(drop=True), "T3_gemma31b", k3)
    T3["dir"] = lon_dir(T3, t0)
    T3["dstr"] = delta_strength(T3, t0)
    T3 = add_lead(T3, lead)
    t2 = pd.read_parquet(D / "T2.parquet")
    k2 = ((t2.visibility != "unseen") & (t2.category == "strength")).to_numpy()
    T2 = model_cols(t2[k2].reset_index(drop=True), "T2", k2)
    T2["dir"] = lon_dir(T2, t0)
    T2 = add_lead(T2, lead)
    t1 = pd.read_parquet(D / "T1.parquet", columns=["token", "cf_name"])
    k1 = t1.cf_name.isin(["faster", "slower", "stop"]).to_numpy()
    T1 = add_lead(model_cols(t1[k1].reset_index(drop=True), "T1", k1), lead)

    S3 = T3[T3.category == "strength"]
    safe = [(f"{m}_safe", f"{LABEL[m]} 안전") for m in MODELS]
    rej = [(f"{m}_reject", f"{LABEL[m]} 거부") for m in ("OursPrev", "Ours")]
    fails = [(f"Ours_fail_{v}", f"Ours {v}") for v in FAIL.values()] + [(f"B3_fail_{v}", f"B3 {v}") for v in FAIL.values()]
    states = ["앞차 없음", "앞차 서 있음", "앞차 달림"]
    dists = ["< 10 m", "10~20 m", "20~50 m", "앞차 없음"]
    egos = ["< 1 m/s", "1~5 m/s", "5~10 m/s", "≥ 10 m/s"]
    md = ["# 14단계 사후 분석 3번: 강도 기준이 약한 이유 (앞차 상태별)", "",
          "`scripts/step14_posthoc_strength.py`가 만든 표. 해석은 `docs/step14_run.md`의 \"(사후 분석) 강도 기준\" 절. "
          "새 테스트 세트에서 한 번 평가한 결과를 다시 나눈 것이다. 이 결과로 모델이나 τ를 바꾸지 않는다. 시드 0·1 평균.",
          f"B3 = `{MODELS['B3']}`, 이전 Ours = `{MODELS['OursPrev']}`, Ours = `{MODELS['Ours']}`. "
          "앞차 = ego 앞 0~50 m, 옆 |y| < 2 m인 가장 가까운 차량(정답 어노테이션), 서 있음 = 0.5 m/s 미만.", "",
          "## 불가능 이유별 비교 (T3 실행 불가능 397개)", "",
          group_table(T3, "category", safe + rej, ["road", "agent", "strength"]), "",
          "## 실패 종류 (경로가 그 항목을 통과하지 못한 비율)", "",
          group_table(T3, "category", fails, ["road", "agent", "strength"]), "",
          "## T3 강도 기준: 앞차 상태별", "", group_table(S3, "state", safe + rej, states), "",
          "### 앞차 거리별", "", group_table(S3, "dist", safe + rej, dists), "",
          "### ego 속도별", "", group_table(S3, "ego", safe + rej, egos), "",
          "### 앞뒤 행동 / 강도 방향별 (사람 결정 대비)", "", group_table(S3, "dir", safe + rej), "",
          "### 실패 종류 (앞차 상태별)", "", group_table(S3, "state", fails, states), "",
          "## T2 강도 기준 (규칙 기반 메뉴 CF⁻, 1,533개): 같은 나눔", "",
          group_table(T2, "state", safe + rej, states), "", group_table(T2, "dist", safe + rej, dists), "",
          group_table(T2, "dir", safe + rej), "",
          "## 대조: T1 faster / slower / stop (실행 가능): 결정–경로 일치율과 거부율", "",
          group_table(T1, "state", [(f"{m}_follow", f"{LABEL[m]} 일치") for m in MODELS] + rej, states), "",
          group_table(T1, "cf_name", [(f"{m}_follow", f"{LABEL[m]} 일치") for m in MODELS] + rej), "",
          "## 결정 구성 비교 (T2 강도 기준 vs T3 강도 기준)", ""]
    comp = pd.concat([T2.assign(set="T2"), S3.assign(set="T3")])
    for col, order in (("state", states), ("dist", dists), ("ego", egos)):
        ct = pd.crosstab(comp[col], comp.set, normalize="columns").reindex(order)
        md += [f"| {col} | T2 | T3 |", "| --- | --- | --- |"] + [f"| {k} | {pct(r.T2)} | {pct(r.T3)} |"
                                                                    for k, r in ct.iterrows()] + [""]
    md.append(f"강도 차이(사람 결정과) 중앙값: T3 강도 기준 {S3.dstr.median():.2f}")
    md += ["", "## 조건 구성으로 설명되는 몫",
           "", "T2 강도 기준의 조건별 안전율이 그대로이고 조건 구성만 T3 강도 기준처럼 바뀌었다면의 안전율.", "",
           "| 조건 | " + " | ".join(f"{LABEL[m]}: T2 → 다시 무게 → T3" for m in MODELS) + " |",
           "|" + " --- |" * (1 + len(MODELS))]
    for name, cols in (("앞차 거리", ["dist"]), ("앞차 상태", ["state"]), ("ego 속도", ["ego"]),
                       ("앞차 거리 x ego 속도", ["dist", "ego"])):
        w = S3.groupby(cols).size() / len(S3)
        row = [name]
        for m in MODELS:
            r2 = T2.groupby(cols)[f"{m}_safe"].mean().reindex(w.index).fillna(T2[f"{m}_safe"].mean())
            row.append(f"{pct(T2[f'{m}_safe'].mean())} → {pct((w * r2).sum())} → {pct(S3[f'{m}_safe'].mean())}")
        md.append("| " + " | ".join(row) + " |")
    OUT.write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
