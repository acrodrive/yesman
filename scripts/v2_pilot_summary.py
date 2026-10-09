"""버전 2 시범 결과 요약: 형식 지키기, 조각 수, 좌우 행동 사용, 사람 행동과의 겹침, 걸리는 시간의 다양성, 물리적으로 무리한 정지.

사용법: python scripts/v2_pilot_summary.py exp/v2/pilot3_gemma31b_raw.parquet [...]
"""

import collections
import json
import sys

import pandas as pd

lab = pd.read_parquet("exp/l1/navtest_labels.parquet").drop_duplicates("token").set_index("token")
for f in sys.argv[1:]:
    raw = pd.read_parquet(f)
    n = len(raw)
    ok = sums = nlat = need = hit = impossible = 0
    nch, durs = collections.Counter(), collections.Counter()
    for r in raw.itertuples():
        try:
            plan = json.loads(r.raw_output)["plan"]
        except Exception:  # noqa: BLE001
            continue
        ok += 1
        sums += abs(sum(c["duration_s"] for c in plan) - 4.0) < 1e-6
        nch[len(plan)] += 1
        if len(plan) > 1:
            durs[tuple(c["duration_s"] for c in plan)] += 1
        h = lab.loc[r.token]
        hl = {h.seg1_lat, h.seg2_lat} - {"keep_lane"}
        hk = {x.split("_")[0] for x in hl}  # turn / lane / offset
        vk = {c["lateral"].split("_")[0] for c in plan} - {"keep"}
        nlat += bool(vk)
        if hk:
            need += 1
            hit += bool(hk & vk)
        # 정지 조각이 첫 조각이고 지금 속도로 그 시간 안에 멈출 수 없으면(평균 감속 3 m/s^2 초과) 무리한 정지로 센다
        c0 = plan[0]
        if c0["longitudinal"] == "stop" and r.speed / c0["duration_s"] > 3.0:
            impossible += 1
    print(f"{f}: JSON {ok}/{n}, 합 4초 {sums}/{ok}, 조각 수 {dict(nch)}, 좌우 행동 조각 사용 {nlat}/{ok}, "
          f"사람이 좌우 행동을 한 장면에서 같은 종류를 냄 {hit}/{need}, 무리한 첫 정지 {impossible}, "
          f"여러 조각의 시간 조합 상위 {durs.most_common(4)}")
