"""공식 채점 CSV(run_score_navtest.sh의 결과) 여러 개를 받아 항목별 평균을 표로 출력한다.

사용법: python scripts/summarize_scores.py 이름=CSV경로 [이름=CSV경로 ...]
"""

import sys

import pandas as pd

COLS = [("score", "EPDMS"), ("no_at_fault_collisions", "NC"), ("drivable_area_compliance", "DAC"),
        ("driving_direction_compliance", "DDC"), ("traffic_light_compliance", "TLC"), ("ego_progress", "EP"),
        ("time_to_collision_within_bound", "TTC"), ("lane_keeping", "LK"), ("history_comfort", "HC"),
        ("two_frame_extended_comfort", "EC")]


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[~df.token.str.startswith("average")].set_index("token")  # 마지막 평균 줄은 뺀다


def main():
    runs = {k: load(v) for k, v in (a.split("=", 1) for a in sys.argv[1:])}
    print("| 경로 | 장면 수 | 실패 | " + " | ".join(n for _, n in COLS) + " |")
    print("| --- " * (len(COLS) + 3) + "|")
    for name, df in runs.items():
        vals = " | ".join(f"{df[c].mean():.3f}" for c, _ in COLS)
        print(f"| {name} | {len(df)} | {(~df.valid.astype(bool)).sum()} | {vals} |")
    names = list(runs)
    if len(names) == 2:
        a, b = runs[names[0]], runs[names[1]]
        common = a.index.intersection(b.index)
        d = a.loc[common, "score"] - b.loc[common, "score"]
        print(f"\n장면별 {names[0]} - {names[1]} (EPDMS): 높음 {(d > 0).mean() * 100:.1f}%, "
              f"같음 {(d == 0).mean() * 100:.1f}%, 낮음 {(d < 0).mean() * 100:.1f}%")
        print(f"EC가 계산된 장면 비율: {names[0]} {a['two_frame_extended_comfort'].notna().mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
