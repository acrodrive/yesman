"""14단계: 새로 받은 navtrain 카메라 조각을 테스트 / 예비 / 학습 추가로 나누고 장면 목록을 만든다.

조각 나누기는 docs/step14_preregistration.md 2.1절에 고정한다. 기존 목록(data_lists/navtrain_sensor_*.txt)은 바꾸지 않는다
(scripts/check_data.py를 다시 돌리면 받은 조각 전부로 그 목록을 덮어쓰므로 14단계에서는 돌리지 않는다).

입력: OPENSCENE_DATA_ROOT/_done/navtrain_cur_<i> (조각 i에 든 로그 이름, scripts/data/download_navsim.sh가 남김)
출력 (data_lists/step14/):
  {test,reserve,train_extra}_logs.txt, {test,reserve,train_extra}_tokens.txt   용도별 로그와 장면(현재 프레임 3캠이 모두 있는 navtrain 장면)
  train_extra_shards.tsv     학습 추가 장면의 조각 번호 (token, shard). 장면 수를 단계적으로 늘릴 때 쓴다
  train_all_tokens.txt       학습 + 검증 장면 = 기존 센서 장면 + 학습 추가 장면 (학습 CF를 만들 장면, plausible CF의 닮은 장면 후보)
  camera_all_{tokens,logs}.txt  카메라가 있는 모든 장면 (LTF 특징을 계산할 장면)
  val_logs.txt               검증 로그 = 기존 검증 로그 + (--extra_val_frac > 0이면) 학습 추가 로그의 일부
  check.md                   겹침 확인: 용도 사이 로그 겹침(있으면 중단), 같은 날짜의 로그 수

사용법 (source scripts/setup/env.sh 후): python scripts/step14_shards.py [--extra_val_frac 0]
"""

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from check_data import DATA_ROOT, load_filter, scan  # noqa: E402

ROOT = Path(os.environ["YESMAN_ROOT"])
OUT = ROOT / "data_lists/step14"
OLD = (1, 5, 9, 13, 17, 21, 25, 29)  # 2단계에서 받은 조각
# docs/step14_preregistration.md 2.1절. 조각 번호는 1~32 (scripts/data/download_navsim.sh)
GROUPS = {"test": (3, 11, 19, 27), "reserve": (7, 15, 23, 31),
          "train_extra": (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32)}


def shard_logs(i: int):
    f = DATA_ROOT / f"_done/navtrain_cur_{i}"
    if not f.exists():
        raise SystemExit(f"조각 {i}를 아직 받지 않았다 ({f} 없음)")
    return f.read_text().split()


def log_date(log: str) -> str:
    return ".".join(log.split(".")[:3])  # nuPlan 로그 이름은 YYYY.MM.DD.로 시작한다


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra_val_frac", type=float, default=0.0,
                    help="학습 추가 로그 중 검증으로 뗄 비율 (0이면 검증 세트는 기존 30개 로그 그대로)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    all_shards = sorted(OLD + sum(GROUPS.values(), ()))
    assert all_shards == list(range(1, 33)), "조각 나누기가 1~32를 한 번씩 덮어야 한다"
    OUT.mkdir(parents=True, exist_ok=True)

    navtrain = load_filter("navtrain")
    nav_logs = set(navtrain.log_names)
    old_logs = set((ROOT / "data_lists/navtrain_sensor_logs.txt").read_text().split())
    old_tokens = (ROOT / "data_lists/navtrain_sensor_tokens.txt").read_text().split()
    logs, tokens, shard_of = {}, {}, {}
    for g, shards in GROUPS.items():
        logs[g], ok, n_missing = [], [], 0
        for i in shards:
            ls = shard_logs(i)
            assert set(ls) <= nav_logs, f"조각 {i}: navtrain이 아닌 로그가 섞였다"
            ok_i, missing_i = scan("trainval", navtrain, ls)
            logs[g] += ls
            ok += ok_i
            n_missing += len(missing_i)
            shard_of.update({t: i for t in ok_i})
        logs[g], tokens[g] = sorted(set(logs[g])), sorted(set(ok))
        print(f"{g}: 조각 {shards}, 로그 {len(logs[g])}, 장면 {len(tokens[g]):,} (3캠이 없는 장면 {n_missing:,})", flush=True)

    # 겹침 확인: 용도 사이에 같은 로그가 있으면 안 된다
    report = ["# 14단계 조각 나누기 확인\n"]
    named = {"old(학습+검증)": old_logs, **{g: set(v) for g, v in logs.items()}}
    names = list(named)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            n = len(named[names[a]] & named[names[b]])
            report.append(f"- 로그 겹침 {names[a]} ∩ {names[b]}: {n}")
            assert n == 0, f"로그가 겹친다: {names[a]}, {names[b]}"
    train_dates = Counter(log_date(lg) for lg in old_logs | set(logs["train_extra"]))
    for g in ("test", "reserve"):
        same = sum(log_date(lg) in train_dates for lg in logs[g])
        report.append(f"- {g} 로그 중 학습 로그와 같은 날짜에 기록된 로그: {same} / {len(logs[g])}")

    for g in GROUPS:
        (OUT / f"{g}_logs.txt").write_text("\n".join(logs[g]) + "\n")
        (OUT / f"{g}_tokens.txt").write_text("\n".join(tokens[g]) + "\n")
    (OUT / "train_extra_shards.tsv").write_text("token\tshard\n" + "".join(f"{t}\t{shard_of[t]}\n" for t in tokens["train_extra"]))
    train_all = sorted(set(old_tokens) | set(tokens["train_extra"]))
    (OUT / "train_all_tokens.txt").write_text("\n".join(train_all) + "\n")
    cam_tokens = sorted(set(train_all) | set(tokens["test"]) | set(tokens["reserve"]))
    cam_logs = sorted(old_logs | set(sum(logs.values(), [])))
    (OUT / "camera_all_tokens.txt").write_text("\n".join(cam_tokens) + "\n")
    (OUT / "camera_all_logs.txt").write_text("\n".join(cam_logs) + "\n")

    val = (ROOT / "data_lists/navtrain_sensor_val_logs.txt").read_text().split()
    if args.extra_val_frac > 0:
        rng = np.random.default_rng(args.seed)
        ex = logs["train_extra"]
        val += sorted(rng.choice(ex, size=int(round(len(ex) * args.extra_val_frac)), replace=False).tolist())
    (OUT / "val_logs.txt").write_text("\n".join(sorted(val)) + "\n")
    report.append(f"- 학습 + 검증 장면 {len(train_all):,} (기존 {len(old_tokens):,} + 추가 {len(tokens['train_extra']):,}), "
                  f"검증 로그 {len(val)}개 (extra_val_frac = {args.extra_val_frac})")
    report.append(f"- 카메라가 있는 장면 전부 {len(cam_tokens):,}, 로그 {len(cam_logs)}")
    (OUT / "check.md").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
