#!/usr/bin/env bash
# 10단계 시험 (가): CF⁺ 학습 목표 일관화 → B-순응 시드 0 검증 시험 → (통과하면) B-순응, Ours 시드 2개 다시 학습 + 전체 평가.
# 고르는 규칙 (결과를 보기 전에 정함): 검증 CF⁺ 따르기 >= 0.620 + 0.03, 원래 결정 따르기 >= 0.955 - 0.01.
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
L=logs/step10d
for b in val train; do
  echo "=== retarget $b $(date +%H:%M)"
  python scripts/l3_retarget.py --bundle $b 2>&1 | grep --line-buffered -v -E "Warn|Loading" > $L/retarget_$b.log
  tail -3 $L/retarget_$b.log
done
name=step10_rt_Bcomply_seed0
echo "=== $name $(date +%H:%M)"
python scripts/train_planner.py --model Bcomply --steps 20000 --eval_every 10000 --seed 0 --bundle_suffix _retarget --name $name \
  2>&1 | grep --line-buffered -v Warn > $L/train_$name.log
grep -E "eval step" $L/train_$name.log
python scripts/predict_planner.py --run $name --sets val train5k 2>&1 | grep --line-buffered -v Warn
python scripts/follow_eval.py --run $name --sets val train5k 2>&1 | grep --line-buffered -v -E "Warn|Loading"
pass=$(python - <<'PY'
import json
v = json.load(open("exp/eval/step10_rt_Bcomply_seed0/follow_summary.json"))["val"]
cf, org = v["sample_type=cf_pos"]["follow"], v["sample_type=original"]["follow"]
print("yes" if cf >= 0.620 + 0.03 and org >= 0.955 - 0.01 else "no")
PY
)
echo "=== decision: pass=$pass $(date +%H:%M)"
if [ "$pass" != "yes" ]; then echo "=== all done (not adopted) $(date +%H:%M)"; exit 0; fi
for seed in 0 1; do
  for spec in "Bcomply Bcomply_rt" "Ours Ours_w025_rt"; do
    set -- $spec; r=${2}_seed$seed
    extra=""; [ "$1" = "Ours" ] && extra="--neg_to_pos 0.25"
    echo "=== train $r $(date +%H:%M)"
    python scripts/train_planner.py --model $1 --steps 20000 --eval_every 10000 --seed $seed --bundle_suffix _retarget $extra --name $r \
      2>&1 | grep --line-buffered -v Warn > $L/train_$r.log
  done
done
for r in Bcomply_rt_seed0 Ours_w025_rt_seed0 Bcomply_rt_seed1 Ours_w025_rt_seed1; do
  echo "=== eval $r $(date +%H:%M)"
  python scripts/predict_planner.py --run $r 2>&1 | grep --line-buffered -v Warn
  python scripts/follow_eval.py --run $r --sets D0 D1 val 2>&1 | grep --line-buffered -v -E "Warn|Loading"
  python scripts/score_planner.py --run $r --threads 28 > $L/score_$r.log 2>&1
  grep "^$r " $L/score_$r.log
  case $r in Ours*) python scripts/flag_threshold.py --run $r 2>&1 | grep -v Warn > $L/threshold_$r.json;; esac
done
echo "=== all done (adopted) $(date +%H:%M)"
