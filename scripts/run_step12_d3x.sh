#!/usr/bin/env bash
# 12단계 (2) 이어서: VLM 결정 파싱 + L2 판정 → 평가 세트 D3x_* 고정(체크섬) → L2 최고 경로 채점 → planner 10개 평가.
set -uo pipefail
source /workspace/yesman/scripts/setup/env.sh
cd /workspace/yesman
L=logs/step12
until grep -q "VLM done" $L/vlm.log && grep -q "all done" $L/b3.log; do sleep 30; done
echo "=== judge $(date +%H:%M)"
python scripts/d3_judge.py --name d3x_gemma12b_ext --scenes exp/d3/scenes_ext.parquet 2>&1 | grep -v -E "Warn|Loading" | tail -5
python scripts/d3_judge.py --name d3x_qwen8b --scenes exp/d3/scenes_3k.parquet 2>&1 | grep -v -E "Warn|Loading" | tail -5
python scripts/d3_judge.py --name d3x_gemma31b --scenes exp/d3/scenes_3k.parquet 2>&1 | grep -v -E "Warn|Loading" | tail -5
python - <<'PY'
import pandas as pd, json, shutil
from pathlib import Path
E = Path("exp/d3"); D = Path("data_lists/eval")
old = pd.read_parquet(D / "D3.parquet"); ext = pd.read_parquet(E / "d3x_gemma12b_ext.parquet")
pd.concat([old, ext], ignore_index=True).sort_values("token").reset_index(drop=True).to_parquet(D / "D3x_gemma12b.parquet", index=False)
for n in ("qwen8b", "gemma31b"):
    shutil.copy(E / f"d3x_{n}.parquet", D / f"D3x_{n}.parquet")
meta = {n: json.load(open(E / f"{f}_meta.json")) for n, f in [("gemma12b", "d3x_gemma12b_ext"), ("qwen8b", "d3x_qwen8b"), ("gemma31b", "d3x_gemma31b")]}
for v in meta.values(): v.pop("system_prompt", None); v.pop("schema", None)  # D3_vlm_meta.json과 같다
json.dump(meta, open(D / "D3x_vlm_meta.json", "w"), indent=1, ensure_ascii=False)
for n in ("gemma12b", "qwen8b", "gemma31b"):
    d = pd.read_parquet(D / f"D3x_{n}.parquet"); print(n, len(d), d.status.value_counts().to_dict())
PY
sha256sum data_lists/eval/D3x_gemma12b.parquet data_lists/eval/D3x_qwen8b.parquet data_lists/eval/D3x_gemma31b.parquet data_lists/eval/D3x_vlm_meta.json >> data_lists/eval/SHA256SUMS
for n in gemma12b qwen8b gemma31b; do
  python scripts/score_rows.py --table data_lists/eval/D3x_$n.parquet --col best_poses --out exp/eval/D3x_${n}_best_scores.parquet 2>&1 | grep -v Warn
done
for r in B1_seed0 B1_seed1 B2_seed0 B2_seed1 Bcomply_rt_seed0 Bcomply_rt_seed1 B3_rt_seed0 B3_rt_seed1 Ours_w025_rt_seed0 Ours_w025_rt_seed1; do
  echo "=== planner $r $(date +%H:%M)"
  python scripts/predict_planner.py --run $r --sets D3x_gemma12b D3x_qwen8b D3x_gemma31b 2>&1 | grep --line-buffered -v Warn
  python scripts/follow_eval.py --run $r --sets D3x_gemma12b D3x_qwen8b D3x_gemma31b 2>&1 | grep --line-buffered -v -E "Warn|Loading"
done
echo "=== all done $(date +%H:%M)"
