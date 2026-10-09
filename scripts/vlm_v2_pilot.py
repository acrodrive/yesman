"""버전 2 시범: VLM이 4초 동안의 maneuver를 길이가 정해지지 않은 조각들(행동, 단계, 걸리는 시간)로 내게 한다.

버전 1(scripts/vlm_d3.py)과 같은 이미지, 같은 차량 상태와 내비게이션 명령, 같은 디코딩(greedy, JSON 강제)을 쓰고
결정 형식만 바꾼다. 조각마다: 앞뒤 행동(go/stop) + 속도 단계(weak/medium/strong) + 좌우 행동 + 좌우 단계
(weak/medium/strong, keep_lane이면 none) + 걸리는 시간(0.5초 단위). 걸리는 시간의 합은 4초여야 한다(스키마로는
강제하지 못하므로 출력 뒤에 확인한다).

사용법 (VLM 환경): VLLM_USE_FLASHINFER_SAMPLER=0 /root/vlm/bin/python scripts/vlm_v2_pilot.py --limit 20 --model /root/hf/...
출력: exp/v2/pilot_raw.parquet, exp/v2/pilot_meta.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

sys.path.insert(0, str(Path(__file__).parent))
from vlm_d3 import COMMAND_TEXT, USER, load_images  # noqa: E402

ROOT = Path(os.environ.get("YESMAN_ROOT", "/workspace/yesman"))
LAT = ["keep_lane", "turn_left", "turn_right", "lane_change_left", "lane_change_right", "offset_left", "offset_right"]
DUR = [0.5 * k for k in range(1, 9)]
CHUNK = {
    "type": "object",
    "properties": {
        "longitudinal": {"enum": ["go", "stop"]},
        "speed_level": {"enum": ["weak", "medium", "strong"]},
        "lateral": {"enum": LAT},
        "lateral_level": {"enum": ["none", "weak", "medium", "strong"]},
        "duration_s": {"enum": DUR},
    },
    "required": ["longitudinal", "speed_level", "lateral", "lateral_level", "duration_s"],
    "additionalProperties": False,
}
SCHEMA = {
    "type": "object",
    "properties": {
        "scene_description": {"type": "string"},
        "key_issue": {"type": "string"},
        "plan": {"type": "array", "items": CHUNK, "minItems": 1, "maxItems": 8},
    },
    "required": ["scene_description", "key_issue", "plan"],
    "additionalProperties": False,
}

SYSTEM = """You are the driving-decision module of an autonomous car. You look at the car's three front cameras and decide the maneuver for the next 4 seconds. A separate motion planner will turn your plan into a trajectory.

Work in three stages:
1. scene_description: describe everything around the ego car that matters for driving (road layout, lanes, intersection, traffic lights and signs, vehicles, pedestrians, cyclists, obstacles), in 2-5 sentences.
2. key_issue: the single most important factor for the ego car's next action, in one sentence.
3. plan: the maneuver for the next 4 seconds as a sequence of chunks, in time order. Use as few chunks as needed: a car cruising in its lane needs one chunk of 4.0 s; an overtake might be keep_lane 0.5 s, then lane_change_left 2.0 s, then keep_lane 1.5 s. The durations must add up to exactly 4.0 s (multiples of 0.5 s).

Each chunk has:
- longitudinal: go (moving) or stop (coming to a stop or staying stopped by the end of the chunk).
- speed_level: for go, the speed during the chunk: weak = crawling or slow (below about 3 m/s), medium = moderate (about 3-7 m/s), strong = fast (above about 7 m/s). For stop, how hard the car brakes: weak = gentle or already standing still, medium = normal braking, strong = hard braking.
- lateral: keep_lane (follow the current lane, including curves), turn_left / turn_right (turn through an intersection or junction into another road), lane_change_left / lane_change_right (move into the adjacent lane), offset_left / offset_right (shift sideways but stay in the same lane, e.g. to give room to a parked car or cyclist).
- lateral_level: none for keep_lane. Otherwise how sharp or large the lateral motion is: weak = gentle or small, medium = normal, strong = sharp or large.
- duration_s: how long the chunk lasts.

Answer only with the JSON object."""

# 시범 2 (2026-10-08): 시범 1에서 좌우 행동 조각을 한 번도 쓰지 않아 예시와 지침을 더함
SYSTEM_B = SYSTEM.replace(
    "Use as few chunks as needed: a car cruising in its lane needs one chunk of 4.0 s; an overtake might be keep_lane 0.5 s, then lane_change_left 2.0 s, then keep_lane 1.5 s.",
    "Use as many chunks as the maneuver needs, in time order. Examples: cruising in the lane = keep_lane 4.0 s; "
    "approaching an intersection and turning left = keep_lane 1.5 s, then turn_left 2.5 s; passing a parked car = "
    "offset_left 1.5 s, then keep_lane 2.5 s; overtaking = keep_lane 0.5 s, then lane_change_left 2.0 s, then keep_lane 1.5 s.")
SYSTEM_B = SYSTEM_B.replace("Answer only with the JSON object.",
    "Follow the navigation command: if it says turn left or right and the ego car is at or near the intersection, "
    "include the turn chunk at the time the car would actually turn. Use lane_change or offset chunks when the "
    "situation needs them (for example to reach a turning lane or to pass an obstacle). Use keep_lane only when "
    "the car should simply follow its lane.\n\nAnswer only with the JSON object.")


# 시범 3 (2026-10-08): 시범 2에서 걸리는 시간이 예시의 숫자를 따라 굳었으므로, 숫자 없는 예시로 바꾼다
SYSTEM_C = SYSTEM.replace(
    "Use as few chunks as needed: a car cruising in its lane needs one chunk of 4.0 s; an overtake might be keep_lane 0.5 s, then lane_change_left 2.0 s, then keep_lane 1.5 s.",
    "Use as many chunks as the maneuver needs, in time order, and choose each duration from what you see "
    "(distances, speeds, where the intersection or the gap is). For example: cruising = one keep_lane chunk; "
    "turning at an intersection = keep_lane until the car reaches it, then the turn; passing a parked car = an offset, "
    "then keep_lane; overtaking = keep_lane, lane_change, keep_lane.")
SYSTEM_C = SYSTEM_C.replace("Answer only with the JSON object.",
    "Follow the navigation command: if it says turn left or right and the ego car is at or near the intersection, "
    "include the turn at the time the car would actually turn. Use lane_change or offset chunks when the situation "
    "needs them. Every chunk must be physically possible from the current speed (for example, a car at 9 m/s cannot "
    "stop within 1 s).\n\nAnswer only with the JSON object.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=Path, default=ROOT / "exp/d3/scenes.parquet")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", default="pilot")
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--prompt", choices=["a", "b", "c"], default="a", help="a: 시범 1, b: 시범 2 (예시와 지침 추가), c: 시범 3 (숫자 없는 예시)")
    args = ap.parse_args()
    scenes = pd.read_parquet(args.scenes).head(args.limit)
    system = {"a": SYSTEM, "b": SYSTEM_B, "c": SYSTEM_C}[args.prompt]
    gemma = "gemma" in args.model.lower()
    mm = {"max_soft_tokens": 560} if gemma else {"max_pixels": 560 * 32 * 32}
    llm = LLM(args.model, max_model_len=args.max_model_len, gpu_memory_utilization=0.92,
              limit_mm_per_prompt={"image": 3}, mm_processor_kwargs=mm, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=1024, seed=0, structured_outputs=StructuredOutputsParams(json=SCHEMA))
    msgs, texts = [], []
    for r in scenes.itertuples():
        text = USER.format(speed=r.speed, ax=r.ax, command_text=COMMAND_TEXT[r.command])
        msgs.append([{"role": "system", "content": system},
                     {"role": "user", "content": [*({"type": "image_pil", "image_pil": im} for im in load_images(r)),
                                                  {"type": "text", "text": text}]}])
        texts.append(text)
    t0 = time.time()
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False} if gemma else None, use_tqdm=False)
    rows = []
    for r, text, o in zip(scenes.itertuples(), texts, outs):
        c = o.outputs[0]
        rows.append(dict(token=r.token, speed=r.speed, command=r.command, raw_output=c.text,
                         finish_reason=c.finish_reason, n_output_tokens=len(c.token_ids)))
    out = ROOT / f"exp/v2/{args.name}_raw.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    json.dump(dict(model=args.model, mm=mm, prompt=args.prompt, system_prompt=system, schema=SCHEMA), open(ROOT / f"exp/v2/{args.name}_meta.json", "w"),
              indent=1, ensure_ascii=False)
    print(f"saved {out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
