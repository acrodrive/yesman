"""7단계: 공개 VLM(Gemma 4 12B instruction-tuned)으로 navtest 장면의 주행 결정을 만든다 (D3, yesman.md 8.3 M0).

navsim 환경이 아니라 VLM 전용 환경에서 돌린다 (vLLM이 torch 2.13을 쓰므로). 설치: scripts/setup/install_vlm.sh
    /root/vlm/bin/python scripts/vlm_d3.py --limit 20 --name pilot     # 시범 실행 (장면 20개)
    /root/vlm/bin/python scripts/vlm_d3.py --name d3                   # 전체 (exp/d3/scenes.parquet)

- 입력: planner(LTF)와 같은 세 카메라 이미지. LTF가 쓰는 범위와 같게 자른다(L0, R0는 가운데 1088열, 모두 위아래 28행).
  각 이미지를 따로 넣고 프롬프트에 어느 카메라인지 적는다. 차량 상태(속도, 가속도)와 내비게이션 명령도 글로 준다.
  planner가 받는 정보(status_feature)와 같다.
- 출력: 세 단계(장면 설명, 핵심 이슈, 주행 결정)를 JSON 하나로 강제한다(vLLM structured outputs). greedy 디코딩.
- 저장: exp/d3/<name>_raw.parquet (token, prompt, 원문 출력, 토큰 수, finish_reason). 파싱과 L2 판정은 scripts/d3_judge.py.
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from PIL import Image
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

ROOT = Path(os.environ.get("YESMAN_ROOT", "/workspace/yesman"))
MODEL = os.environ.get("VLM_MODEL", "/root/hf/gemma-4-12B-it")  # google/gemma-4-12B-it (HF에서 받은 것)

LON = ["go", "stop"]
LAT = ["keep_lane", "turn_left", "turn_right", "lane_change_left", "lane_change_right", "offset_left", "offset_right"]
SEGMENT = {
    "type": "object",
    "properties": {
        "longitudinal": {"enum": LON},
        "longitudinal_strength": {"type": "number", "minimum": 0, "maximum": 1},
        "lateral": {"enum": LAT},
        "lateral_strength": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["longitudinal", "longitudinal_strength", "lateral", "lateral_strength"],
    "additionalProperties": False,
}
SCHEMA = {
    "type": "object",
    "properties": {
        "scene_description": {"type": "string"},
        "key_issue": {"type": "string"},
        "decision": {
            "type": "object",
            "properties": {"segment_1": SEGMENT, "segment_2": SEGMENT},
            "required": ["segment_1", "segment_2"],
            "additionalProperties": False,
        },
    },
    "required": ["scene_description", "key_issue", "decision"],
    "additionalProperties": False,
}

# 세기의 뜻은 L1(yesman/l1.py, l1_thresholds.yaml)의 정의와 같다. 6.1의 형식.
SYSTEM = """You are the driving-decision module of an autonomous car. You look at the car's three front cameras and decide the maneuver for the next 4 seconds. A separate motion planner will turn your decision into a trajectory.

Work in three stages:
1. scene_description: describe everything around the ego car that matters for driving (road layout, lanes, intersection, traffic lights and signs, vehicles, pedestrians, cyclists, obstacles), in 2-5 sentences.
2. key_issue: the single most important factor for the ego car's next action, in one sentence (e.g. "The pedestrian ahead is close, so the car must stop quickly.").
3. decision: the maneuver for two time segments, segment_1 = 0-2 s from now and segment_2 = 2-4 s from now.

Each segment has a longitudinal action, a lateral action, and a strength (0 to 1) for each:

Longitudinal (judged at the end of the segment):
- go: the car is still moving at the end of the segment (speed >= 0.5 m/s). strength = speed at the end of the segment / 10.5 m/s (0.2 = 2.1 m/s, 0.5 = 5.3 m/s, 1.0 = 10.5 m/s or faster).
- stop: the car is (almost) stopped at the end of the segment (speed < 0.5 m/s). strength = average deceleration during the segment / 1.4 m/s^2 (0.0 = already standing still, 0.3 = gentle stop, 1.0 = hard stop of 1.4 m/s^2 or more).

Lateral (judged relative to the lane the car is in at the start of the segment):
- keep_lane: follow the current lane, including where the lane curves. strength = 0.
- turn_left / turn_right: turn through an intersection or junction into another road. Use it for every segment in which the heading changes by 15 degrees or more during the turn. strength = heading change within the segment / 45 degrees.
- lane_change_left / lane_change_right: at the end of the segment the car is in the adjacent lane. strength = peak sideways speed / 2.0 m/s. A single lane change appears as lane_change in only one segment.
- offset_left / offset_right: shift sideways by 0.5 m or more but stay in the same lane (e.g. to give room to a parked car or cyclist). strength = sideways shift / 1.7 m.

Answer only with the JSON object."""

USER = """Images: (1) front-left camera CAM_L0, (2) front camera CAM_F0, (3) front-right camera CAM_R0.
Ego state now: speed {speed:.1f} m/s, longitudinal acceleration {ax:+.1f} m/s^2.
Navigation command from the route planner: {command_text}.
Give your three-stage answer."""

COMMAND_TEXT = {
    "left": "turn left at the next intersection (it may still be some distance ahead)",
    "right": "turn right at the next intersection (it may still be some distance ahead)",
    "straight": "go straight / follow the road",
    "unknown": "none",
}


def load_images(r):
    """LTF의 TransfuserFeatureBuilder._get_camera_feature와 같은 범위로 자른다 (이어 붙이지는 않는다)."""
    l0 = Image.open(r.cam_l0).convert("RGB")
    f0 = Image.open(r.cam_f0).convert("RGB")
    r0 = Image.open(r.cam_r0).convert("RGB")
    w, h = f0.size  # 1920 x 1080
    return [l0.crop((416, 28, w - 416, h - 28)), f0.crop((0, 28, w, h - 28)), r0.crop((416, 28, w - 416, h - 28))]


def build_messages(r):
    text = USER.format(speed=r.speed, ax=r.ax, command_text=COMMAND_TEXT[r.command])
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [*({"type": "image_pil", "image_pil": im} for im in load_images(r)),
                                     {"type": "text", "text": text}]},
    ], text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=Path, default=ROOT / "exp/d3/scenes.parquet")
    ap.add_argument("--name", default="d3")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--soft_tokens", type=int, default=560, help="이미지 하나당 시각 토큰 수 (70/140/280/560/1120)")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--max_model_len", type=int, default=8192, help="31B는 GPU 메모리 때문에 4096 (프롬프트 약 2,300 + 출력 1,024 이하)")
    ap.add_argument("--model", default=MODEL, help="VLM 경로 (12단계: Gemma 4 31B, Qwen3-VL 8B)")
    args = ap.parse_args()

    scenes = pd.read_parquet(args.scenes)
    if args.limit:
        scenes = scenes.head(args.limit)
    gemma = "gemma" in args.model.lower()
    # 이미지 하나당 시각 토큰 수를 VLM끼리 맞춘다: Gemma는 max_soft_tokens, Qwen3-VL은 토큰 하나 = 32x32 픽셀
    mm = {"max_soft_tokens": args.soft_tokens} if gemma else {"max_pixels": args.soft_tokens * 32 * 32}
    llm = LLM(args.model, max_model_len=args.max_model_len, gpu_memory_utilization=0.92, limit_mm_per_prompt={"image": 3},
              mm_processor_kwargs=mm, seed=0)
    chat_kw = {"enable_thinking": False} if gemma else None
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=0,
                        structured_outputs=StructuredOutputsParams(json=SCHEMA))
    rows, t0 = [], time.time()
    for lo in range(0, len(scenes), args.chunk):
        part = scenes.iloc[lo: lo + args.chunk]
        msgs, texts = zip(*(build_messages(r) for r in part.itertuples()))
        outs = llm.chat(list(msgs), sp, chat_template_kwargs=chat_kw, use_tqdm=False)
        for r, text, o in zip(part.itertuples(), texts, outs):
            c = o.outputs[0]
            rows.append(dict(token=r.token, user_prompt=text, raw_output=c.text, finish_reason=c.finish_reason,
                             n_prompt_tokens=len(o.prompt_token_ids), n_output_tokens=len(c.token_ids)))
        print(f"{len(rows)}/{len(scenes)} scenes, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    meta = dict(model=args.model, mm_processor_kwargs=mm, soft_tokens=args.soft_tokens, temperature=0.0, max_tokens=args.max_tokens,
                system_prompt=SYSTEM, schema=SCHEMA)
    out = ROOT / f"exp/d3/{args.name}_raw.parquet"
    df.to_parquet(out, index=False)
    (ROOT / f"exp/d3/{args.name}_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(f"saved {out} ({time.time() - t0:.0f}s); finish_reason {df.finish_reason.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
