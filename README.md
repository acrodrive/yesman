# YESMAN: Your Ego-planner Shouldn't Mindlessly Accept Nonsense

VLM이 틀린 주행 결정을 내릴 때, 그 결정을 받는 planner는 어떻게 행동하는가를 NAVSIM에서 같은 조건으로 측정한 연구이다. 보고서: [docs/report.md](docs/report.md), 연구 계획서: [yesman.md](yesman.md).

![Figure 1](docs/figs/step14/fig1_follow_safety.png)

*새 테스트 세트에서 결정–경로 일치율(가로)과 불가능한 결정의 안전율(세로). 왼쪽: 규칙으로 만든 실행 불가능한 결정, 오른쪽: 실제 VLM(Gemma 4 31B)이 낸 실행 불가능한 결정. 오른쪽 위가 바람직하다.*

## 주요 결과 (새 테스트 세트, 사전 등록, 시드 2개 평균)

navtest는 13단계까지 결과를 보며 설계를 바꾼 개발 세트이다. 주 결과는 받지 않았던 navtrain 카메라 조각으로 만든 새 테스트 세트에서, 고르는 규칙과 성공 기준을 미리 정하고 한 번만 평가한 값이다.

| 모델 | 결정–경로 일치율 (T1) | 불가능한 결정의 안전율: 규칙 기반 CF⁻ (T2) | 같은 지표: 실제 Gemma 4 31B 결정 (T3, n = 397) | 주행 점수 (T0) |
| --- | --- | --- | --- | --- |
| B0 (결정을 보지 않음) | 3.1% | 96.9% | 97.5% | 0.938 |
| B1 (결정 입력, 사람 결정만 학습) | 34.6% | 16.2% | 25.4% | 0.946 |
| B2 (+ CF⁺) | **80.5%** | 32.3% | 28.5% | 0.939 |
| B3 (+ CF⁻) | 76.5% | 91.8% | 70.9% | 0.939 |
| 이전 Ours (B3 + DiT와 병렬인 판단 head) | 74.1% | 93.3% | 73.4% | 0.941 |
| **Ours** (B3 + DiT 내부 표현에서 판단) | 73.3% | 93.0% | 72.7% | 0.937 |

1. 실행할 수 없는 결정의 예(CF⁻)로 학습하면 처음 보는 VLM의 틀린 결정에서도 크게 안전해진다(Ours − B2 = +44.2%p [+39.4, +48.7]).
2. 학습 장면을 3배로 늘리고 CF를 개량하자, 같은 navtest에서 Gemma 4 31B 오답의 안전율이 30~36%에서 55%로 올랐고 결정을 따르는 능력도 함께 올랐다.
3. Ours는 B3보다 안전에서 낮지 않고(+1.8%p, 성공) 판단과 경로의 불일치를 이전 Ours보다 줄이지만(−1.1%p, 성공), 실행 가능한 결정을 B3보다 덜 따른다(−3.2%p, 허용 폭 −2%p를 넘어 실패).
4. 판단은 규칙 기반 오답의 90%를 거르지만 실제 VLM 오답은 37%만 거른다. 실제 VLM 오답의 절반 이상이 거리와 속도를 판단해야 하는 강도 기준이다.

13단계까지의 결과(순응 학습과 맹목적 추종, 규칙 기반 CF⁻와 실제 VLM 오답의 차이, plausible CF⁻, probe)는 보고서 5절에 있다.

### 모델 이름

| 이 문서와 보고서 | 13단계까지의 이름 | 14단계 코드 실행 이름 | 13단계 코드 실행 이름 |
| --- | --- | --- | --- |
| B0 | B1 | `B1_s14` | `B1` |
| B1 | B2 | `B2_s14` | `B2` |
| B2 | B-순응 | `Bcomply_s14` | `Bcomply_syn` |
| B3 | B3 | `B3_s14_pl04_bs_r05` | `B3_syn` |
| 이전 Ours | Ours | `Ours_s14_pl04_r1` | `Ours_syn` |
| Ours | (없음) | `OursDiT_s14_pl04_r1_jl2` | – |

## 저장소 구성

| 경로 | 내용 |
| --- | --- |
| `yesman/` | 패키지: 결정 형식(`decision.py`), L1 라벨러(`l1.py`), L2 실행 가능 판정(`l2.py`), L3 counterfactual(`l3.py`), target 경로 합성(`synth.py`), planner(`model.py`), 학습 데이터(`train_data.py`), 채점(`scoring.py`) |
| `scripts/` | 단계별 실행 스크립트 (아래) |
| `data_lists/eval_step14/` | 새 테스트 세트 T0~T3 (14단계, 고정, `SHA256SUMS`) |
| `data_lists/eval/` | navtest 평가 세트 D0~D3, D3x(VLM 3개), `SHA256SUMS` (14단계부터 개발 세트) |
| `data_lists/step14/` | 14단계 조각 나누기와 장면 목록 |
| `docs/stepNN_*.md` | 단계별 결과와 계획서와 달라진 점. 14단계는 `step14_preregistration.md`(사전 등록)와 `step14_run.md`(실행 기록) |
| `docs/report.md` | 보고서 |
| `tests/` | L1, 모델 단위 테스트 |

`exp/`(특징, metric cache, 판정 결과, 체크포인트, 평가 결과)와 `third_party/navsim`, `checkpoints/`는 용량 때문에 저장소에 넣지 않았다. 아래 순서로 다시 만든다.

## 재현

평가 세트가 저장소에 있으므로, 평가 세트를 다시 만들지 않고 학습과 평가만 재현할 수도 있다. 확인: `(cd data_lists/eval_step14 && sha256sum -c SHA256SUMS)`, `sha256sum -c data_lists/eval/SHA256SUMS`. 아래 0~4는 13단계까지, 5는 14단계(주 결과)이다.

### 0. 환경과 데이터 (1~3단계)

```bash
scripts/setup/install_navsim.sh                 # conda 환경 navsim, NAVSIM 0a380a9, torch 2.7.1+cu128
source scripts/setup/env.sh                     # 작업할 때마다
# LTF 체크포인트: huggingface.co/autonomousvision/navsim_baselines 의 ltf/ltf_seed_0.ckpt → checkpoints/ltf/
scripts/data/download_navsim.sh maps logs_test logs_trainval navtest_keep \
    "testcam $(seq -s' ' 0 31)" "navtrain 1 5 9 13 17 21 25 29"
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
    train_test_split=navtest metric_cache_path=$NAVSIM_EXP_ROOT/metric_cache/navtest worker.threads_per_node=16
python scripts/run_metric_caching_tokens.py --split navtrain --tokens data_lists/navtrain_sensor_tokens.txt \
    --out exp/metric_cache/navtrain_sensor --workers 16
```

### 1. 라벨과 평가 세트 (4~6단계, CPU)

```bash
python scripts/l1_extract.py --split navtrain --workers 16 && python scripts/l1_label.py --split navtrain
python scripts/l1_extract.py --split navtest --workers 16 && python scripts/l1_label.py --split navtest
python scripts/l2_build_bank.py                                     # 경로 모음
python scripts/l3_run.py --split navtest --name navtest_cf          # CF 판정 (평가 세트용)
python scripts/l3_run.py --split navtrain --cache navtrain_sensor --tokens data_lists/navtrain_sensor_tokens.txt \
    --n_cf 4 --skip_original --name navtrain_sensor_cf               # CF 판정 (학습용)
# 회전 CF 다시 판정 (6단계 수정): --only_turn --merge_into exp/l3/<name>.parquet --name <name>_v2
python scripts/l3_build_sets.py --navtest exp/l3/navtest_cf_v2.parquet --navtrain exp/l3/navtrain_sensor_cf_v2.parquet
python scripts/l3_retarget.py --bundle train && python scripts/l3_retarget.py --bundle val   # CF⁺ 목표 일관화 (10단계)
python scripts/l3_synth.py --bundle train && python scripts/l3_synth.py --bundle val         # CF⁺ 목표 합성 (12단계, 최종)
```

### 2. 특징과 VLM 결정 (7, 12단계, GPU)

```bash
python scripts/ltf_features.py --split navtest && python scripts/ltf_features.py --split navtrain
python scripts/ltf_features_check.py
scripts/setup/install_vlm.sh                    # vLLM 0.31 (torch 2.13) 환경, navsim 환경과 따로
python scripts/d3_prepare.py                    # D3 장면 (D0에서 무작위 1,000개)
VLLM_USE_FLASHINFER_SAMPLER=0 /root/vlm/bin/python scripts/vlm_d3.py --name d3 --model <VLM 경로>
python scripts/d3_judge.py --name d3
# D3 확장(VLM 3개 x 3,000장면): scripts/run_step12_vlm.sh, run_step12_vlm31.sh, run_step12_d3x.sh
```

### 3. 학습 (8~12단계, GPU, 모델 하나에 약 5분. 13단계 이름)

```bash
for s in 0 1; do
  python scripts/train_planner.py --model B1 --steps 20000 --seed $s --name B1_seed$s
  python scripts/train_planner.py --model B2 --steps 20000 --seed $s --name B2_seed$s
  python scripts/train_planner.py --model Bcomply --steps 20000 --seed $s --bundle_suffix _synth --name Bcomply_syn_seed$s
  python scripts/train_planner.py --model B3 --steps 20000 --seed $s --bundle_suffix _synth --neg_to_pos 0.25 --name B3_syn_seed$s
  python scripts/train_planner.py --model Ours --steps 20000 --seed $s --bundle_suffix _synth --neg_to_pos 0.25 --name Ours_syn_seed$s
done
```

### 4. 평가와 그림 (11~13단계)

```bash
r=Ours_syn_seed0                                                    # 모델마다
python scripts/predict_planner.py --run $r                          # D0~D3, 검증 세트 경로
python scripts/follow_eval.py --run $r --sets D0 D1 D2 D3 val       # L1 결정–경로 일치율 판정
python scripts/score_planner.py --run $r                            # D0 공식 EPDMS
python scripts/score_rows.py --table exp/eval/$r/D2.parquet         # D2 불가능한 결정의 안전율
python scripts/flag_threshold.py --run $r                           # Ours: 검증 세트로 기준값
python scripts/predict_rule.py --run $r && python scripts/score_rows.py --table exp/eval/$r/D2_rule.parquet   # 규칙 대체
python scripts/analyze_step11.py --final && python scripts/viz_step11.py --final    # 표, Figure 1~3
python scripts/analyze_step12.py --final && python scripts/analyze_step12_d3safety.py --synth   # 실제 VLM 결정
```

### 5. 14단계: 새 테스트 세트, 데이터 확장, 새 Ours (GPU, 데이터 준비 약 1.5일, 학습 약 18시간, 테스트 평가 약 7시간)

```bash
bash scripts/run_step14_pipeline.sh      # 남은 24조각, 조각 나누기, metric cache, LTF 특징, T0~T3(Gemma 4 31B)와 고정,
                                         # 학습 CF 판정(메뉴 전부, 확장, plausible), 학습 묶음, 그럴싸함 점수
bash scripts/run_step14_train.sh         # step 수·크기, 장면 수 단계, 상한선, B3·Ours의 CF 구성, 이전 Ours, Ours 설정, 시드 1
CONFIRM=1 bash scripts/run_step14_final.sh   # 새 테스트 세트에서 한 번 평가 → logs/step14/final/eval.md
bash scripts/run_step14_navtest.sh && python scripts/step14_navtest_compare.py   # navtest(개발 세트) 비교
python scripts/viz_step14.py             # 보고서 Figure 1, 2
```

컨테이너의 메모리(60 GB)와 CPU(13.6코어) 한도, 볼륨 공간 때문에 바꾼 점은 `docs/step14_run.md`에 있다.

## 단계별 기록

| 단계 | 문서 |
| --- | --- |
| 1~3. 설치, 데이터, 채점 도구 확인 | [step01](docs/step01_setup.md), [step02](docs/step02_data.md), [step03](docs/step03_eval_tools.md) |
| 4. L1 결정 라벨러 | [step04](docs/step04_l1.md) |
| 5. 경로 모음과 L2 | [step05](docs/step05_l2.md) |
| 6. counterfactual과 평가 세트 | [step06](docs/step06_cf.md) |
| 7. 특징과 VLM 결정(D3) | [step07](docs/step07_features_vlm.md) |
| 8. 디코더 | [step08](docs/step08_decoder.md) |
| 9. 비교 모델 | [step09](docs/step09_baselines.md) |
| 10. 제안 방법, flag–경로 불일치 원인 시험 | [step10](docs/step10_ours.md), [step10b](docs/step10b_decoder.md) |
| 11. 평가와 분석 | [step11](docs/step11_eval.md) |
| 12. 추가 실험(B3, VLM 3개, 목표 합성, 확장된 규칙 기반 CF, 물체 정보, plausible CF⁻, probe) | [step12](docs/step12_extra.md) |
| 13. 보고서와 공개 | [step13](docs/step13_release.md) |
| 14. 새 테스트 세트, 데이터 확장, 새 Ours (사전 등록) | [사전 등록](docs/step14_preregistration.md), [실행 기록](docs/step14_run.md) |

## 라이선스와 출처

코드는 이 저장소의 것이다. NAVSIM(Apache-2.0)과 LTF 체크포인트(Apache-2.0), nuPlan/OpenScene 데이터, Gemma 4와 Qwen3-VL 모델은 각자의 라이선스를 따른다.
