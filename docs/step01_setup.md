# 1단계. 사전 확인 결과 (2026-10-06)

yesman.md 11절 1단계의 결과를 정리한 것이다.

## 끝났다는 기준 확인

| 기준 | 결과 |
| --- | --- |
| NAVSIM 예제 코드가 오류 없이 실행된다 | 통과. `scripts/check_navsim_example.py` (공식 튜토리얼의 BEV, 등속 agent, 카메라 그림을 test 데이터와 전방 3캠으로 실행) |
| LTF 체크포인트를 불러올 수 있고, 2단계 첫 조각의 장면 하나에서 특징이 나온다 | 통과. 현재 main 코드의 `TransfuserAgent.initialize()`로 그대로 불러온다. v1.1 브랜치는 필요 없다. `scripts/check_ltf.py` |
| 받을 조각과 예상 용량을 확정하였다 | 확정. 아래 표 |

## 설치 환경

- 경로 규칙: 코드와 가중치는 `/workspace/yesman`, 데이터는 `/workspace/datasets/navsim` (NAVSIM 공식 구조), conda는 `/workspace/miniforge3`이다.
- 작업을 시작할 때마다 `source /workspace/yesman/scripts/setup/env.sh`를 실행한다. 이 파일이 환경변수(`OPENSCENE_DATA_ROOT` 등)를 정하고 `navsim` 환경을 켠다.
- 처음부터 다시 설치하려면 `scripts/setup/install_navsim.sh`를 실행한다.
- NAVSIM: main 브랜치의 커밋 `0a380a9` (2025-10-27)이다. `third_party/navsim`에 두고 git에서는 제외하였다.
- **torch를 바꿈**: NAVSIM은 `torch==2.0.1`로 고정되어 있지만, 이 버전은 RTX 5090(sm_120)에서 돌아가지 않는다. 그래서 `torch 2.7.1 + torchvision 0.22.1 (CUDA 12.8)`로 바꾸었다. 나머지 패키지는 NAVSIM의 requirements를 그대로 쓴다(`pip check`에서 걸리는 것은 torch와 torchvision 두 개뿐이다). CPU pod에서 확인했으므로 GPU에서 실제로 동작하는지는 7단계 전에 GPU pod에서 다시 확인한다.

## LTF 확인 결과 (`logs/check_ltf.log`)

- 체크포인트: `checkpoints/ltf/ltf_seed_0.ckpt` (673,235,500 bytes)
- 파라미터 669개가 모델과 정확히 맞는다(누락 0, 잉여 0). `latent=True`용 `lidar_latent` 파라미터도 있다.
- 장면 특징 (장면 하나 기준)

| 이름 | 모양 | 설명 |
| --- | --- | --- |
| camera_feature | 3 × 256 × 1024 | L0·F0·R0를 이어 붙인 입력 이미지 |
| status_feature | 8 | 내비게이션 명령(4) + 속도(2) + 가속도(2) |
| bev | 512 × 8 × 8 | 백본의 BEV 특징 (디코더에는 256 × 64 토큰으로 들어간다) |
| bev_upscale | 64 × 64 × 64 | BEV 지도 분할 head의 입력 |
| query_out | 31 × 256 | 트랜스포머 디코더 출력 (경로 쿼리 1개 + 물체 쿼리 30개) |

- navtest 20장면에서 LTF가 예측한 경로와 사람 경로의 L2 오차는 평균 0.59 m, 4초 끝점 1.50 m이다. 학습된 가중치가 제대로 들어갔다고 판단한다.
- 7단계에서 어떤 특징을 저장할지(bev 토큰, query_out 등)는 8단계의 디코더 설계와 함께 정한다.

## 받을 조각 (확정)

- 지도: nuPlan maps v1.1 전체
- 로그: trainval, test 전체
- navtest: test 카메라 아카이브 32조각(0~31) 전체. navtest 장면의 현재 프레임 3캠(36,438장 = 12,146장면 × 3)만 남긴다.
- navtrain: `navtrain_current` 32조각(번호는 **1~32**) 중 1, 5, 9, 13, 17, 21, 25, 29의 8조각. 받는 양은 68.1 GB이다(9.47 + 5.83 + 7.14 + 13.06 + 7.96 + 9.97 + 4.50 + 10.21).

## 예상 용량 (실측 반영)

| 항목 | 볼륨에 남는 양 (GB) | 근거 |
| --- | --- | --- |
| 로그 (trainval, test) | 약 15.5 | test는 실측 1.0 (압축 0.48). trainval은 압축 7.05에 같은 비율을 적용한 추정 |
| 지도 | 1.4 | 실측 |
| navtest 카메라 | 약 9.2 | 첫 조각 실측: 242장면 726장이 183 MB → 36,438장으로 환산 |
| navtrain 카메라 | 약 15 (미확정) | 계획서의 추정을 그대로 쓴다. 첫 조각을 받은 뒤 실측해서 고친다 |
| conda 환경과 코드 | 약 22 | 실측 (miniforge3 21 + yesman 0.8). 계획서의 10보다 크다. site-packages가 15.5 GB이다(torch와 CUDA 라이브러리 6.3, ray 0.9 등) |
| 특징, metric cache, 체크포인트 | 약 20 | 계획서의 추정 |
| 합계 | 약 83 | 네트워크 볼륨 100 GB 안에 들어간다 |

## 2단계를 위해 확인한 점

- 다운로드 스크립트: `scripts/data/download_navsim.sh`. 계획서의 스크립트를 이 저장소의 경로에 맞추고 아래 문제를 고쳤다.
  - nuPlan 지도 S3는 연결 하나당 약 40 KB/s로 느리다(970 MB에 약 7시간). 그래서 64개 구간으로 나눠 병렬로 받고, 크기와 `unzip -t`로 검증한다(약 1분).
  - 로그 아카이브는 `meta_datas/<split>/*.pkl` 구조이다. 그래서 `navsim_logs/<split>/*.pkl`이 되도록 한 단계 아래 폴더를 옮긴다.
  - 압축을 푸는 임시 폴더는 컨테이너 디스크(`/tmp`)에 둔다. 남길 파일만 볼륨으로 옮기므로 볼륨에 따로 여유 공간이 필요 없다. 조각마다 `_done/` 표시를 하므로 pod가 꺼져도 이어서 받을 수 있다.
  - navtrain 조각마다 `_done/navtrain_cur_<i>`에 그 조각에 들어 있는 로그 목록을 저장한다. 이것이 "센서를 받은 navtrain 로그 목록"이 된다.
- HuggingFace에서 받는 속도는 약 20 MB/s이다. 남은 다운로드(약 200 GB)는 계산상 3시간 정도 걸린다.
- 이미 받은 것: 지도, test 로그, navtest 이미지 목록(`navtest_keep.txt`), test 카메라 0번 조각(`_done/test_cam_0`).
- 남은 것: trainval 로그, test 카메라 1~31번 조각, navtrain 8조각. 아래 명령으로 받는다.

```bash
source /workspace/yesman/scripts/setup/env.sh
scripts/data/download_navsim.sh logs_trainval "testcam $(seq -s' ' 1 31)" "navtrain 1 5 9 13 17 21 25 29"
```
