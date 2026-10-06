# 2단계. 데이터 다운로드 결과 (2026-10-06)

yesman.md 11절 2단계의 결과와 계획서와 달라진 점을 정리한 것이다.

## 끝났다는 기준 확인 (`scripts/check_data.py`, `logs/check_data.log`)

| 기준 | 결과 |
| --- | --- |
| 파일 개수와 크기가 예상과 맞는다 | 통과. 아래 표 |
| 무작위 장면을 불러왔을 때 카메라 이미지, 차량 상태, 미래 경로가 모두 읽힌다 | 통과. navtest와 navtrain에서 각각 20장면씩 확인하였다(3캠 1080×1920×3, 차량 상태 유한값, 미래 경로 8×3) |
| 센서를 받은 navtrain 로그 목록을 저장하였다 | 통과. `data_lists/navtrain_sensor_logs.txt` (304개 로그), `data_lists/navtrain_sensor_tokens.txt` (23,388장면) |

## 받은 데이터 (`/workspace/datasets/navsim`, NAVSIM 공식 구조)

| 항목 | 개수 | 볼륨 용량 (실측) | 계획서 추정 |
| --- | --- | --- | --- |
| 지도 `maps/` | 4개 도시 | 1.4 GB | 5 이하 |
| 로그 `navsim_logs/test` | 147개 로그 | 1.0 GB | 약 15 (로그 합계) |
| 로그 `navsim_logs/trainval` | 1,310개 로그 (navtrain 1,192개를 모두 포함) | 14 GB | |
| navtest 카메라 `sensor_blobs/test` | 12,146장면 × 3캠 = 36,438장, 빠진 것 0 | 8.2 GB | 약 10 |
| navtrain 카메라 `sensor_blobs/trainval` | 8조각, 304개 로그, 23,388장면 × 3캠 = 70,164장, 빠진 것 0 | 16 GB | 약 15 |

- 학습 장면은 navtrain 103,288장면 중 22.6%(23,388장면)이다. 계획서의 "약 25%"에 가깝다.
- 모든 JPEG 106,602장의 시작과 끝 표시(SOI/EOI)를 확인하였다. 잘린 파일은 없다.
- 조각별로 받은 로그는 `_done/navtrain_cur_<i>`에 있다. 1, 5, 9, 13, 17, 21, 25, 29번 조각이고, 조각마다 로그 38개이다.

## 볼륨 사용량 (2단계 끝난 시점)

| 항목 | 용량 |
| --- | --- |
| 데이터 `/workspace/datasets/navsim` | 약 40 GB |
| conda `/workspace/miniforge3` | 21 GB |
| 코드와 체크포인트 `/workspace/yesman` | 0.8 GB |
| 합계 (`du -sh /workspace`) | 62 GB |

7단계 이후에 특징, metric cache, 체크포인트가 약 20 GB 더 생기면 합계는 약 82 GB가 된다. 네트워크 볼륨 100 GB 안에 들어간다.

## 계획서와 달라진 점

- **볼륨에 병렬로 쓰기**: 네트워크 볼륨은 파일마다 지연이 커서, rsync로 하나씩 쓰면 초당 약 20개(약 4.5 MB/s)에 그쳤다. test 조각 하나에 약 10분이 걸렸다. 그래서 32개씩 동시에 복사하도록 바꾸었다(`copy_par`, 초당 약 155개). 그 결과 조각 하나가 1~5분에 끝났다.
- **동시에 여러 갈래로 받기**: test 조각을 두 갈래, navtrain을 한 갈래로 나눠 동시에 받았다. 바꾼 뒤 남은 test 29조각과 navtrain 6조각을 받는 데 약 50분이 걸렸다.
- **CPU pod**: 계획서대로 GPU 없는 RunPod CPU pod(4코어, 메모리 8 GB)에서 받았다. 컨테이너 디스크(50 GB)는 압축을 푸는 임시 공간으로만 썼다.
- **작업 순서**: 계획서의 순서(지도와 로그 → navtest 카메라 → navtrain 카메라)와 달리, navtest와 navtrain을 동시에 받았다. navtrain은 계획서대로 첫 조각을 먼저 받아 폴더 구조(`navtrain_current_<i>/<로그>/CAM_*`)를 확인한 뒤 나머지를 받았다.

## 다음 단계를 위한 참고

- 학습 장면은 `data_lists/navtrain_sensor_tokens.txt`로 제한한다. 경로 모음(5단계)은 센서가 필요 없으므로 navtrain 로그 1,192개 전체로 만든다.
- 3단계(데이터와 채점 도구 확인)는 지도와 로그만 있으면 시작할 수 있다. metric cache 생성에는 CPU와 메모리가 많이 필요할 수 있다. 이 pod는 메모리가 8 GB이므로, 3단계를 시작하면서 필요한 자원을 확인한다.
