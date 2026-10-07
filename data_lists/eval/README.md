# 평가 세트 D0~D2 (navtest)

- 만든 날: 2026-10-07, 코드 커밋: 19b20e6, 입력: navtest_cf_v2.parquet
- 상태: **고정** (2026-10-07, 사용자 확인). 이후 바꾸지 않는다. 확인: `sha256sum -c data_lists/eval/SHA256SUMS`
- 검증 세트 로그 분할(`data_lists/navtrain_sensor_val_logs.txt`)도 함께 고정한다
- D0 11,751개 (L2 가능 11,500), D1 48,577개, D2 14,498개 (주 결과 13,755)
- 열 설명과 만드는 방법: docs/step06_cf.md, scripts/l3_build_sets.py

# 평가 세트 D3 (VLM 결정, 7단계)

- 만든 날: 2026-10-07. 상태: **고정** (7단계 계획서 "L2로 판정하여 D3로 고정"). 체크섬은 위 SHA256SUMS에 함께 있다
- D3.parquet: navtest 1,000장면(D0에서 무작위, 시드 0). VLM 원문 출력(`raw_output`), 장면 설명, 핵심 이슈, 파싱한 결정(`d_seg*`),
  사람 결정과 같은지(`same_as_human`), L2 판정(`status`, `reason`, `category`, `visibility`, `best_score`, `best_poses`)
- D3_vlm_meta.json: 모델, 시스템 프롬프트, JSON 스키마, 디코딩 설정
- 파싱 실패 0개, L2 가능 728 / 불가능 31 / 판정 불가 241
- 열 설명과 만드는 방법: docs/step07_features_vlm.md, scripts/d3_prepare.py, scripts/vlm_d3.py, scripts/d3_judge.py
