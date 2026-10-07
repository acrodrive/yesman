# 평가 세트 D0~D2 (navtest)

- 만든 날: 2026-10-07, 코드 커밋: 19b20e6, 입력: navtest_cf_v2.parquet
- 상태: **고정** (2026-10-07, 사용자 확인). 이후 바꾸지 않는다. 확인: `sha256sum -c data_lists/eval/SHA256SUMS`
- 검증 세트 로그 분할(`data_lists/navtrain_sensor_val_logs.txt`)도 함께 고정한다
- D0 11,751개 (L2 가능 11,500), D1 48,577개, D2 14,498개 (주 결과 13,755)
- 열 설명과 만드는 방법: docs/step06_cf.md, scripts/l3_build_sets.py
