# 14단계 새 테스트 세트 T0~T3

- 만든 날: 2026-10-15, 코드 커밋: d5dbf36
- 장면: navtrain 카메라 조각 중 테스트 조각(data_lists/step14/test_tokens.txt). 학습에 쓰지 않은 로그
- 상태: **고정**. 이후 바꾸지 않는다. 확인: `cd data_lists/eval_step14 && sha256sum -c SHA256SUMS`
- 한 번만 평가한다 (docs/step14_preregistration.md 8절)
- 행 수: {'T0': 18219, 'T1': 75767, 'T2': 18485, 'T3_gemma31b': 18219}
- 정의: T0~T2는 navtest D0~D2와 같은 규칙(scripts/l3_build_sets.py), T3_*는 D3와 같은 규칙(scripts/d3_judge.py)
