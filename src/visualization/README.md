# E1 결과 시각화

프로젝트 루트에서 실행합니다.

```bash
.venv/bin/python src/visualization/plot_e1.py
```

`plot_e1.py` 상단의 `INCLUDE_FP32 = True`가 기본값입니다.
FP16 결과에 없는 샘플만 FP32 재추론 결과로 보완합니다.
그림에 혼합 정밀도임을 표시하고 각 정밀도로 계산된 샘플 수를 기록합니다.
`False`로 바꾸면 FP16 결과만 사용합니다.
추가 명령행 인자는 없습니다.

출력은 `src/visualization/outputs/` 아래에 PNG, PDF, 지표 JSON으로 저장합니다.
기본 파일 이름은 `e1_main_fp16_fp32`, FP16만 사용할 때는 `e1_main_fp16`입니다.

| 패널 | 계산 |
|---|---|
| (a) Yes / No Accuracy | 정답 Yes, No 조건 각각에서 올바른 응답 수 / 유효 추론 수 |
| (b) False-Yes Rate | 정답 No 조건에서 Yes 응답 수 / 유효 추론 수 |
| (c) Counterexample Sensitivity | 그룹별 `ΔN = s(A_N) - s(B_N)`의 평균 |
| (d) Class-wise False-Yes | 최대 N의 No 조건에서 클래스별 false-Yes율 |
| (e) Logit Margin Distribution | N별 Yes/No 조건의 `s(X)` 분포; 상자는 사분위 범위, 중앙선은 중앙값 |
| (f) Group-level Sensitivity Change | 그룹별 `Δ8 - Δ2`; 음수이면 반례 민감도 감소 |

`s(X) = yes_logit - no_logit`이며, A/B는 한 장만 다른 minimal pair입니다.
(c), (f)는 모든 N에서 A/B 결과가 모두 있는 동일한 그룹만 사용합니다.
그래프의 그룹 수는 실제 분석 가능한 수를 표시하며 100으로 고정하지 않습니다.

비율의 구간은 95% Wilson 구간입니다. 민감도 평균의 구간은 그룹 단위
bootstrap 2,000회로 계산한 점별 95% 구간입니다. N 사이의 상관을 유지하도록
매 반복에서 같은 그룹들을 모든 N에 사용합니다. 시드는 42입니다.
(e)의 whisker는 사분위 범위의 1.5배 이내 관측값이며, 밖의 값은 점으로 표시합니다.

미완료·비정상 logit은 분석에서 제외하고 누락 수와 ID를 지표 JSON에 남깁니다.
Yes/No로 해석할 수 없는 응답은 정확도에서 오답으로 계산하지만,
명시적 false-Yes/false-No로 분류하지 않습니다. 해당 건수도 지표 JSON에 저장합니다.
(d)의 평균선은 클래스별 분모를 반영한 전체 No 샘플의 비율입니다.
원본 추론의 저장된 `correct`를 그대로 평균하지 않고 정답과 예측을 다시 비교합니다.
