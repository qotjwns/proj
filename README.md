# 이미지 3장 Yes/No 실험

모델·데이터 다운로드 → pool과 케이스 준비 → A/B/C 추론 → 결과 평가·시각화 순서로 사용합니다.
명령은 프로젝트 루트에서 실행합니다. 현재 모델 로더는 로컬 **Qwen2.5-VL** 계열용입니다.

## 코드 구조

```text
src/
├── download/
│   ├── download_model.py       # Hugging Face 모델 다운로드
│   └── download_data.py        # COCO val2017 / COCO-ReM 다운로드
├── data_manipulate/
│   ├── make_pool.py            # 클래스별 양성 / 음성 후보
│   ├── make_inference_pool.py  # 클래스별 5개 그룹
│   ├── build_cases.py          # 그룹 → 800개 케이스
│   └── convert_results.py      # 구형 결과 → 공통 결과 형식
├── experiments/
│   ├── a/run.py               # 3장 동시 입력 → 최종 Yes/No
│   ├── b/run.py               # 1장씩 판단 → Python AND
│   └── c/run.py               # 3장 동시 입력 → 라벨 3개 → Python AND
├── common/
│   ├── paths.py               # 공통 경로
│   ├── data.py                # JSON 입출력 / 필터
│   ├── images.py              # EXIF 회전 / RGB 로딩
│   ├── model.py               # 모델 로딩 / 동일한 전처리 / 생성
│   ├── results.py             # 공통 결과 필드 / 답변 파싱 / AND
│   └── experiment.py          # 추론 인자 / 케이스 선택 / 결과 저장
└── visualize/
    ├── evaluate.py            # 실험별 지표 / NNN·PNN·PPN·PPP별 오류
    ├── compare.py             # 같은 케이스의 A/B/C 비교
    ├── plot_errors.py         # 오답 케이스 이미지
    ├── plot_inference_cases.py  # 입력 3장씩 그룹별 PNG
    ├── plot_inference_groups.py # 그룹의 양성 4장 / 음성 8장
    ├── plot_pool_samples.py   # 원본 후보 pool 확인
    ├── metrics.py             # 지표 계산
    └── reporting.py           # 표 PNG / CSV 저장
```

`experiments`는 케이스별 JSONL만 저장합니다. 지표 집계와 PNG·CSV 생성은 모두 `visualize`에서 수행합니다.
공통 모듈은 직접 실행하지 않습니다. 각 파일 첫 주석에 해당 모듈을 사용하는 실행 명령을 적었습니다.

## 1. 모델·데이터 준비

```bash
pip install -r requirements.txt
python src/download/download_model.py --model-name Qwen/Qwen2.5-VL-3B-Instruct
python src/download/download_data.py
```

모델은 `data/models/Qwen/Qwen2.5-VL-3B-Instruct`, 이미지는 `data/coco/val2017`,
주석은 `data/coco_rem/instances_valrem.json`에 저장합니다.
7B를 사용하려면 다운로드의 `--model-name`과 추론·평가의 `--model-path`를 함께 바꿉니다.

## 2. Pool·케이스 준비

```bash
python src/data_manipulate/make_pool.py
python src/data_manipulate/make_inference_pool.py
python src/data_manipulate/build_cases.py
```

이미 케이스가 있다면 이 단계는 건너뛰고 **A/B/C 모두 기존 JSONL을 그대로 사용합니다.**

- 양성은 대상 객체의 bbox 하나가 이미지 면적의 5% 이상인 이미지입니다.
- 음성 후보는 해당 객체의 주석이 없는 이미지입니다. 사람이 검수한 음성이라는 의미는 아닙니다.
- 기본 그룹은 20클래스 × 5그룹 = 100그룹입니다. 그룹당 양성 4장과 음성 후보 8장을 참조하며 이미지는 복사하지 않습니다.
- `build_cases.py`는 이 100그룹을 사용해 그룹당 Yes 4개·No 4개를 만듭니다.

| 이미지 구성 조건 | 구성 | 정답 | 전체 개수 |
| --- | --- | --- | ---: |
| A | NNN | No | 134 |
| B | PNN | No | 133 |
| C | PPN | No | 133 |
| D | PPP | Yes | 400 |

이미지 구성 조건 A~D와 **실험 A/B/C는 별개**입니다.
케이스 파일은 `data/inference_cases/inference_groups_seed42_balanced3_seed42_cases.jsonl`입니다.
기존 케이스 필드·ID·이미지 순서는 유지하며, 새 결과의 `image_order`도 그대로 따릅니다.
파일명 앞에는 각 행의 `image_dir`를 붙이면 입력 이미지를 찾을 수 있습니다.

`build_cases.py`는 기존 파일을 덮어쓰지 않습니다. 다른 케이스가 필요하면 상단의 `OUTPUT_JSONL`을 바꿉니다.
복원된 생성기는 기존과 같은 필드·ID 계산 규칙·개수를 사용하지만, 과거 음성 조합까지 동일하다고 보장하지 않습니다.
기존 실험 비교를 위해 케이스를 다시 생성하지 마세요.

## 3. A/B/C 추론

각 실험에서 우선 앞의 20개로 형식을 확인할 수 있습니다.

```bash
python src/experiments/a/run.py --limit 20
python src/experiments/b/run.py --limit 20
python src/experiments/c/run.py --limit 20
```

전체 실행은 `--limit`를 생략합니다. 이전 C 코드의 `--mode full`은 사용하지 않습니다.

```bash
python src/experiments/a/run.py --model-path data/models/Qwen/Qwen2.5-VL-3B-Instruct
python src/experiments/b/run.py --model-path data/models/Qwen/Qwen2.5-VL-3B-Instruct
python src/experiments/c/run.py --model-path data/models/Qwen/Qwen2.5-VL-3B-Instruct
```

모든 실험은 같은 케이스 파일과 이미지 로딩·전처리 코드를 사용합니다.
생성은 greedy이며 기본 최대 생성 토큰은 A/B 8개, C 32개입니다.
B는 A 결과 없이 실행할 수 있고, 같은 이미지·질문의 반복 추론은 한 실행 안에서 재사용합니다.
기존 B 예측을 재사용하려면 `--reuse-jsonl <B 결과>`를 지정합니다.
같은 모델 경로·생성 토큰 설정·이미지·질문의 결과만 재사용합니다.

| 실험 | 모델 입력 / 응답 | 최종 판정 |
| --- | --- | --- |
| A | 이미지 3장 / Yes 또는 No 하나 | 생성 답변 |
| B | 이미지 1장씩 3회 / 각 Yes 또는 No | Python AND |
| C | 이미지 3장 / `Yes, No, Yes` 형태 | Python AND |

A는 케이스의 `Is there a {object} in all 3 images? Please answer with Yes or No.`를 사용합니다.
B는 `in all 3 images`를 `in this image`로 바꿉니다. C는 다음 프롬프트를 사용합니다.

```text
For each of the 3 images, determine whether there is at least one {object} in that image. Answer with exactly three labels (Yes or No) in image order, separated by commas. Do not provide explanations or a final combined answer.
```

공통 선택 인자는 `--model-path`, `--cases-jsonl`, `--output`, `--limit`, `--group`, `--category`,
`--condition A|B|C|D`, `--device auto|cuda|mps|cpu`, `--max-new-tokens`입니다.
다른 실행 조건의 결과를 보관하려면 `--output`으로 별도 파일을 지정합니다. 같은 출력 경로로 재실행하면 이전 추론 결과를 교체합니다.

기본 저장 경로:

```text
data/experiments/<a|b|c>/results/<모델 폴더명>/full/results.jsonl
data/experiments/<a|b|c>/results/<모델 폴더명>/pilot_20/results.jsonl
```

그룹·클래스·조건을 제한하면 `full` 대신 `filtered_<선택값>` 폴더를 사용합니다.

## 공통 결과 필드

A/B/C 모두 **케이스당 한 행**이며 같은 필드 이름과 자료형을 사용합니다.

| 필드 | 의미 |
| --- | --- |
| `schema_version`, `experiment` | 결과 형식 버전 `1`, 실험 `a`/`b`/`c` |
| `case_id`, `group_id`, `class` | 원본 케이스 연결 정보 |
| `model_path`, `max_new_tokens` | 모델과 생성 설정 |
| `image_dir`, `image_order` | 이미지 폴더와 입력 파일명 3개 |
| `gt`, `image_gt` | 최종 정답, 이미지별 Yes/No 정답 3개 |
| `prompt` | 실제 사용한 질문 |
| `model_answers` | 원문 답변 배열: A 1개 / B 3개 / C 1개 |
| `image_predictions` | B/C의 이미지별 예측 3개; A와 파싱 실패한 C는 `null` |
| `prediction` | 최종 Yes/No; 파싱 실패 시 `null` |
| `correct` | 최종 정답 여부; 파싱 실패는 `false` |
| `parse_valid` | 최종 판정에 필요한 답변 형식이 유효한지 |
| `yes_logits`, `no_logits` | 각 모델 응답의 첫 생성 위치에서 얻은 Yes/No 원시 logit 배열 |

A/B의 logit 배열 길이는 `model_answers`와 같습니다. C는 세 라벨의 위치별 점수를 추출하지 않으므로
`yes_logits`와 `no_logits`가 각각 `[null]`입니다. 이 값을 세 이미지의 confidence로 해석하지 않습니다.
logit은 확률이 아닙니다. A/B에서 Yes/No 두 후보의 조건부 확률은 해당 쌍에 softmax를 적용해 후처리할 수 있습니다.

C 파서는 정확히 세 개의 대문자 시작 Yes/No와 쉼표를 요구하며 공백은 허용합니다.
설명·마지막 마침표·추가 라벨이 있으면 원문은 보존하고 실패로 기록합니다.
B에서 한 답변만 파싱되지 않아도 해당 위치는 `null`, 최종 `prediction`도 `null`입니다.

## 4. 평가·시각화

```bash
python src/visualize/evaluate.py --experiment a
python src/visualize/evaluate.py --experiment b
python src/visualize/evaluate.py --experiment c
python src/visualize/compare.py
```

7B 등 다른 모델은 각 명령에 같은 `--model-path`를 붙입니다. 평가는 모델을 로드하지 않습니다.
20개 pilot 평가는 `--run pilot_20`, 별도 결과는 `--input-jsonl <파일>`을 사용합니다.

실험별 결과는 `data/experiments/<실험>/evaluation/<모델>/<실행명>/`에 저장합니다.

- `metrics.json`, `metrics.png`: Accuracy, Precision/Recall/F1, False-Yes/False-No, 형식 실패, 이미지별 정확도, 세 장 모두 정답 비율.
- `composition.json`, `composition.csv`, `composition.png`: NNN/PNN/PPN/PPP별 개수와 오류율.
- 비교 결과는 `data/experiments/comparison/<모델>/full/`의 `comparison.json`, `comparison.csv`, `comparison.png`, `cases.csv`입니다.

이미지별 정확도는 케이스 내 이미지 출현 횟수를 분모로 씁니다. 같은 이미지가 여러 케이스에 나오면 각각 셉니다.
형식이 틀린 케이스는 최종 Accuracy에서 실패로 셉니다.
C에서 세 라벨을 파싱하지 못하면 이미지 세 장도 모두 실패로 셉니다.
B는 답변이 이미지별로 분리되어 있으므로 파싱에 실패한 위치만 이미지 오답으로 셉니다.
False-Yes/False-No에는 유효한 Yes/No 오답만 포함하고 형식 실패는 따로 셉니다.
Recall의 분모에는 형식 실패한 GT Yes 케이스도 포함합니다. A의 이미지별 지표는 `null`/N/A입니다.

전체 비교는 원본 케이스가 모두 있는지, 이미지 순서와 정답이 같은지 확인합니다.
일부 결과의 교집합만 골라 전체 성능으로 표시하지 않습니다. 실험마다 질문이 달라 결과만으로 원인을 단정할 수 없습니다.

```bash
# 오답 이미지 / PPN만
python src/visualize/plot_errors.py --experiment c
python src/visualize/plot_errors.py --experiment a --condition C --error-type false-yes

# 입력 이미지: 그룹당 PNG 하나, 케이스당 이미지 3장
python src/visualize/plot_inference_cases.py
python src/visualize/plot_inference_cases.py --category cat --condition D

# 후보 pool과 그룹 구성 확인
python src/visualize/plot_pool_samples.py
python src/visualize/plot_inference_groups.py
```

입력 케이스 플롯은 기존처럼 `data/inference_group_plots/`에 저장합니다.
오답 플롯은 해당 실험의 evaluation 폴더 안에 저장합니다.

## 기존 결과 사용

기존 케이스·이미지·결과는 자동 변환하거나 다시 생성하지 않습니다.
구형 결과는 아래처럼 **별도 파일**로 변환한 후 평가합니다.

```bash
python src/data_manipulate/convert_results.py --experiment a --input-jsonl data/experiments/a/results/inference_groups_seed42_balanced3_seed42_cases_results.jsonl
python src/visualize/evaluate.py --experiment a --input-jsonl data/experiments/a/results/inference_groups_seed42_balanced3_seed42_cases_results_v1.jsonl
```

기본 변환 출력은 입력 파일명 뒤에 `_v1`을 붙이며 기존 파일을 덮어쓰지 않습니다.
현재 남아 있는 구형 A 결과는 601행이므로 단독 평가는 가능하지만 800개 전체 비교에는 추가 추론이 필요합니다.
B 변환에는 이미지별 원문이 담긴 `single_image_results.jsonl`을 사용합니다. AND 집계 파일만으로는 원문 세 개를 복원할 수 없습니다.
과거 최대 생성 토큰 설정을 알면 변환 시 `--max-new-tokens`로 지정합니다. 모르면 `null`로 두며 B 캐시 재사용 대상이 되지 않습니다.

## 로직 확인

```bash
python -m unittest discover -s tests -p 'test_results.py'
```

이 테스트는 모델 없이 공통 필드, C 파서, AND와 이미지 정확도의 차이, 형식 실패 집계, 케이스 정렬을 확인합니다.

## Vast.ai 새 환경 실행 순서

저장소를 받은 뒤 프로젝트 루트에서 실행합니다. 아래는 기본 3B 모델 기준입니다.
**기존 800개와 비교하려면 로컬의 `data/inference_cases/`를 서버의 같은 위치에 복사하고 3번을 건너뜁니다.**

```bash
# 1. Python 환경 준비 (이미 설치된 PyTorch도 사용)
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
# CUDA: True인지 확인한 뒤 진행

# 2. 모델·데이터 다운로드
python src/download/download_model.py
python src/download/download_data.py

# 3. 새 케이스를 만들 때만: pool → 그룹 → 800개 케이스
python src/data_manipulate/make_pool.py
python src/data_manipulate/make_inference_pool.py
python src/data_manipulate/build_cases.py

# 4. 각 실험 20개 확인
python src/experiments/a/run.py --device cuda --limit 20
python src/experiments/b/run.py --device cuda --limit 20
python src/experiments/c/run.py --device cuda --limit 20
python src/visualize/evaluate.py --experiment c --run pilot_20

# 5. 20개 결과의 출력 형식을 확인한 뒤 전체 실행
python src/experiments/a/run.py --device cuda
python src/experiments/b/run.py --device cuda
python src/experiments/c/run.py --device cuda

# 6. 실험별 평가·시각화 → A/B/C 비교
python src/visualize/evaluate.py --experiment a
python src/visualize/evaluate.py --experiment b
python src/visualize/evaluate.py --experiment c
python src/visualize/compare.py

# 7. 이미지 확인이 필요하면
python src/visualize/plot_inference_cases.py
python src/visualize/plot_errors.py --experiment c
```
