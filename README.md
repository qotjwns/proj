# 그룹별 이미지 pool과 3장 YES/NO 실험

COCO 이미지와 COCO-ReM 주석에서 그룹별 양성·음성 후보를 만들고, 이미지 3장에 대한 YES/NO 평가 입력과 그룹별 시각화를 생성합니다. Qwen2.5-VL로 이미지를 직접 입력해 답변을 확인하는 스크립트도 제공합니다.

## 준비된 3장 평가 입력 추론

`run_inference.py`는 기존 `data/inference_cases/inference_groups_seed42_balanced3_seed42_cases.jsonl`을 읽습니다. 모델은 `data/models/Qwen/Qwen2.5-VL-3B-Instruct/`의 로컬 파일만 사용하며, 이미지들은 `data/coco/val2017/`에서 `image_order` 순서대로 읽습니다. pool이나 평가 입력을 다시 만들 필요 없이 실행할 수 있습니다.

프로젝트 루트에서 패키지가 설치된 가상환경을 활성화한 뒤 실행합니다.

```bash
# 먼저 한 그룹 8개만 확인
python src/inference/run_inference.py --group group_01_cat \
  --output data/inference_results/cat_group01_results.jsonl

# 전체 800개 추론
python src/inference/run_inference.py

# 특정 클래스 또는 조건만 추론
python src/inference/run_inference.py --category cat --condition B \
  --output data/inference_results/cat_condition_B_results.jsonl

# 앞의 8개 샘플만 추론
python src/inference/run_inference.py --limit 8 \
  --output data/inference_results/pilot_results.jsonl
```

기본 출력은 `data/inference_results/inference_groups_seed42_balanced3_seed42_cases_results.jsonl`입니다. 한 줄이 샘플 하나이며 다음 **6개 필드만** 저장합니다.

| 필드 | 내용 |
| --- | --- |
| `case_id` | 입력 JSONL의 샘플 ID |
| `image_order` | 실제 입력 순서의 이미지 파일명 3개 |
| `model_answer` | 생성 답변 원문. 특수 토큰은 제거하고 Yes/No로 강제 변환하지 않음 |
| `answer` | Ground Truth. 입력 JSONL의 `gt`를 그대로 기록 |
| `yes_logit` | 첫 생성 토큰 위치에서 공백 없는 `Yes` 토큰의 원시 logit |
| `no_logit` | 같은 위치에서 공백 없는 `No` 토큰의 원시 logit |

모델에는 원본 이미지 3장과 JSONL에 저장된 `prompt`만 전달합니다. 정답, P/N 라벨, A/B/C/D 조건, plot 이미지는 전달하지 않습니다. `correct`, `p_yes_given_yes_no`, `logit_margin`은 후처리에서 계산합니다.

모델은 실행 중 한 번만 로딩하고 샘플을 하나씩 추론합니다. 기본 최대 생성 길이는 8토큰이며, `do_sample=False`, `num_beams=1`, `repetition_penalty=1.0`으로 생성합니다. 로컬 모델의 기본 반복 페널티와 temperature 대신 이번 실행에 명시한 설정을 사용합니다. `output_logits=True`의 원시 점수를 읽으며, 처리된 `scores`나 softmax 확률을 logit으로 저장하지 않습니다.

`Yes`와 `No`는 토크나이저로 단일 토큰 여부를 확인합니다. 소문자나 앞에 공백이 붙은 별도 토큰은 합산하지 않습니다. 실제 첫 출력이 줄바꿈이나 다른 토큰이어도 점수 측정 위치는 첫 생성 위치로 고정합니다. 따라서 이 점수는 해당 위치에서 두 후보의 선호도이며, 생성된 문장 전체가 정답일 확률은 아닙니다. 답변이 Yes/No 형식을 벗어나도 원문 그대로 저장해 후처리에서 판정합니다.

| 인자 | 기본값·역할 |
| --- | --- |
| `--input-jsonl` | 기본 balanced3 평가 JSONL |
| `--model-path` | `data/models/Qwen/Qwen2.5-VL-3B-Instruct/`, 로컬 Qwen2.5-VL 폴더 |
| `--output` | 결과 JSONL 파일 경로 |
| `--group`, `--category`, `--condition` | 그룹·클래스·A/B/C/D 필터, 여러 필터는 동시에 적용 |
| `--limit` | 필터 적용 후 앞에서부터 추론할 샘플 수 |
| `--max-new-tokens` | 8 |
| `--device` | `auto`: CUDA → MPS → CPU, 또는 장치 직접 지정 |

샘플 하나가 끝날 때마다 결과를 파일에 쓰고 flush합니다. 추론 오류가 나면 실패한 `case_id`를 출력하고 중단하며, 앞서 완료한 결과는 남습니다. 자동 이어하기 기능은 없으며 같은 출력 경로로 다시 실행하면 기존 파일을 덮어씁니다. 필터는 기본 출력 파일명에 포함되지 않으므로 파일럿·조건별 결과를 따로 보관하려면 예시처럼 `--output`을 지정합니다.

## 현재 실험의 실행 순서

프로젝트 루트에서 실행합니다. 이미지·주석·모델이 이미 있으면 다운로드는 생략합니다.

```bash
# 1. 클래스별 후보 pool 생성 (이미 data/pools/가 있으면 생략)
python src/data_manipulate/make_pool.py

# 2. 기본 20클래스 × 5그룹 생성
python src/data_manipulate/make_inference_pool.py

# 3. 이미지 3장씩 YES 400개·NO 400개 생성
python src/inference/build_cases.py --mode balanced3

# 4. 그룹당 PNG 1개 생성 (기본 8행 × 3열, 전체 100개)
python src/visualize/plot_inference_cases.py
```

평가 입력은 `data/inference_cases/inference_groups_seed42_balanced3_seed42_cases.jsonl`, 그림은 `data/inference_group_plots/<그룹명>.png`에 저장합니다. 각 단계는 파일을 통해 연결되며 다음 단계나 모델 추론을 자동 실행하지 않습니다. `build_cases.py`의 기존 기본 모드는 `paired`로 유지하므로 3장 실험에는 `--mode balanced3`를 지정합니다.

`plot_inference_groups.py`는 후보 12장 확인용 보조 도구입니다. 현재 3장 실험에는 필요하지 않으며, 평가 샘플 그림과 구분해 `data/inference_pool_plots/`에 저장합니다. 기존 `single`·`paired`·`permutation` 모드도 이전 실험 재현을 위해 유지합니다.

## 프로젝트 구조

```text
proj/
├── README.md
├── requirements.txt
└── src/
    ├── common/
    │   ├── paths.py            # 공통 데이터·모델 경로
    │   ├── data.py             # JSON 입출력, 그룹 검증·필터
    │   └── images.py           # EXIF 회전·RGB 이미지 로딩
    ├── download/
    │   ├── download_model.py   # Hugging Face 모델 다운로드
    │   └── download_data.py    # 이미지·annotation 다운로드 및 압축 해제
    ├── data_manipulate/
    │   ├── make_pool.py         # 카테고리별 양성·음성 후보 pool 생성
    │   └── make_inference_pool.py # 클래스별 8+8 pool에서 추론용 그룹 5개 생성
    ├── visualize/
    │   ├── plot_inference_cases.py # 현재 실험: 3장짜리 샘플을 그룹별 PNG로 저장
    │   ├── plot_pool_samples.py # 클래스별 양성 4장·음성 후보 4장 시각화
    │   └── plot_inference_groups.py # 보조: 그룹 후보 12장 시각화
    └── inference/
        ├── run_inference.py   # 기존 3장 평가 JSONL의 답변·Yes/No logits 생성
        └── inference_test.py  # 한 장 이상의 이미지를 직접 입력해 답변 생성
```

아래 명령은 프로젝트 루트에서 실행합니다. 저장 폴더는 다운로드 과정에서 생성됩니다.

## 환경 준비

Python 3.10 이상 환경에서 가상환경을 생성하고 필요한 패키지를 설치합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

이미 `.venv`가 있다면 생성 단계를 생략할 수 있습니다. `requirements.txt`는 사용하는 패키지 목록이며 버전 고정 파일은 아닙니다. 데이터 다운로드·pool 생성·평가 입력 생성은 Python 표준 라이브러리만 사용합니다. 이미지 처리를 공통 모듈로 분리했지만 이 데이터 생성 단계에서는 Pillow나 PyTorch를 불러오지 않습니다.

## 모델 다운로드

### 기본 모델

기본 모델은 [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)입니다.

```bash
python src/download/download_model.py
```

저장 위치:

```text
data/models/Qwen/Qwen2.5-VL-3B-Instruct/
```

### 다른 모델 지정

`--model-name`에 Hugging Face 저장소 ID를 전달합니다.

```bash
python src/download/download_model.py --model-name Qwen/Qwen2.5-3B-Instruct
```

모델 가중치, `config.json`, 토크나이저 등 저장소 파일이 `data/models/<모델 이름>/`에 저장됩니다. 위 예시의 저장 경로는 `data/models/Qwen/Qwen2.5-3B-Instruct/`입니다.

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--model-name` | `Qwen/Qwen2.5-VL-3B-Instruct` | 다운로드할 Hugging Face 모델 저장소 ID |
| `--help` | — | 사용법과 인자 설명 출력 |

### Python에서 사용

```python
from src.download.download_model import download_model

model_path = download_model()
print(model_path)
```

다른 모델은 `download_model(model_name="Qwen/Qwen2.5-3B-Instruct")`처럼 지정합니다.

## 이미지 직접 입력 추론

다운로드와 추론 코드 모두 기본 모델 경로로 `data/models/Qwen/Qwen2.5-VL-3B-Instruct/`를 사용합니다. 다른 모델을 다운로드할 수 있지만 이 추론 스크립트에는 Qwen2.5-VL 모델이 필요합니다.

추론에 필요한 패키지를 설치하고 VL 모델을 다운로드합니다.

```bash
python -m pip install -U torch torchvision transformers pillow
python src/download/download_model.py --model-name Qwen/Qwen2.5-VL-3B-Instruct
```

인자 없이 실행하면 `data/coco/val2017/000000124798.jpg`와 `data/coco/val2017/000000377393.jpg`를 사용하고, 기본 질문인 “두 이미지의 차이를 설명해줘”에 답합니다.

```bash
python src/inference/inference_test.py
```

다른 이미지는 `--image` 뒤에 한 개 이상의 경로를 입력 순서대로 지정합니다. 아래 경로는 실제 이미지 경로로 바꿉니다.

```bash
python src/inference/inference_test.py --image /path/to/image1.jpg /path/to/image2.jpg
```

3장 실험은 다음처럼 이미지와 질문을 직접 지정할 수 있습니다. 기본 질문은 기존 두 이미지 비교용이므로 3장 실험에서는 `--prompt`를 함께 지정합니다.

```bash
python src/inference/inference_test.py \
  --image /path/to/image1.jpg /path/to/image2.jpg /path/to/image3.jpg \
  --prompt "Is there a cat in all 3 images? Please answer with Yes or No."
```

이 명령은 직접 지정한 이미지 묶음 하나를 추론합니다. JSONL 전체 추론과 Yes/No logits 저장에는 `run_inference.py`를 사용합니다. 정답률 집계는 후처리 단계에서 수행합니다.

질문을 직접 지정할 수도 있습니다.

```bash
python src/inference/inference_test.py \
  --image /path/to/image1.jpg /path/to/image2.jpg \
  --prompt "두 이미지의 공통점과 차이점을 설명해줘." \
  --max-new-tokens 256
```

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--image` | `data/coco/val2017/`의 `000000124798.jpg`, `000000377393.jpg` | 입력 순서대로 이미지 경로 한 개 이상 지정 |
| `--model-path` | `data/models/Qwen/Qwen2.5-VL-3B-Instruct/` | 다운로드된 Qwen2.5-VL 모델 폴더 |
| `--prompt` | `두 이미지의 차이를 설명해줘` | 입력 이미지들에 대한 질문 |
| `--max-new-tokens` | `256` | 새로 생성할 최대 토큰 수 |
| `--device` | `auto` | `auto`, `cuda`, `mps`, `cpu` 중 선택 |

`auto`는 사용 가능한 CUDA, Mac MPS, CPU 순서로 장치를 선택합니다. CPU를 직접 지정하려면 `--device cpu`를 추가합니다.

모델 로딩, 이미지 읽기, 입력 전처리, 답변 생성은 각각 `load_model`, `load_image`, `prepare_inputs`, `infer_images` 함수로 분리되어 있습니다. 이미지들은 한 메시지에 입력 순서대로 들어가며, `generate()`를 한 번 호출해 답변 하나를 생성합니다. 추론 코드는 로컬 모델만 읽으며, 모델 파일이 없으면 다운로드 없이 오류를 표시합니다. 이미지 전처리는 이미지별로 최대 `1280 × 28 × 28` 픽셀로 제한하고, 입력 프롬프트를 제외한 새 답변만 출력합니다.

## 데이터 다운로드

다운로드 대상은 다음과 같습니다.

| 대상 | 원본 파일 | 압축 해제 후 경로 |
| --- | --- | --- |
| COCO 2017 validation 이미지 | [val2017.zip](http://images.cocodataset.org/zips/val2017.zip) | `data/coco/val2017/` |
| COCO-ReM annotation | [instances_valrem.json.zip](https://huggingface.co/datasets/kdexd/coco-rem/resolve/main/instances_valrem.json.zip) | `data/coco_rem/instances_valrem.json` |

**현재 `download_data.py`의 실행부에서는 `main()` 호출이 주석 처리되어 있습니다.** 다음 명령은 저장 경로만 출력합니다.

```bash
python src/download/download_data.py
```

실제 다운로드와 압축 해제를 시작하려면 프로젝트 루트에서 `main()`을 직접 호출합니다.

```bash
python -c "from src.download.download_data import main; raise SystemExit(main())"
```

이미지 다운로드 및 압축 해제가 끝나면 annotation 다운로드 및 압축 해제를 진행합니다.

생성되는 데이터 폴더:

```text
data/
├── archives/
│   ├── val2017.zip
│   └── instances_valrem.json.zip
├── coco/
│   └── val2017/
└── coco_rem/
    └── instances_valrem.json
```

모델과 데이터셋 모두 프로젝트 루트의 `data/` 아래에 저장합니다. 공통 저장 위치는 `src/common/paths.py`에서 정의합니다. 이전 경로에 있던 파일을 자동 이동하지는 않습니다.

### 재실행 시 동작

- 완성된 ZIP 파일이 있으면 재사용합니다. ZIP 읽기 오류는 압축 해제 단계에서 그대로 표시됩니다.
- 압축 해제된 파일의 크기와 CRC가 ZIP 내부 정보와 일치하면 해당 파일을 건너뜁니다.
- 다운로드 중인 파일은 `.part`로 저장하고, 완료 후 원래 파일명으로 변경합니다.
- 중단된 `.part` 파일은 다음 실행에서 처음부터 다시 다운로드합니다.
- 압축 해제 중에는 `.extracting` 임시 파일을 사용하고, 파일 쓰기가 끝나면 최종 경로로 변경합니다.

## 추론용 그룹 생성

기존 `data/pools/`에서 클래스별로 양성 8장과 음성 후보 8장을 중복 없이 뽑고, 양성 4장·음성 후보 8장으로 구성된 그룹을 클래스마다 5개씩 만듭니다. 기본 실행은 양성 후보가 많은 순서로 20개 클래스를 선택해 총 100그룹을 저장합니다. 후보 수가 같으면 카테고리 ID순으로 선택합니다. 기본 시드는 `42`이며, Python 표준 라이브러리만 사용합니다.

```bash
python src/data_manipulate/make_inference_pool.py
```

기본 출력은 `data/inference_pools/inference_groups_seed42.json`입니다. 최상위 키는 `group_01_cat`처럼 그룹 번호와 클래스명이고, 각 그룹에는 원래 클래스 이름인 `class`, `pos_image` 파일명 4개, `neg_image` 파일명 8개를 저장합니다. 그룹 키에서는 클래스명의 공백을 밑줄로 바꾸고, `class` 값에서는 원래 공백을 유지합니다. 이미지 파일은 복사하지 않으며, 실제 파일은 `data/coco/val2017/`에서 읽습니다.

양성 8장을 선택된 순서대로 P1~P8이라고 하면 그룹의 양성 구성은 다음과 같습니다. 모든 그룹의 음성 후보는 클래스에서 한 번 뽑은 동일한 N1~N8이며, 목록 순서도 같습니다.

| 그룹 | 양성 4장 | 음성 후보 |
| --- | --- | --- |
| G1 | P1, P2, P3, P4 | N1~N8 |
| G2 | P5, P6, P7, P8 | N1~N8 |
| G3 | P1, P3, P5, P7 | N1~N8 |
| G4 | P2, P4, P6, P8 | N1~N8 |
| G5 | P1, P4, P5, P8 | N1~N8 |

```bash
# cat 클래스만 선택해 5그룹 생성
python src/data_manipulate/make_inference_pool.py --category cat

# 클래스를 직접 여러 개 지정
python src/data_manipulate/make_inference_pool.py --category cat dog "traffic light"

# 양성 후보가 많은 순서로 10클래스 선택 → 50그룹
python src/data_manipulate/make_inference_pool.py --num-classes 10

# 다른 시드로 이미지 선택
python src/data_manipulate/make_inference_pool.py --seed 123
```

`--category`를 지정하면 `--num-classes` 대신 지정한 클래스 목록을 사용합니다. `--pool-dir`와 `--output-dir`로 후보 pool과 출력 폴더를 바꿀 수 있습니다. 같은 시드·출력 폴더로 다시 실행하면 선택한 클래스가 달라도 동일 JSON을 덮어쓰므로, 결과를 따로 보관하려면 `--output-dir`를 지정합니다.

양성·음성 후보 중 하나라도 8장 미만인 클래스는 자동 선택에서 제외하고 수량을 출력합니다. 사용자가 직접 지정한 클래스의 후보가 부족하거나, 요청한 수만큼 클래스를 선택할 수 없으면 오류가 발생하며 JSON을 저장하지 않습니다. 목록의 중복·양음성 겹침과 파일명으로 참조할 이미지의 경로 기준을 확인합니다. 실제 파일의 존재 여부는 이후 시각화나 추론에서 이미지를 열 때 확인됩니다. 음성은 주석 기준의 후보이며, 실제 이미지 내용은 재검증하지 않습니다. 같은 클래스의 후보 목록과 시드가 같으면 자동 선택과 직접 지정 모두 같은 이미지와 순서가 선택됩니다. 클래스 간 이미지 재사용은 허용합니다.

각 그룹의 12개 파일명은 후보 목록입니다. 이 목록에서 `build_cases.py`가 단독 이미지 평가, 양성 한 장 교체, 동일 이미지 구성의 순서 변경에 사용할 입력을 만듭니다. 그룹 생성 스크립트는 원본 이미지 선택과 그룹 구성까지만 수행합니다.

## YES/NO 평가 입력 생성

`build_cases.py`는 그룹 JSON에서 이미지 입력 순서·질문·정답을 정해 JSONL로 저장합니다. 한 줄이 평가 문제 하나이며, Python 표준 라이브러리만 사용합니다. 이미지 파일을 열거나 모델을 실행하지 않습니다.

```bash
# 단독 이미지의 객체 유무 판단
python src/inference/build_cases.py --mode single

# YES에서 양성 한 장만 음성으로 교체 (기본 모드)
python src/inference/build_cases.py --mode paired

# 동일한 양성 3장과 음성 1장의 순서 변경
python src/inference/build_cases.py --mode permutation
```

기본 입력은 `data/inference_pools/inference_groups_seed42.json`입니다. 인자 없이 실행하면 `paired` 모드로 생성합니다. 기본 출력은 입력 파일명과 모드에 따라 나뉩니다.

```text
data/inference_cases/
├── inference_groups_seed42_single_cases.jsonl
├── inference_groups_seed42_paired_cases.jsonl
└── inference_groups_seed42_permutation_cases.jsonl
```

각 명령은 선택한 모드의 파일 하나를 생성합니다. 그룹의 양성이 P1~P4, 사용할 음성이 N일 때 구성은 다음과 같습니다.

| 모드 | 생성 규칙 |
| --- | --- |
| `single` | 양성과 음성을 각각 한 장씩 입력. 클래스·파일명 조합으로 그룹 간 중복 제거 |
| `paired` | YES는 P1/P2/P3/P4. NO는 N/P2/P3/P4, P1/N/P3/P4, P1/P2/N/P4, P1/P2/P3/N |
| `permutation` | YES는 P1/P2/P3/P4. NO는 N/P1/P2/P3, P1/N/P2/P3, P1/P2/N/P3, P1/P2/P3/N |

`paired`는 교체한 한 장을 제외한 이미지와 위치를 유지합니다. `permutation`은 같은 양성 3장의 구성과 상대 순서를 유지하며, 음성 삽입에 따라 일부 양성의 절대 위치도 바뀝니다. 다중 이미지 모드는 음성마다 위치 0~3을 평가하고, 그룹당 YES는 한 번만 생성합니다.

프롬프트의 `{class_name}`에는 각 클래스명이 들어갑니다.

- `single`: `Is there at least one {class_name} in this image? Answer only Yes or No.`
- `paired`, `permutation`: `Is there at least one {class_name} in each of the 4 images? Answer only Yes or No.`

20클래스에서 양성 8장·음성 8장을 선택해 클래스마다 5그룹을 만든 기본 입력 기준 수량입니다.

| 모드 | 음성 8장 사용 | 음성 1장 사용 |
| --- | --- | --- |
| `single` | YES 160개 + NO 160개 = 320개 | YES 160개 + NO 20개 = 180개 |
| `paired` | YES 100개 + NO 3,200개 = 3,300개 | YES 100개 + NO 400개 = 500개 |
| `permutation` | YES 100개 + NO 3,200개 = 3,300개 | YES 100개 + NO 400개 = 500개 |

```bash
# 경량 파일럿: 그룹당 첫 음성만 사용 → YES 1개·NO 4개
python src/inference/build_cases.py \
  --mode paired --negatives-per-group 1 --output data/inference_cases/paired_pilot.jsonl

# 파일럿과 같은 이미지의 단독 평가 입력
python src/inference/build_cases.py \
  --mode single --negatives-per-group 1 --output data/inference_cases/single_pilot.jsonl

# 그룹 하나만 생성 → 기본 33개
python src/inference/build_cases.py \
  --mode paired --group group_01_person --output data/inference_cases/group_01_person_paired.jsonl

# 특정 클래스만 생성
python src/inference/build_cases.py \
  --mode permutation --category cat --output data/inference_cases/cat_permutation.jsonl
```

`--input-json`으로 다른 그룹 JSON을 지정할 수 있습니다. `--group`과 `--category`는 모든 모드에서 먼저 그룹을 선택합니다. `single`은 선택된 그룹에 등장하는 이미지들만 중복 제거하므로, 그룹 하나를 지정하면 양성 4장·음성 8장으로 기본 12개가 생성됩니다. 같은 이미지라도 클래스가 다르면 질문이 다르므로 별도 케이스입니다.

`--negatives-per-group 1`은 모든 모드에서 각 그룹의 첫 번째 음성만 사용합니다. 기본 출력 파일명은 `<입력 파일명>_<모드>_cases.jsonl`이며, 필터와 음성 사용 수를 바꾸어도 같은 모드의 기본 경로는 같습니다. 결과를 함께 보관하려면 `--output`에 서로 다른 경로를 지정합니다. 같은 출력 경로로 재실행하면 기존 JSONL을 덮어씁니다.

| 필드 | 내용 |
| --- | --- |
| `case_id` | 모드·클래스/그룹·입력 내용의 해시로 만든 식별자 |
| `experiment_type` | `single`, `paired`, `permutation` |
| `group_id` | 다중 이미지 케이스의 그룹명. `single`은 여러 그룹에서 공유하므로 `null` |
| `class` | 대상 클래스 |
| `condition`, `gt` | 조건은 `YES`/`NO`, 정답은 `Yes`/`No` |
| `image_order` | 실제 입력 순서의 파일명. `single`은 1개, 나머지는 4개 |
| `image_path_base`, `image_dir` | 프로젝트 루트 기준 `data/coco/val2017` |
| `prompt` | 한 장 또는 네 장 각각에 해당 클래스의 객체가 있는지 묻는 질문 |
| `negative_image` | 사용한 음성 파일명, YES이면 `null` |
| `negative_position` | 다중 이미지 NO의 음성 위치 0~3. `single`의 NO는 0, YES는 `null` |
| `excluded_positive` | `paired`는 해당 위치에서 교체한 양성, `permutation`은 네 번째 양성. YES와 `single`은 `null` |
| `baseline_case_id` | 다중 이미지 NO가 참조하는 같은 모드·그룹의 YES ID. YES와 `single`은 `null` |

같은 클래스의 음성 목록은 공유되므로 파일럿에서는 그 클래스의 그룹들이 동일한 첫 음성을 사용합니다. 같은 모드의 같은 입력 조건은 전체 실행과 파일럿·필터 실행에서도 같은 `case_id`를 가집니다. 서로 다른 모드의 ID는 이미지 순서가 같아도 구분됩니다. `single`은 ID에 그룹을 포함하지 않아 어느 그룹에서 선택했는지와 관계없이 같은 클래스·이미지·라벨·프롬프트가 같은 ID를 가집니다.

`paired`의 `baseline_case_id`는 한 장 교체 전후의 비교에 사용합니다. `permutation`의 YES는 그룹의 원래 양성 4장을 기록한 기준선이며, 각 NO와 비교하면 양성 위치도 바뀔 수 있습니다. 순서 민감도는 같은 그룹·음성의 NO 네 개를 비교합니다.

목록의 양음성 라벨을 기준으로 GT를 만들며 실제 이미지 내용은 재검증하지 않습니다. 추론 단계에서는 `image_order`의 이미지와 `prompt`만 모델에 전달하고, GT·음성 위치·기준선 ID 등은 평가용으로 보관해야 합니다. 모델 답변과 `p_yes`는 이후 추론 결과에 기록합니다.

## 3장 입력: YES 400개·NO 400개 생성

`balanced3` 모드는 기존 20클래스·100그룹에서 이미지 3장짜리 평가 문제를 총 800개 만듭니다. 한 샘플은 이미지 3장과 질문 하나로 구성된 평가 문제입니다. P는 대상 클래스의 양성 이미지, N은 음성 후보 이미지입니다.

```bash
python src/inference/build_cases.py --mode balanced3
```

| 조건 | 이미지 구성 및 입력 순서 | 정답 | 기본 100그룹 기준 |
| --- | --- | --- | --- |
| A | N + N + N | No | 134개 |
| B | P + N + N | No | 133개 |
| C | P + P + N | No | 133개 |
| D | P + P + P | Yes | 400개 |
| 합계 | 이미지 3장씩 | Yes 400개 / No 400개 | 800개 |

질문은 `Is there a {class_name} in all 3 images? Please answer with Yes or No.`입니다. 세 이미지 각각에 대상 객체가 있어야 Yes입니다. P를 앞에, N을 뒤에 배치하므로 이 모드는 음성 위치를 바꾸는 실험은 아닙니다.

그룹마다 양성 4장 중 3장을 고르는 모든 조합으로 Yes 4개를 만듭니다. No는 A/B/C를 하나씩 만든 뒤 추가 1개의 조건을 그룹 순서대로 A → B → C로 순환합니다. 따라서 그룹당 Yes 4개·No 4개, 클래스당 Yes 20개·No 20개이며, 전체 No의 세 조건 수량은 최대 1개 차이입니다.

No 이미지 조합은 기본 시드 `42`로 선택합니다. 같은 클래스에서 같은 이미지 3장의 조합은 순서가 달라도 중복으로 취급해 제외합니다. 한 문제 안에는 서로 다른 이미지 3장이 들어가며, 서로 다른 문제에서 개별 이미지를 재사용하는 것은 허용합니다. 클래스가 다르면 질문이 다르므로 같은 이미지 조합을 사용할 수 있습니다. 그룹 간 양음성 라벨 충돌, 중복 PPP 조합 또는 고유한 No 조합 부족이 있으면 저장 전에 오류를 냅니다.

기본 출력:

```text
data/inference_cases/inference_groups_seed42_balanced3_seed42_cases.jsonl
```

다른 음성 조합이나 일부 그룹만 생성하려면 다음과 같이 실행합니다.

```bash
# 양성 조합은 유지하고 음성 조합의 샘플링 시드를 변경
python src/inference/build_cases.py --mode balanced3 --seed 123

# cat의 5그룹 → Yes 20개·No 20개
python src/inference/build_cases.py --mode balanced3 --category cat \
  --output data/inference_cases/cat_balanced3.jsonl

# 한 그룹 → Yes 4개·No 4개
python src/inference/build_cases.py --mode balanced3 --group group_01_person \
  --output data/inference_cases/person_group01_balanced3.jsonl
```

전체·필터 실행에서 선택된 그룹의 순서대로 조건 수량을 다시 배분하고 중복을 제거하므로, 필터 실행의 No 조합은 전체 실행에서 해당 그룹만 추출한 결과와 다를 수 있습니다. 동일한 입력 JSON·필터·시드로 실행하면 결과가 같습니다. `--negatives-per-group 1`은 NNN 조건을 만들 수 없어 지원하지 않습니다. 필터는 기본 출력 파일명에 포함되지 않으므로 필터 결과를 별도 보관하려면 `--output`을 지정합니다.

기존 모드와 같은 `case_id`, `experiment_type`, `group_id`, `class`, `condition`, `image_path_base`, `image_dir`, `image_order`, `prompt`, `gt` 필드를 사용합니다. `condition`은 기존처럼 `YES`/`NO`이고, A/B/C/D는 별도의 `condition_code`에 기록합니다. 추가로 `composition`, `image_labels`, `positive_count`, `negative_count`를 저장합니다. 음성이 여러 장일 수 있으므로 단수 `negative_image`·`negative_position` 대신 `negative_images`·`negative_positions` 목록을 사용합니다. `excluded_positive`와 `baseline_case_id`는 사용하지 않습니다.

이 모드도 이미지 목록과 주석 기준 정답을 생성하며 모델을 실행하지 않습니다. 모델에는 `image_order`의 실제 이미지와 `prompt`만 전달해야 합니다. 생성된 JSONL은 `run_inference.py`로 순차 추론할 수 있습니다.

## 3장 평가 샘플을 그룹별로 확인

`plot_inference_cases.py`는 `balanced3` JSONL에 기록된 샘플을 그룹별로 모아 PNG 한 장씩 저장합니다. 새로 샘플링하지 않으며 JSONL의 샘플 순서와 이미지 순서를 유지합니다. 기본 100그룹에서는 그룹당 8행 × 3열이며, 한 행이 평가 문제 하나입니다. P/N, 조건 A~D, 정답과 질문을 함께 표시합니다.

```bash
# 전체 그룹 → PNG 100개
python src/visualize/plot_inference_cases.py

# 한 그룹 → PNG 1개
python src/visualize/plot_inference_cases.py --group group_01_cat

# cat의 B(PNN) 조건만 표시
python src/visualize/plot_inference_cases.py --category cat --condition B

# 앞의 두 그룹만 표시
python src/visualize/plot_inference_cases.py --limit 2
```

기본 저장 폴더는 `data/inference_group_plots/`입니다. 전체 그룹은 `group_01_cat.png`로 저장하고, 조건·정답 필터를 사용하면 `group_01_cat__condition_B.png`, `group_01_cat__gt_Yes.png`처럼 접미사를 붙입니다. 둘 다 지정하면 두 접미사가 모두 붙습니다. 필터를 사용한 그림이 전체 그룹 그림을 덮어쓰지 않습니다. 동일 조건으로 재실행하면 해당 PNG를 덮어씁니다.

`--input-jsonl`로 입력을 바꾸고 `--output-dir`로 저장 폴더를 지정할 수 있습니다. 여러 시드의 결과를 함께 보관하려면 서로 다른 출력 폴더를 지정합니다. `--image-dir`를 생략하면 JSONL의 이미지 폴더를 프로젝트 루트 기준으로 읽습니다. `--limit`은 필터를 적용한 뒤 저장할 그룹 수입니다.

## 보조: 그룹의 후보 12장 확인

`data/inference_pools/inference_groups_seed42.json`에 기록된 이미지를 다시 샘플링하지 않고, JSON의 목록 순서대로 그룹마다 PNG 한 장으로 저장합니다. `matplotlib`, `pillow`가 필요합니다.

```bash
python src/visualize/plot_inference_groups.py
```

그림은 3행 4열로 구성되며 첫 줄은 양성 4장, 둘째 줄은 음성 후보 1~4번, 셋째 줄은 음성 후보 5~8번입니다. 제목에는 그룹명과 `class`를, 각 이미지 위에는 POS/NEG 구분과 파일명을 표시합니다. 출력은 `data/inference_pool_plots/group_01_person.png`처럼 그룹별로 저장됩니다. 현재 실험의 그룹별 평가 그림과 저장 폴더를 분리했습니다.

```bash
# 그룹 하나만 확인
python src/visualize/plot_inference_groups.py --group group_01_person

# 해당 클래스의 모든 그룹 확인
python src/visualize/plot_inference_groups.py --category cat

# 다른 JSON과 별도 출력 폴더 지정
python src/visualize/plot_inference_groups.py \
  --input-json data/inference_pools/inference_groups_seed123.json \
  --output-dir data/inference_pool_plots_seed123
```

기본 이미지 폴더는 `data/coco/val2017/`이며 `--image-dir`로 바꿀 수 있습니다. `--group`과 `--category`를 함께 지정하면 두 조건을 모두 만족하는 그룹을 선택합니다. 같은 출력 폴더에서 다시 실행하면 같은 그룹 이름의 PNG를 덮어씁니다. 이미지를 읽지 못하면 원래 예외를 표시하며 실행이 중단됩니다. JSON이나 원본 이미지 파일은 수정하지 않습니다.

## 클래스별 pool 샘플 확인

`data/pools/<클래스>/positive.json`과 `negative_candidates.json`에서 각각 최대 4장을 무작위로 뽑아, 위쪽에 양성·아래쪽에 음성 후보를 배치한 2행 4열 PNG를 저장합니다. 이미지 경로는 프로젝트 루트를 기준으로 읽습니다.

필요한 패키지는 `matplotlib`, `pillow`입니다.

```bash
python -m pip install matplotlib pillow
python src/visualize/plot_pool_samples.py
```

출력은 `data/pool_samples/01_person.png`처럼 클래스별 파일로 저장됩니다. 같은 출력 폴더에서 다시 실행하면 해당 클래스의 PNG를 덮어씁니다.

```bash
# 특정 클래스만 확인
python src/visualize/plot_pool_samples.py --category person

# 다른 샘플을 선택하고 별도 폴더에 저장
python src/visualize/plot_pool_samples.py --seed 123 --output-dir data/pool_samples_seed123
```

`--category`는 클래스 이름(`person`, `"traffic light"`)이나 폴더명(`01_person`)을 받습니다. `--pool-dir`로 입력 폴더도 지정할 수 있습니다. 기본 시드는 `42`이며, pool 내용과 시드가 같으면 전체 클래스 실행과 특정 클래스 실행에서 같은 샘플이 선택됩니다.

후보가 4장보다 적으면 가진 이미지만 표시하고 나머지 칸에 `Not enough samples`를 표시합니다. 이미지를 읽지 못하면 원래 예외를 표시하며 실행이 중단됩니다. 음성 후보는 해당 클래스의 주석이 없는 이미지이며, 이 시각화가 실제 객체 부재를 자동으로 검증하지는 않습니다.
