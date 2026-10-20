# 모델·데이터 다운로드 및 이미지 추론

Hugging Face 모델과 COCO 2017 validation 이미지, COCO-ReM annotation을 다운로드하고, Qwen2.5-VL로 이미지 두 장을 비교하는 Python 스크립트입니다.

## 프로젝트 구조

```text
proj/
├── README.md
└── src/
    ├── download/
    │   ├── download_model.py   # Hugging Face 모델 다운로드
    │   └── download_data.py    # 이미지·annotation 다운로드 및 압축 해제
    ├── data_manipulate/
    │   ├── make_pool.py         # 카테고리별 양성·음성 후보 pool 생성
    │   └── plot_pool_samples.py # 클래스별 양성 4장·음성 후보 4장 시각화
    └── inference/
        └── inference_test.py  # 이미지 두 장을 비교하는 텍스트 답변 생성
```

아래 명령은 프로젝트 루트에서 실행합니다. 저장 폴더는 다운로드 과정에서 생성됩니다.

## 환경 준비

Python 3 환경에서 가상환경을 생성하고 필요한 패키지를 설치합니다.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install huggingface_hub
```

이미 `.venv`가 있다면 생성 단계를 생략할 수 있습니다. 모델 다운로드에는 `huggingface_hub`를 사용하고, 데이터 다운로드에는 Python 표준 라이브러리를 사용합니다.

## 모델 다운로드

### 기본 모델

기본 모델은 [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)입니다.

```bash
python src/download/download_model.py
```

저장 위치:

```text
data/models/Qwen/Qwen2.5-0.5B-Instruct/
```

### 다른 모델 지정

`--model-name`에 Hugging Face 저장소 ID를 전달합니다.

```bash
python src/download/download_model.py --model-name Qwen/Qwen2.5-3B-Instruct
```

모델 가중치, `config.json`, 토크나이저 등 저장소 파일이 `data/models/<모델 이름>/`에 저장됩니다. 위 예시의 저장 경로는 `data/models/Qwen/Qwen2.5-3B-Instruct/`입니다.

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--model-name` | `Qwen/Qwen2.5-0.5B-Instruct` | 다운로드할 Hugging Face 모델 저장소 ID |
| `--help` | — | 사용법과 인자 설명 출력 |

### Python에서 사용

```python
from src.download.download_model import download_model

model_path = download_model()
print(model_path)
```

다른 모델은 `download_model(model_name="Qwen/Qwen2.5-3B-Instruct")`처럼 지정합니다.

## 이미지 두 장 비교

기본 다운로드 모델인 `Qwen2.5-0.5B-Instruct`는 텍스트 전용입니다. 이미지 추론에는 별도의 [Qwen2.5-VL 모델](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)이 필요하며, 추론 코드의 기본 모델 경로는 `data/models/Qwen/Qwen2.5-VL-3B-Instruct/`입니다.

추론에 필요한 패키지를 설치하고 VL 모델을 다운로드합니다.

```bash
python -m pip install -U torch torchvision transformers pillow
python src/download/download_model.py --model-name Qwen/Qwen2.5-VL-3B-Instruct
```

인자 없이 실행하면 `data/coco/val2017/000000124798.jpg`와 `data/coco/val2017/000000377393.jpg`를 사용하고, 기본 질문인 “두 이미지의 차이를 설명해줘”에 답합니다.

```bash
python src/inference/inference_test.py
```

다른 이미지는 `--image` 뒤에 두 경로를 순서대로 지정합니다. 아래 경로는 실제 이미지 경로로 바꿉니다.

```bash
python src/inference/inference_test.py --image /path/to/image1.jpg /path/to/image2.jpg
```

질문을 직접 지정할 수도 있습니다.

```bash
python src/inference/inference_test.py \
  --image /path/to/image1.jpg /path/to/image2.jpg \
  --prompt "두 이미지의 공통점과 차이점을 설명해줘." \
  --max-new-tokens 256
```

| 인자 | 기본값 | 설명 |
| --- | --- | --- |
| `--image` | `data/coco/val2017/`의 `000000124798.jpg`, `000000377393.jpg` | 첫 번째·두 번째 이미지 경로, 정확히 두 개 지정 |
| `--model-path` | `data/models/Qwen/Qwen2.5-VL-3B-Instruct/` | 다운로드된 Qwen2.5-VL 모델 폴더 |
| `--prompt` | `두 이미지의 차이를 설명해줘` | 두 이미지에 대한 질문 |
| `--max-new-tokens` | `256` | 새로 생성할 최대 토큰 수 |
| `--device` | `auto` | `auto`, `cuda`, `mps`, `cpu` 중 선택 |

`auto`는 사용 가능한 CUDA, Mac MPS, CPU 순서로 장치를 선택합니다. CPU를 직접 지정하려면 `--device cpu`를 추가합니다.

모델 로딩, 이미지 읽기, 입력 전처리, 답변 생성은 각각 `load_model`, `load_image`, `prepare_inputs`, `infer_images` 함수로 분리되어 있습니다. 두 이미지는 한 메시지에 입력 순서대로 들어가며, `generate()`를 한 번 호출해 답변 하나를 생성합니다. 추론 코드는 로컬 모델만 읽으며, 모델 파일이 없으면 다운로드 없이 오류를 표시합니다. 이미지 전처리는 이미지별로 최대 `1280 × 28 × 28` 픽셀로 제한하고, 입력 프롬프트를 제외한 새 답변만 출력합니다.

## 데이터 다운로드

다운로드 대상은 다음과 같습니다.

| 대상 | 원본 파일 | 압축 해제 후 경로 |
| --- | --- | --- |
| COCO 2017 validation 이미지 | [val2017.zip](http://images.cocodataset.org/zips/val2017.zip) | `src/download/data/coco/val2017/` |
| COCO-ReM annotation | [instances_valrem.json.zip](https://huggingface.co/datasets/kdexd/coco-rem/resolve/main/instances_valrem.json.zip) | `src/download/data/coco_rem/instances_valrem.json` |

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
src/download/data/
├── archives/
│   ├── val2017.zip
│   └── instances_valrem.json.zip
├── coco/
│   └── val2017/
└── coco_rem/
    └── instances_valrem.json
```

모델은 프로젝트 루트의 `data/models/`에, 데이터셋은 `download_data.py`가 위치한 폴더 아래의 `data/`에 저장됩니다. 각 스크립트의 `MODEL_DIR`, `DATA_DIR`에서 저장 위치를 정의합니다.

### 재실행 시 동작

- 완성된 ZIP 파일이 있으면 ZIP 형식을 확인한 뒤 재사용합니다.
- 압축 해제된 파일의 크기와 CRC가 ZIP 내부 정보와 일치하면 해당 파일을 건너뜁니다.
- 다운로드 중인 파일은 `.part`로 저장하고, 완료 후 원래 파일명으로 변경합니다.
- 중단된 `.part` 파일은 다음 실행에서 처음부터 다시 다운로드합니다.
- 압축 해제 중에는 `.extracting` 임시 파일을 사용하고, 파일 쓰기가 끝나면 최종 경로로 변경합니다.

## 클래스별 pool 샘플 확인

`data/pools/<클래스>/positive.json`과 `negative_candidates.json`에서 각각 최대 4장을 무작위로 뽑아, 위쪽에 양성·아래쪽에 음성 후보를 배치한 2행 4열 PNG를 저장합니다. 이미지 경로는 프로젝트 루트를 기준으로 읽습니다.

필요한 패키지는 `matplotlib`, `pillow`입니다.

```bash
python -m pip install matplotlib pillow
python src/data_manipulate/plot_pool_samples.py
```

출력은 `data/pool_samples/01_person.png`처럼 클래스별 파일로 저장됩니다. 같은 출력 폴더에서 다시 실행하면 해당 클래스의 PNG를 덮어씁니다.

```bash
# 특정 클래스만 확인
python src/data_manipulate/plot_pool_samples.py --category person

# 다른 샘플을 선택하고 별도 폴더에 저장
python src/data_manipulate/plot_pool_samples.py --seed 123 --output-dir data/pool_samples_seed123
```

`--category`는 클래스 이름(`person`, `"traffic light"`)이나 폴더명(`01_person`)을 받습니다. `--pool-dir`로 입력 폴더도 지정할 수 있습니다. 기본 시드는 `42`이며, pool 내용과 시드가 같으면 전체 클래스 실행과 특정 클래스 실행에서 같은 샘플이 선택됩니다.

후보가 4장보다 적으면 가진 이미지만 표시하고 나머지 칸에 `Not enough samples`를 표시합니다. 읽지 못한 이미지는 해당 칸에 `Image load failed`를 표시하고 터미널에 경로와 오류를 출력합니다. 음성 후보는 해당 클래스의 주석이 없는 이미지이며, 이 시각화가 실제 객체 부재를 자동으로 검증하지는 않습니다.
