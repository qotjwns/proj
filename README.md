# 실행 순서

프로젝트 루트에서 실행합니다. 데이터는 `data/`에, 모델은 `data/models/`에 저장됩니다.

## macOS

### 1. 가상환경 및 패키지 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. 데이터 다운로드 및 압축 해제

```bash
python -c "from src.download.download_data import main; raise SystemExit(main())"
```

### 3. Qwen 비전 모델 다운로드

기본 모델은 이미지 입력을 지원하는 Qwen2.5-VL-3B-Instruct입니다.

```bash
python src/download/download_model.py
```

7B 모델을 선택하려면 다음 명령을 사용합니다. 추론 시에도 같은 로컬 경로를 `--model-path`로 지정해야 합니다.

```bash
python src/download/download_model.py --model-name Qwen/Qwen2.5-VL-7B-Instruct
```

### 4. 이미지 후보 pool과 추론 그룹 생성

```bash
python src/data_manipulate/make_pool.py
python src/data_manipulate/make_inference_pool.py
```

첫 명령은 클래스별 양성·음성 후보를 만들고, 두 번째는 기본 20개 클래스에서 100개 그룹을 생성합니다.

### 5. 평가 입력이 준비되어 있으면 추론 실행

저장소에 `src/inference/build_cases.py`가 없어 pool에서 평가 JSONL을 새로 만드는 단계는 현재 제공되지 않습니다. 기존 JSONL이 `data/inference_cases/inference_groups_seed42_balanced3_seed42_cases.jsonl`에 있을 때만 다음 명령을 실행할 수 있습니다.

```bash
python src/inference/run_inference.py --limit 8 --output data/inference_results/pilot_results.jsonl
python src/inference/run_inference.py
```

macOS에서는 CUDA 대신 지원되는 Mac에서 MPS를 사용하고, 그렇지 않으면 CPU로 실행합니다.

## Windows (PowerShell)

### 1. 가상환경 및 패키지 설치

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

NVIDIA GPU를 사용한다면 PyTorch CUDA 빌드를 설치합니다. 아래는 CUDA 13.2 빌드 예시입니다.

```powershell
python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu132
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### 2. 데이터 다운로드 및 압축 해제

```powershell
python -c "from src.download.download_data import main; raise SystemExit(main())"
```

### 3. Qwen 비전 모델 다운로드

기본 모델은 이미지 입력을 지원하는 Qwen2.5-VL-3B-Instruct입니다.

```powershell
python src/download/download_model.py
```

7B 모델을 선택하려면 다음 명령을 사용합니다. 추론 시에도 같은 로컬 경로를 `--model-path`로 지정해야 합니다.

```powershell
python src/download/download_model.py --model-name Qwen/Qwen2.5-VL-7B-Instruct
```

### 4. 이미지 후보 pool과 추론 그룹 생성

```powershell
python src/data_manipulate/make_pool.py
python src/data_manipulate/make_inference_pool.py
```

첫 명령은 클래스별 양성·음성 후보를 만들고, 두 번째는 기본 20개 클래스에서 100개 그룹을 생성합니다.

### 5. 평가 입력이 준비되어 있으면 추론 실행

저장소에 `src/inference/build_cases.py`가 없어 pool에서 평가 JSONL을 새로 만드는 단계는 현재 제공되지 않습니다. 기존 JSONL이 `data\inference_cases\inference_groups_seed42_balanced3_seed42_cases.jsonl`에 있을 때만 다음 명령을 실행할 수 있습니다.

```powershell
python src/inference/run_inference.py --limit 8 --output data/inference_results/pilot_results.jsonl
python src/inference/run_inference.py
```

모델을 바꾸려면 `--model-path`에 로컬 비전 모델 폴더를 지정합니다. 예를 들어 Qwen2.5-VL-7B-Instruct는 `Qwen/Qwen2.5-VL-7B-Instruct`입니다. `Qwen2.5-7B-Instruct`처럼 `VL`이 없는 텍스트 전용 모델은 이 이미지 추론 코드에서 사용할 수 없습니다. 7B 모델은 GPU 메모리 부족으로 실행에 실패할 수 있습니다.
