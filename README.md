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

### 5. 평가 입력 생성 및 추론

`build_cases.py`가 그룹 JSON에서 3장짜리 Yes 400개·No 400개를 만듭니다. 기존 `data/inference_cases/inference_groups_seed42_balanced3_seed42_cases.jsonl`이 있으면 생성 단계는 건너뛰고 그대로 추론에 사용합니다.

```bash
# cases JSONL이 없을 때만 생성
python src/inference/build_cases.py

# 준비된 cases JSONL로 추론
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

### 5. 평가 입력 생성 및 추론

`build_cases.py`가 그룹 JSON에서 3장짜리 Yes 400개·No 400개를 만듭니다. 기존 `data\inference_cases\inference_groups_seed42_balanced3_seed42_cases.jsonl`이 있으면 생성 단계는 건너뛰고 그대로 추론에 사용합니다.

```powershell
# cases JSONL이 없을 때만 생성
python src/inference/build_cases.py

# 준비된 cases JSONL로 추론
python src/inference/run_inference.py --limit 8 --output data/inference_results/pilot_results.jsonl
python src/inference/run_inference.py
```

모델을 바꾸려면 `--model-path`에 로컬 비전 모델 폴더를 지정합니다. 예를 들어 Qwen2.5-VL-7B-Instruct는 `Qwen/Qwen2.5-VL-7B-Instruct`입니다. `Qwen2.5-7B-Instruct`처럼 `VL`이 없는 텍스트 전용 모델은 이 이미지 추론 코드에서 사용할 수 없습니다. 7B 모델은 GPU 메모리 부족으로 실행에 실패할 수 있습니다.

## 평가 입력 생성 코드

`src/inference/build_cases.py` 한 파일에서 현재 사용하는 `balanced3`만 생성합니다. argparse 없이 상단의 `SEED`, `GROUPS_JSON`, `OUTPUT_JSONL`로 설정합니다.

- 입력: `data/inference_pools/inference_groups_seed42.json`의 100개 그룹
- 출력: `data/inference_cases/inference_groups_seed42_balanced3_seed42_cases.jsonl`
- 그룹당 PPP 4개와 No 4개. 전체 D(PPP) 400개, A(NNN) 134개, B(PNN) 133개, C(PPN) 133개입니다.
- No는 A/B/C 하나씩에 그룹 순서대로 A→B→C 하나를 추가합니다. P를 앞에, N을 뒤에 두며 같은 클래스의 이미지 3장 조합은 중복을 제외합니다.
- 질문은 `Is there a {object} in all 3 images? Please answer with Yes or No.`입니다. 기존 JSONL의 필드와 `case_id` 해시 규칙을 유지하고, `negative_positions`는 0부터 시작합니다.

사라진 원본 대신 생성 규칙에 맞춰 재작성한 코드이므로, 시드가 42여도 이전 파일의 음성 조합을 그대로 재현한다고 보장하지 않습니다. 진행 중인 A/B/C 비교에는 기존 800개 JSONL을 계속 사용합니다. 새 조합이 필요하면 `OUTPUT_JSONL`을 다른 파일명으로 바꿉니다. 파일이 이미 존재하면 덮어쓰지 않고 `FileExistsError`로 중단합니다. 이미지 복사나 모델 실행은 하지 않습니다.

## 기본 성능 평가 (macOS / Windows 공통)

저장된 `model_answer`와 `gt`로 정확도, 혼동행렬, Precision·Recall·F1, Yes 응답률을 계산합니다. PNG 생성에는 `requirements.txt`에 포함된 Matplotlib을 사용합니다.

```bash
# data/inference_results/*_results.jsonl을 파일별로 평가
python src/evaluation/evaluate_results.py

# 결과 파일 하나만 평가
python src/evaluation/evaluate_results.py --input-jsonl data/inference_results/Qwen2.5-VL-3B-Instruct_results.jsonl
```

실행하면 터미널에 지표를 표시하고 `data/evaluation_results/<모델명>_metrics.png`에 저장합니다. PNG 한 장에 모델명, 샘플 수, 지표 표, 혼동행렬을 담습니다. 저장 폴더는 `--output-dir`로 바꿀 수 있습니다.

- **양성은 Yes**입니다. TP는 정답 Yes·예측 Yes, TN은 정답 No·예측 No, FP는 정답 No·예측 Yes, FN은 정답 Yes·예측 No입니다.
- 정확도는 `(TP + TN) / 전체 샘플 수`, Precision은 `TP / (TP + FP)`, Recall은 `TP / (TP + FN)`, F1은 `2TP / (2TP + FP + FN)`입니다.
- Yes 응답률은 `(TP + FP) / 전체 샘플 수`입니다.
- 비율 지표는 PNG와 터미널에 백분율로 표시하며, 혼동행렬은 샘플 수로 표시합니다. 지표의 분모가 0이면 0으로 표시합니다.
- 답변의 대소문자, 양끝 공백, 끝 문장부호(`.!?。！？`)를 정리하므로 `Yes.`도 `Yes`로 평가합니다. 정리 후 단독 Yes/No가 아니거나 필수 답변 필드가 없으면 해당 결과 위치를 알리고 중단합니다. 빈 결과 파일도 오류로 처리합니다.

## 이미지 구성별 오류 표

`case_id`로 추론 결과와 원본 cases JSONL을 연결하여 `3N`, `1P+2N`, `2P+1N`, `NO 전체`, `3P` 순서의 PNG 표를 만듭니다. P는 대상이 있는 이미지, N은 대상이 없는 이미지입니다.

```bash
# 결과 파일별로 구성별 오류 표 생성
python src/evaluation/evaluate_composition.py

# 3B 결과만 평가
python src/evaluation/evaluate_composition.py --input-jsonl data/inference_results/Qwen2.5-VL-3B-Instruct_results.jsonl
```

저장 위치는 `data/evaluation_results/<모델명>_composition_errors.png`입니다. `--cases-jsonl`로 원본 cases 파일을, `--output-dir`로 저장 폴더를 바꿀 수 있습니다.

- `3N`, `1P+2N`, `2P+1N`은 정답이 No이므로 Yes로 잘못 답한 **False-Yes** 수를 집계합니다. `NO 전체`는 이 세 행의 합계입니다.
- `3P`는 정답이 Yes이므로 No로 잘못 답한 **False-No** 수를 집계합니다.
- 오류율은 **해당 행의 오류 수 / 해당 행의 샘플 수**입니다. 샘플이 없는 행은 오류율을 `N/A`로 표시합니다.
- 결과 파일에 실제로 있는 샘플만 집계합니다. 중복 case_id, 원본 케이스 누락, 이미지 순서·정답 불일치, 해석 불가능한 답변은 오류로 처리합니다.
- 한국어 글꼴을 찾으면 한국어 표를 저장하며, 한국어 글꼴이 없는 환경에서는 영어로 표시합니다.

## False-Yes / False-No 이미지별 추론

기존 3장 입력에서 틀린 케이스를 선택하고, 케이스 안의 이미지 3장을 **각각 독립적으로** 모델에 넣습니다. 원래 질문의 `in all 3 images`만 `in this image`로 바꾸며, 이미지 한 장의 정답은 P이면 Yes, N이면 No입니다.

```bash
# 기본 3B 모델: False-Yes와 False-No 모두
python src/inference/run_single_image_errors.py

# 오류 유형 하나만 선택
python src/inference/run_single_image_errors.py --error-type false-yes
python src/inference/run_single_image_errors.py --error-type false-no

# 7B 모델과 해당 모델의 기존 결과 사용
python src/inference/run_single_image_errors.py --model-path data/models/Qwen/Qwen2.5-VL-7B-Instruct

# 원본 오류 케이스 2개만 확인하고 별도 파일로 저장
python src/inference/run_single_image_errors.py --limit 2 --output data/single_image_results/pilot.jsonl
```

기본 입력은 `data/inference_results/<모델명>_results.jsonl`, 출력은 `data/single_image_results/<모델명>_<오류종류>_single_image_results.jsonl`입니다. `--input-jsonl`로 입력 파일을 직접 지정할 때에도 사용할 모델은 `--model-path`로 지정합니다. `--cases-jsonl`, `--device`, `--max-new-tokens`도 지정할 수 있습니다.

- 결과 한 행은 원본 케이스의 이미지 한 장에 해당합니다. `case_id`와 1부터 시작하는 `image_position`으로 원본 케이스와 위치를 찾을 수 있습니다.
- `source_error_type`은 원래 3장 입력의 오류 유형이며, `source_result`에는 그때의 답변·정답·이미지 순서·logit이 보존됩니다.
- `image_label`, `gt`, `prompt`, `model_answer`, `yes_logit`, `no_logit`은 한 장씩 입력한 결과입니다. Yes/No로 해석할 수 없는 생성 답변도 저장하고 `normalized_answer`와 `is_correct`를 `null`로 기록합니다.
- 같은 이미지와 같은 질문이 반복되면 추론은 한 번만 하고, 각 원본 케이스·위치에 결과를 기록합니다. 같은 이미지라도 대상 클래스 등 질문이 달라지면 별도로 추론합니다.
- 분석 대상이 원래 틀린 케이스에 한정되므로, 이 결과의 정확도는 전체 데이터셋의 단일 이미지 성능을 의미하지 않습니다.

## 단일 이미지 추론 결과 표

```bash
python src/evaluation/evaluate_single_image_results.py
```

`data/single_image_results/*_single_image_results.jsonl`을 파일별로 읽고, `data/evaluation_results/<모델명>_<오류종류>_single_image_summary.png`를 만듭니다. `--input-jsonl`로 파일 하나를, `--output-dir`로 저장 폴더를 지정할 수 있습니다. 계산과 표 생성은 `evaluate_single_image_results.py`에 함께 있습니다.

PNG에는 기본 지표·혼동행렬 수치, 원래 False-Yes/False-No 및 P/N별 오류 수, 원본 케이스별로 개별 이미지 3장을 모두 맞힌 비율을 표시합니다. 기본 지표는 케이스 내 이미지 출현 횟수 기준과 동일 모델·이미지·질문 중복 제거 기준을 함께 제공합니다. 누락된 이미지 위치, 중복 케이스·위치, 해석 불가능한 답변은 오류로 처리합니다.

## 단일 이미지 예측 AND와 기존 예측 비교

```bash
python src/evaluation/compare_single_image_and.py
```

저장된 단일 이미지 예측을 `case_id`로 묶고 입력 위치 1·2·3 순서로 AND 계산합니다. **세 예측이 모두 Yes일 때만 최종 Yes**, 하나라도 No이면 최종 No입니다. `all(개별 예측 == 개별 정답)`과는 다른 계산이며, 정답 라벨은 AND 예측을 만든 후 평가할 때만 사용합니다.

동일 케이스의 `source_result`에 보존된 기존 3장 동시 입력 예측과 비교하여 `data/evaluation_results/<모델명>_<오류종류>_and_comparison.png`를 저장합니다. 같은 이름의 CSV에는 이미지 3개, 각 단일 예측, AND 예측, 기존 예측, 정답과 정오 여부를 남깁니다. `--input-jsonl`과 `--output-dir`을 지정할 수 있으며 모델 추론은 추가로 실행하지 않습니다.

현재 단일 이미지 파일은 **기존 오답 케이스만** 포함하므로, 비교표의 기존 정확도는 해당 부분집합에서 0%입니다. AND 정확도 역시 같은 부분집합 기준입니다. 전체 데이터셋에서 AND 성능을 비교하려면 기존 정답 케이스에도 단일 이미지 추론이 필요합니다.

## 전체 800개 Single + Python AND 실험

```bash
# 3B: 전체 케이스의 단일 이미지 추론 → Python AND → 기존 결과 비교
python src/inference/run_single_image_and.py

# 기존 오류 케이스의 단일 이미지 결과를 재사용하고 빠진 이미지·질문만 추론
python src/inference/run_single_image_and.py --reuse-jsonl data/single_image_results/Qwen2.5-VL-3B-Instruct_all_single_image_results.jsonl

# 7B 모델과 해당 모델의 전체 기존 결과 사용
python src/inference/run_single_image_and.py --model-path data/models/Qwen/Qwen2.5-VL-7B-Instruct
```

`run_single_image_and.py`는 기존 정답 여부에 따른 필터 없이 전체 cases와 기존 추론 결과를 연결합니다. 두 파일의 케이스 목록이 같고 기본 800개인지 확인한 후, 케이스마다 이미지 3장을 각각 독립적으로 추론합니다. 같은 이미지·질문은 한 번만 추론하고 각 케이스의 위치에 재사용합니다. 800개 케이스이면 이미지별 결과는 2,400행입니다.

**세 단일 예측이 모두 Yes이면 최종 Yes, 하나라도 No이면 No**입니다. 정답 라벨과 기존 모델의 예측은 AND 결합에 사용하지 않습니다. 기존 정답 617개와 오답 183개를 모두 포함하는 현재 3B 결과에서는, AND로 개선되는 경우와 기존 정답을 놓치게 되는 경우를 함께 비교할 수 있습니다.

실행 시 `data/single_image_and/<모델명>/`에 다음을 저장합니다.

- `single_image_results.jsonl`: 케이스·이미지 위치별 단일 예측과 logit, 개별 정답, 원본 결과. 기존 정답인 케이스는 `source_error_type`이 `null`이며, `source_outcome`은 TP/TN/FP/FN입니다.
- `and_results.jsonl`: 케이스별 단일 예측 3개, AND 최종 예측, 기존 예측, 정답과 정오 변화.
- `comparison.csv`: 같은 케이스별 비교를 표 형태로 저장한 파일.
- `comparison.png`: 동일한 전체 케이스에서의 기본 지표·혼동행렬 수치와 정답→오답, 오답→정답 등의 변화 표.

`--input-jsonl`, `--cases-jsonl`, `--output-dir`, `--device`, `--max-new-tokens`를 지정할 수 있습니다. 다른 크기의 데이터셋에는 `--expected-cases`로 검증할 개수를 지정합니다. 일부 케이스만 처리한 결과를 전체 결과로 표시하지 않도록, 목록·개수·정답·이미지 순서가 맞지 않으면 중단합니다.

재사용은 `--reuse-jsonl`로 명시했을 때만 수행하며 모델 경로·이미지 경로·질문·정답을 확인합니다. `max_new_tokens`가 기록된 파일은 현재 설정과도 비교합니다. 이전 오류 전용 결과에는 생성 설정이 없으므로, 이 파일은 동일한 모델과 생성 설정을 유지할 때 재사용하세요. 실행 중 중단된 경우에도 이미 저장된 `single_image_results.jsonl`을 재사용 입력으로 지정할 수 있습니다. Yes/No로 해석할 수 없는 새 답변은 원문을 저장하고 중단하며, 미완성 결과로 비교표를 만들지 않습니다.
