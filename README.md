```bash
cd /Users/baeseojun/proj
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/dataset/validate_data.py
.venv/bin/python src/dataset/prepare_pools.py
.venv/bin/python src/visualization/create_pool_examples.py
.venv/bin/python src/inference/download_model.py qwen2.5-vl-3b-instruct
.venv/bin/python src/inference/run_e1.py qwen2.5-vl-3b-instruct
.venv/bin/python src/visualization/plot_e1.py qwen2.5-vl-3b-instruct
```

`configs/model.json`에 사용할 모델들을 이름별로 등록합니다.
커맨드에 넘긴 이름으로 설정을 찾아 해당 모델을 다운로드하거나 추론합니다.

```json
{
  "qwen2.5-vl-3b-instruct": {
    "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
    "save_dir": "data/models/qwen2.5-vl-3b-instruct",
    "revision": "66285546d2b821cf421d4f5eb2576359d3770cd3"
  },
  "qwen2.5-vl-7b-instruct": {
    "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
    "save_dir": "data/models/qwen2.5-vl-7b-instruct",
    "revision": "cc594898137f460bfe9f0759e9844b3ce807cfb5"
  }
}
```

예를 들어 7B 모델을 선택하려면 두 명령에 같은 이름을 넘깁니다.

```bash
.venv/bin/python src/inference/download_model.py qwen2.5-vl-7b-instruct
.venv/bin/python src/inference/run_e1.py qwen2.5-vl-7b-instruct
.venv/bin/python src/visualization/plot_e1.py qwen2.5-vl-7b-instruct
```

`save_dir`의 상대경로는 프로젝트 루트를 기준으로 해석합니다.
각 명령의 `--list-models`로 등록된 모델 목록을 확인할 수 있습니다.
모델 이름에는 영문·숫자·점·밑줄·하이픈을 사용하며 영문이나 숫자로 시작합니다.
추론은 다운로드된 `save_dir`에서 `AutoProcessor`와 `AutoModelForImageTextToText`를 로드합니다.
추가 모델도 config에 같은 구조로 등록하면 명령에서 선택할 수 있습니다.
현재 E1 입력·점수 계산은 다중 이미지 채팅, `image_token_id`, 단일 토큰 Yes/No를 사용하는 모델을 전제로 합니다.

새 실행 결과는 `logs/<모델 이름>/`, 시각화는 `src/visualization/outputs/<모델 이름>/`에 저장합니다.
기존 `logs/`에 같은 모델 설정으로 저장된 실행이 있으면 해당 로그를 이어서 사용합니다.
