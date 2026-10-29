# python src/download/download_model.py --model-name Qwen/Qwen2.5-VL-3B-Instruct

import argparse
from pathlib import Path

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from huggingface_hub import snapshot_download

from src.common.paths import DEFAULT_MODEL, MODEL_DIR


def download_model(model_name: str = DEFAULT_MODEL) -> Path:

    save_dir = MODEL_DIR / model_name

    print(f"다운로드 모델: {model_name}", flush=True)
    print(f"저장 경로: {save_dir}", flush=True)

    model_path = snapshot_download(repo_id=model_name, repo_type="model", local_dir=str(save_dir))

    return Path(model_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="모델 이름으로 Hugging Face 모델을 다운로드")
    parser.add_argument("--model-name", default=DEFAULT_MODEL,
        help="Hugging Face 모델 이름 (기본값: %(default)s)",
    )
    args = parser.parse_args()

    model_path = download_model(model_name=args.model_name)
    print(f"다운로드 완료: {model_path}")


if __name__ == "__main__":
    main()
