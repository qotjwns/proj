# .venv/bin/python src/inference/download_model.py qwen2.5-vl-3b-instruct
import sys
from huggingface_hub import snapshot_download

from model_config import parse_model_config


def main() -> int:
    model_name, config, save_dir = parse_model_config(
        "configs/model.json에 등록된 Hugging Face 모델을 다운로드합니다."
    )
    model_id, revision = config["model_id"], config["revision"]

    print(f"모델: {model_name} ({model_id})")
    print(f"버전: {revision}")
    print(f"저장 경로: {save_dir}", flush=True)

    downloaded_path = snapshot_download(
        repo_id=model_id,
        repo_type="model",
        revision=revision,
        local_dir=str(save_dir),
    )

    print(f"다운로드 완료: {downloaded_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
