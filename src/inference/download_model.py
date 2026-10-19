# .venv/bin/python src/inference/download_model.py configs/model.json
import argparse
import json
from pathlib import Path
import sys
from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_config(config_path: Path) -> tuple[str, Path, str]:
    config = json.loads(config_path.expanduser().read_text(encoding="utf-8"))

    save_dir = (PROJECT_ROOT / Path(config["save_dir"]).expanduser()).resolve()

    return config["model_id"], save_dir, config["revision"]

def main() -> int:
    parser = argparse.ArgumentParser(
        description="JSON 설정에 지정된 Hugging Face 모델을 다운로드합니다."
    )
    parser.add_argument(
        "config",
        type=Path,
        help="설정 파일 경로 (상대경로는 현재 작업 디렉토리 기준)",
    )
    args = parser.parse_args()
    model_id, save_dir, revision = load_config(args.config)

    print(f"모델: {model_id}")
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
