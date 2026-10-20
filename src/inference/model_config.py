"""모델 이름으로 공통 설정을 선택하는 명령행 인터페이스."""
import argparse
import json
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_PATH = PROJECT_ROOT / "configs/model.json"


def load_model_configs() -> dict[str, dict[str, str]]:
    models = json.loads(MODEL_CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(models, dict) or not models:
        raise ValueError(f"{MODEL_CONFIG_PATH}: 모델 이름별 설정이 필요합니다.")
    for name, config in models.items():
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) or not isinstance(config, dict):
            raise ValueError(f"{MODEL_CONFIG_PATH}: 잘못된 모델 설정: {name!r}")
        for key in ("model_id", "save_dir", "revision"):
            if not isinstance(config.get(key), str) or not config[key].strip():
                raise ValueError(f"모델 {name!r}: {key}에 비어 있지 않은 문자열이 필요합니다.")
    return models


def parse_model_config(description: str) -> tuple[str, dict[str, str], Path]:
    parser = argparse.ArgumentParser(description=description)
    try:
        models = load_model_configs()
    except (OSError, ValueError) as error:
        parser.error(str(error))
    parser.add_argument(
        "model",
        choices=models,
        help="configs/model.json에 등록된 모델 이름",
    )
    parser.add_argument(
        "--list-models",
        action="version",
        version="\n".join(f"{name}: {config['model_id']}" for name, config in models.items()),
        help="등록된 모델 목록을 출력하고 종료",
    )
    args = parser.parse_args()
    config = models[args.model]
    model_path = (PROJECT_ROOT / Path(config["save_dir"]).expanduser()).resolve()
    return args.model, config, model_path


def get_log_dir(model_name: str, config: dict[str, str]) -> Path:
    log_root = PROJECT_ROOT / "logs"
    log_dir = log_root / model_name
    # 기존 공용 로그는 설정이 일치하는 모델만 이어서 사용합니다.
    legacy_meta = log_root / "baseline_e1_meta.json"
    if not log_dir.exists() and legacy_meta.exists():
        metadata = json.loads(legacy_meta.read_text(encoding="utf-8"))
        if metadata.get("model") == config:
            return log_root
    return log_dir
