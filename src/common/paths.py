"""프로젝트의 데이터·모델 경로를 한곳에서 정의합니다."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
IMAGE_DIR = DATA_DIR / "coco" / "val2017"
ANNOTATION_PATH = DATA_DIR / "coco_rem" / "instances_valrem.json"
MODEL_DIR = DATA_DIR / "models"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_MODEL_PATH = MODEL_DIR / DEFAULT_MODEL
POOL_DIR = DATA_DIR / "pools"
INFERENCE_POOL_DIR = DATA_DIR / "inference_pools"
CASE_DIR = DATA_DIR / "inference_cases"
RESULT_DIR = DATA_DIR / "inference_results"
SINGLE_IMAGE_RESULT_DIR = DATA_DIR / "single_image_results"
SINGLE_IMAGE_AND_DIR = DATA_DIR / "single_image_and"
EVALUATION_RESULT_DIR = DATA_DIR / "evaluation_results"
GROUP_PLOT_DIR = DATA_DIR / "inference_group_plots"
DEFAULT_GROUPS_JSON = INFERENCE_POOL_DIR / "inference_groups_seed42.json"
DEFAULT_CASES_JSONL = CASE_DIR / "inference_groups_seed42_balanced3_seed42_cases.jsonl"


def project_path(path: str | Path) -> Path:
    """데이터 목록에 기록된 상대 경로를 프로젝트 루트 기준으로 해석합니다."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
