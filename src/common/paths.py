# 공통 경로 모듈. 사용: python src/experiments/a/run.py --model-path data/models/Qwen/Qwen2.5-VL-3B-Instruct

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
EXPERIMENT_DIR = DATA_DIR / "experiments"
ABC_COMPARISON_DIR = EXPERIMENT_DIR / "comparison"
GROUP_PLOT_DIR = DATA_DIR / "inference_group_plots"
DEFAULT_GROUPS_JSON = INFERENCE_POOL_DIR / "inference_groups_seed42.json"
DEFAULT_CASES_JSONL = CASE_DIR / "inference_groups_seed42_balanced3_seed42_cases.jsonl"


def result_path(experiment: str, model_path: Path, run: str = "full") -> Path:
    return EXPERIMENT_DIR / experiment / "results" / model_path.name / run / "results.jsonl"


def report_dir(experiment: str, model_path: Path, run: str = "full") -> Path:
    return EXPERIMENT_DIR / experiment / "evaluation" / model_path.name / run


def project_path(path: str | Path) -> Path:
    """데이터 목록에 기록된 상대 경로를 프로젝트 루트 기준으로 해석합니다."""
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path
