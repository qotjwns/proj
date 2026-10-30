# 전체: python src/experiments/c/run.py --model-path data/models/Qwen/Qwen2.5-VL-3B-Instruct
# 20개 확인: python src/experiments/c/run.py --limit 20

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.common.experiment import make_parser, run


def main() -> None:
    args = make_parser("c").parse_args()
    run("c", args)


if __name__ == "__main__":
    main()
