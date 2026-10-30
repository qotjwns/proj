# 전체 비교: python src/visualize/compare.py --model-path data/models/Qwen/Qwen2.5-VL-3B-Instruct
# A/B만 비교: python src/visualize/compare.py --experiments a b

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.data import read_jsonl, write_json
from src.common.paths import ABC_COMPARISON_DIR, DEFAULT_CASES_JSONL, DEFAULT_MODEL_PATH, result_path
from src.common.results import align_results, read_results
from src.visualize.metrics import summarize
from src.visualize.reporting import percentage, save_table, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="같은 케이스의 A/B/C 결과 비교. 모델 추론 없음")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--experiments", nargs="+", choices=("a", "b", "c"), default=["a", "b", "c"])
    parser.add_argument("--cases-jsonl", type=Path, default=DEFAULT_CASES_JSONL)
    parser.add_argument("--output-dir", type=Path)
    for experiment in ("a", "b", "c"):
        parser.add_argument(f"--{experiment}-jsonl", type=Path)
    args = parser.parse_args()
    model_path = args.model_path.expanduser().resolve()
    cases = read_jsonl(args.cases_jsonl)
    experiments = list(dict.fromkeys(args.experiments))
    paths = {
        experiment: getattr(args, f"{experiment}_jsonl") or result_path(experiment, model_path)
        for experiment in experiments
    }
    results = {
        experiment: align_results(cases, read_results(path), complete=True)
        for experiment, path in paths.items()
    }
    for experiment, rows in results.items():
        if any(row["experiment"] != experiment or Path(row["model_path"]).name != model_path.name for row in rows):
            raise ValueError(f"{experiment}: 선택한 실험·모델과 저장된 결과가 다릅니다.")
    reports = [{"experiment": experiment, **summarize(rows)} for experiment, rows in results.items()]
    output_dir = args.output_dir or ABC_COMPARISON_DIR / model_path.name / "full"
    write_json({"sources": {key: str(path) for key, path in paths.items()}, "metrics": reports}, output_dir / "comparison.json")
    write_csv(reports, output_dir / "comparison.csv")
    case_rows = []
    for index, case in enumerate(cases):
        row = {"case_id": case["case_id"], "gt": case["gt"]}
        for experiment in experiments:
            prediction = results[experiment][index]["prediction"]
            row[f"{experiment}_prediction"] = prediction
            row[f"{experiment}_correct"] = prediction == case["gt"]
        case_rows.append(row)
    write_csv(case_rows, output_dir / "cases.csv")
    cells = [[
        row["experiment"].upper(), row["cases"], percentage(row["accuracy"]),
        percentage(row["image_accuracy"]), percentage(row["all_images_correct_rate"]),
        row["false_yes"], row["false_no"], row["invalid_answers"],
    ] for row in reports]
    note = (
        "Same full set of cases. Invalid answers count as failures. "
        "Prompts differ, so this does not isolate a single cause."
    )
    save_table(
        ["Experiment", "Cases", "Accuracy", "Image acc.", "All 3 correct", "False-Yes", "False-No", "Invalid"],
        cells, model_path.name, output_dir / "comparison.png", note,
    )
    print(f"저장: {output_dir}")


if __name__ == "__main__":
    main()
