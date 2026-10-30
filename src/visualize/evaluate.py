# 실행: python src/visualize/evaluate.py --experiment a --model-path data/models/Qwen/Qwen2.5-VL-3B-Instruct
# 파일 지정: python src/visualize/evaluate.py --experiment b --input-jsonl path/to/results.jsonl
# 20개 확인 결과: python src/visualize/evaluate.py --experiment c --run pilot_20

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.data import read_jsonl, write_json
from src.common.paths import DEFAULT_CASES_JSONL, DEFAULT_MODEL_PATH, report_dir, result_path
from src.common.results import align_results, read_results
from src.visualize.metrics import by_composition, summarize
from src.visualize.reporting import percentage, save_table, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="저장된 JSONL의 지표·구성별 오류를 JSON/CSV/PNG로 저장")
    parser.add_argument("--experiment", choices=("a", "b", "c"), required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--run", default="full", help="full, pilot_20 등 결과 폴더명")
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--cases-jsonl", type=Path, default=DEFAULT_CASES_JSONL)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    model_path = args.model_path.expanduser().resolve()
    source = args.input_jsonl.expanduser().resolve() if args.input_jsonl else result_path(args.experiment, model_path, args.run)
    rows = align_results(read_jsonl(args.cases_jsonl), read_results(source))
    if any(row["experiment"] != args.experiment or Path(row["model_path"]).name != model_path.name for row in rows):
        raise ValueError("선택한 실험·모델과 입력 결과가 다릅니다.")
    output_dir = args.output_dir or report_dir(args.experiment, model_path, source.parent.name)
    metrics = summarize(rows)
    composition = by_composition(rows)
    write_json({
        "source": str(source), "experiment": args.experiment,
        "model_path": str(model_path), **metrics,
    }, output_dir / "metrics.json")
    write_json(composition, output_dir / "composition.json")
    write_csv(composition, output_dir / "composition.csv")
    metric_rows = [[key, percentage(metrics[key])] for key in (
        "accuracy", "precision", "recall", "f1", "yes_response_rate",
        "valid_answer_rate", "image_accuracy", "all_images_correct_rate",
    )]
    if metrics["image_accuracy_by_position"] is not None:
        metric_rows += [
            [f"image_{index}_accuracy", percentage(value)]
            for index, value in enumerate(metrics["image_accuracy_by_position"], start=1)
        ]
    metric_rows += [[key, str(metrics[key])] for key in (
        "cases", "true_yes", "true_no", "false_yes", "false_no", "invalid_answers",
    )]
    title = f"{model_path.name} | Experiment {args.experiment.upper()} | {len(rows)} cases"
    save_table(
        ["Metric", "Value"], metric_rows, title, output_dir / "metrics.png",
        "Invalid answers remain in the accuracy denominator. Image metrics count case-image occurrences.",
    )
    composition_rows = [[
        row["composition"], row["cases"], row["errors"], percentage(row["error_rate"]),
        row["false_yes"], row["false_no"], row["invalid_answers"],
    ] for row in composition]
    save_table(
        ["Composition", "Cases", "Errors", "Error rate", "False-Yes", "False-No", "Invalid"],
        composition_rows, title, output_dir / "composition.png",
    )
    print(f"Accuracy: {metrics['accuracy']:.2%} | False-Yes: {metrics['false_yes']} | False-No: {metrics['false_no']}")
    print(f"저장: {output_dir}")


if __name__ == "__main__":
    main()
