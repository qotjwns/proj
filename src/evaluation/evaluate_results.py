# 전체 결과 평가: python src/evaluation/evaluate_results.py
# 파일 하나 평가: python src/evaluation/evaluate_results.py --input-jsonl data/inference_results/Qwen2.5-VL-3B-Instruct_results.jsonl
# 저장 위치: data/evaluation_results/<모델명>_metrics.png

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.data import read_jsonl
from src.common.paths import EVALUATION_RESULT_DIR, RESULT_DIR
from src.visualize.plot_evaluation_results import save_evaluation_plot


def normalize_answer(value: object) -> str | None:
    """단독 Yes/No의 대소문자, 양끝 공백, 끝 문장부호를 정리합니다."""
    if not isinstance(value, str):
        return None
    answer = value.strip().rstrip(".!?。！？").strip().casefold()
    return {"yes": "Yes", "no": "No"}.get(answer)


def evaluate_results(results: list[dict]) -> dict:
    """생성 답변을 평가합니다. 양성은 Yes이며 분모가 0인 지표는 0입니다."""
    if not results:
        raise ValueError("평가할 결과가 없습니다.")

    confusion = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for index, result in enumerate(results, start=1):
        if not isinstance(result, dict):
            raise ValueError(f"{index}번째 결과는 JSON 객체여야 합니다.")
        prediction = normalize_answer(result.get("model_answer"))
        target = normalize_answer(result.get("gt"))
        if prediction is None or target is None:
            raise ValueError(
                f"{index}번째 결과 (case_id={result.get('case_id')!r}): "
                "model_answer와 gt는 단독 Yes/No여야 합니다. "
                f"model_answer={result.get('model_answer')!r}, gt={result.get('gt')!r}"
            )

        if target == "Yes":
            confusion["TP" if prediction == "Yes" else "FN"] += 1
        else:
            confusion["FP" if prediction == "Yes" else "TN"] += 1

    tp, tn, fp, fn = (confusion[key] for key in ("TP", "TN", "FP", "FN"))
    total = len(results)
    return {
        "total_samples": total,
        "positive_label": "Yes",
        "zero_division": 0,
        "accuracy": (tp + tn) / total,
        "confusion_matrix": confusion,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "yes_response_rate": (tp + fp) / total,
    }


def print_metrics(input_path: Path, metrics: dict, output_path: Path) -> None:
    print(f"\n{input_path.name} | 샘플 {metrics['total_samples']}개 | 양성: Yes")
    for key, label in (
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("yes_response_rate", "Yes response rate"),
    ):
        print(f"{label}: {metrics[key]:.2%}")
    confusion = metrics["confusion_matrix"]
    print("혼동행렬 (행: 정답, 열: 예측)")
    print(f"{'GT / Pred':>10} {'Yes':>8} {'No':>8}")
    print(f"{'Yes':>10} {confusion['TP']:>8} {confusion['FN']:>8}")
    print(f"{'No':>10} {confusion['FP']:>8} {confusion['TN']:>8}")
    print(f"저장: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="추론 결과의 기본 Yes/No 성능 평가 및 PNG 저장")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model-path", type=Path, help="선택 모델의 결과만 평가 (생략하면 전체 모델)")
    source.add_argument(
        "--input-jsonl", type=Path,
        help="평가할 결과 파일 (생략하면 data/inference_results/*_results.jsonl 전체)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=EVALUATION_RESULT_DIR,
        help="지표 PNG 저장 폴더 (기본: data/evaluation_results)",
    )
    args = parser.parse_args()

    if args.input_jsonl:
        input_paths = [args.input_jsonl.expanduser().resolve()]
    elif args.model_path:
        model_name = args.model_path.expanduser().resolve().name
        input_paths = [RESULT_DIR / f"{model_name}_results.jsonl"]
    else:
        input_paths = sorted(RESULT_DIR.glob("*_results.jsonl"))
    if not input_paths:
        parser.error(f"평가할 *_results.jsonl 파일이 없습니다: {RESULT_DIR}")
    output_dir = args.output_dir.expanduser().resolve()

    for input_path in input_paths:
        try:
            metrics = evaluate_results(read_jsonl(input_path))
        except (OSError, ValueError) as error:
            parser.error(f"{input_path}: {error}")
        output_name = input_path.stem.removesuffix("_results") + "_metrics.png"
        output_path = output_dir / output_name
        save_evaluation_plot(input_path, metrics, output_path)
        print_metrics(input_path, metrics, output_path)


if __name__ == "__main__":
    main()
