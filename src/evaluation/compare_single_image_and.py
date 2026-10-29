# 단일 이미지 예측 3개를 AND로 결합하고 기존 3장 입력의 예측과 비교합니다.
# 실행: python src/evaluation/compare_single_image_and.py
# 저장: data/evaluation_results/<모델명>_<오류종류>_and_comparison.png / .csv

import argparse
import csv
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from src.common.data import read_jsonl
from src.common.paths import EVALUATION_RESULT_DIR, SINGLE_IMAGE_RESULT_DIR
from src.evaluation.evaluate_results import evaluate_results, normalize_answer
from src.evaluation.evaluate_single_image_results import draw_table, summarize_results


def combine_predictions(answers: list[str]) -> str:
    """세 예측이 모두 Yes일 때만 Yes를 반환합니다. 정답 라벨은 사용하지 않습니다."""
    if len(answers) != 3:
        raise ValueError("AND 계산에는 예측 3개가 필요합니다.")
    normalized = [normalize_answer(answer) for answer in answers]
    if any(answer is None for answer in normalized):
        raise ValueError("AND 계산에는 Yes/No로 해석 가능한 예측만 사용할 수 있습니다.")
    return "Yes" if all(answer == "Yes" for answer in normalized) else "No"


def compare_results(records: list[dict]) -> tuple[list[dict], dict]:
    """동일한 원본 케이스에서 기존 예측과 단일 예측 AND를 비교합니다."""
    # 누락·중복 위치, 정답, 원본 결과 및 모델의 일관성을 먼저 확인합니다.
    summary = summarize_results(records)
    grouped = {}
    for record in records:
        grouped.setdefault(record["case_id"], {})[record["image_position"]] = record

    comparisons = []
    for case_id, positions in grouped.items():
        rows = [positions[position] for position in (1, 2, 3)]
        first = rows[0]
        source = first["source_result"]
        answers = [normalize_answer(row["model_answer"]) for row in rows]
        prediction = combine_predictions(answers)
        original = normalize_answer(source["model_answer"])
        target = normalize_answer(source["gt"])
        comparisons.append({
            "case_id": case_id,
            "class": first["class"],
            "composition": "".join(row["image_label"] for row in rows),
            "source_error_type": first["source_error_type"],
            **{f"image_{index}": row["image_order"][0] for index, row in enumerate(rows, start=1)},
            **{f"single_prediction_{index}": answer for index, answer in enumerate(answers, start=1)},
            "gt": target,
            "original_prediction": original,
            "and_prediction": prediction,
            "original_correct": original == target,
            "and_correct": prediction == target,
            "prediction_changed": original != prediction,
        })

    original_metrics = evaluate_results([
        {"gt": row["gt"], "model_answer": row["original_prediction"]} for row in comparisons
    ])
    and_metrics = evaluate_results([
        {"gt": row["gt"], "model_answer": row["and_prediction"]} for row in comparisons
    ])
    error_rows = []
    for error_type in ("false-yes", "false-no", "total"):
        selected = [row for row in comparisons if error_type == "total" or row["source_error_type"] == error_type]
        error_rows.append({
            "source": error_type,
            "count": len(selected),
            "original_correct": sum(row["original_correct"] for row in selected),
            "and_correct": sum(row["and_correct"] for row in selected),
        })
    return comparisons, {
        "model_name": summary["model_name"],
        "case_count": len(comparisons),
        "original_metrics": original_metrics,
        "and_metrics": and_metrics,
        "error_rows": error_rows,
    }


def save_comparison_plot(report: dict, output_path: Path) -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in (
        "Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR",
    ) if name in available), None)

    def text(korean: str, english: str) -> str:
        return korean if font else english

    with plt.rc_context({"font.family": font} if font else {}):
        fig, axes = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={"height_ratios": [1.25, 1]})
        try:
            fig.suptitle(
                f"{report['model_name']} | Joint inference vs. Single-image AND\n"
                + text(f"동일한 원래 오답 케이스 {report['case_count']}개 비교", f"Same {report['case_count']} originally incorrect cases"),
                fontsize=16,
            )
            metric_rows = []
            for label, key in (
                (text("기존: 3장 동시 입력", "Original: three images together"), "original_metrics"),
                (text("개별 예측 3개 + Python AND", "Single-image predictions + AND"), "and_metrics"),
            ):
                metrics = report[key]
                metric_rows.append([label, str(metrics["total_samples"])] + [
                    f"{metrics[name]:.2%}" for name in ("accuracy", "precision", "recall", "f1", "yes_response_rate")
                ])
            draw_table(
                axes[0], text("1. 케이스별 최종 예측 성능 (양성: Yes)", "1. Final case prediction metrics (positive: Yes)"),
                [text("예측 방법", "Method"), text("케이스 수", "Cases"), "Accuracy", "Precision", "Recall", "F1", "Yes rate"],
                metric_rows, [0.28, 0.08, 0.128, 0.128, 0.128, 0.128, 0.128], bottom=0.29,
            )
            matrices = []
            for label, key in ((text("기존", "Original"), "original_metrics"), ("AND", "and_metrics")):
                counts = report[key]["confusion_matrix"]
                matrices.append(label + ": " + " / ".join(f"{name} {counts[name]}" for name in ("TP", "TN", "FP", "FN")))
            axes[0].text(0, 0.05, "\n".join(matrices), transform=axes[0].transAxes, fontsize=10, color="#52606d")

            error_rows = []
            for row in report["error_rows"]:
                label = {"false-yes": "False-Yes", "false-no": "False-No", "total": text("전체", "Total")}[row["source"]]
                accuracy = f"{row['and_correct'] / row['count']:.2%}" if row["count"] else "N/A"
                error_rows.append([
                    label, str(row["count"]), str(row["original_correct"]),
                    str(row["and_correct"]), str(row["count"] - row["and_correct"]), accuracy,
                ])
            draw_table(
                axes[1], text("2. 원래 오류 유형별 AND 결과", "2. AND results by original error type"),
                [text("원래 오류", "Original error"), text("케이스 수", "Cases"), text("기존 정답", "Original correct"), text("AND 정답", "AND correct"), text("AND 오답", "AND incorrect"), text("AND 정확도", "AND accuracy")],
                error_rows, [0.22, 0.15, 0.15, 0.15, 0.15, 0.18],
            )
            fig.text(
                0.5, 0.035,
                text(
                    "AND 규칙: 예측이 Yes / Yes / Yes일 때만 Yes, 하나라도 No이면 No. 정답은 결합 규칙에 사용하지 않습니다.\n"
                    "기존 오답 케이스만 선택했으므로 이 범위의 기존 정확도는 0%입니다. 전체 데이터셋의 AND 정확도는 아닙니다.\n"
                    "저장된 단일 이미지 답변을 결합한 결과이며 모델 추가 추론은 없습니다.",
                    "AND: Yes only for Yes / Yes / Yes; otherwise No. Ground truth is not used in the combination.\n"
                    "Original accuracy is 0% because only originally incorrect cases are selected. This is not full-dataset AND accuracy.\n"
                    "Predictions come from stored single-image answers; no additional model inference.",
                ), ha="center", fontsize=10, color="#52606d",
            )
            fig.tight_layout(rect=(0.02, 0.16, 0.98, 0.90), h_pad=2.5)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
        finally:
            plt.close(fig)


def save_case_csv(comparisons: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)


def main() -> None:
    parser = argparse.ArgumentParser(description="단일 이미지 예측 AND와 기존 3장 입력 결과의 PNG·CSV 비교")
    parser.add_argument("--input-jsonl", type=Path, help="생략하면 data/single_image_results/*_single_image_results.jsonl 전체")
    parser.add_argument("--output-dir", type=Path, default=EVALUATION_RESULT_DIR)
    args = parser.parse_args()
    inputs = [args.input_jsonl.expanduser().resolve()] if args.input_jsonl else sorted(SINGLE_IMAGE_RESULT_DIR.glob("*_single_image_results.jsonl"))
    if not inputs:
        parser.error("단일 이미지 추론 결과가 없습니다.")
    output_dir = args.output_dir.expanduser().resolve()
    for input_path in inputs:
        try:
            comparisons, report = compare_results(read_jsonl(input_path))
        except (OSError, ValueError) as error:
            parser.error(f"{input_path}: {error}")
        stem = input_path.stem.removesuffix("_single_image_results") + "_and_comparison"
        save_comparison_plot(report, output_dir / f"{stem}.png")
        save_case_csv(comparisons, output_dir / f"{stem}.csv")
        print(f"PNG: {output_dir / (stem + '.png')}")
        print(f"CSV: {output_dir / (stem + '.csv')}")
        print(f"Original: {report['original_metrics']}")
        print(f"AND: {report['and_metrics']}")
        print(f"By original error: {report['error_rows']}")


if __name__ == "__main__":
    main()
