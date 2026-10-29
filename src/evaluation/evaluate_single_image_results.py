# 전체 단일 이미지 결과: python src/evaluation/evaluate_single_image_results.py
# 파일 하나: python src/evaluation/evaluate_single_image_results.py --input-jsonl data/single_image_results/Qwen2.5-VL-3B-Instruct_all_single_image_results.jsonl
# 계산과 PNG 표 생성을 이 파일에서 함께 수행합니다.

import argparse
from collections import defaultdict
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


def summarize_results(records: list[dict]) -> dict:
    """이미지 출현 횟수·고유 질문 기준 지표와 원본 케이스별 회복 여부를 계산합니다."""
    if not records:
        raise ValueError("평가할 단일 이미지 결과가 없습니다.")
    cases = defaultdict(dict)
    unique = {}
    models = set()
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"{index}번째 결과는 JSON 객체여야 합니다.")
        case_id = record.get("case_id")
        position = record.get("image_position")
        source = record.get("source_result")
        if not isinstance(case_id, str) or not case_id or type(position) is not int or position not in (1, 2, 3):
            raise ValueError(f"{index}번째 결과의 case_id 또는 image_position이 잘못되었습니다.")
        if position in cases[case_id]:
            raise ValueError(f"중복된 케이스·이미지 위치입니다: {case_id}, {position}")
        if not isinstance(source, dict) or source.get("case_id") != case_id:
            raise ValueError(f"{case_id}: source_result가 원본 케이스와 일치하지 않습니다.")
        source_order = source.get("image_order")
        if (
            not isinstance(source_order, list) or len(source_order) != 3
            or record.get("image_order") != [source_order[position - 1]]
        ):
            raise ValueError(f"{case_id}: 이미지 위치와 원본 파일명이 일치하지 않습니다.")
        error_type = record.get("source_error_type")
        expected_source = {"false-yes": ("No", "Yes"), "false-no": ("Yes", "No")}
        if error_type not in expected_source or (
            normalize_answer(source.get("gt")), normalize_answer(source.get("model_answer"))
        ) != expected_source[error_type]:
            raise ValueError(f"{case_id}: 원본의 오류 유형·정답·답변이 일치하지 않습니다.")
        target = normalize_answer(record.get("gt"))
        answer = normalize_answer(record.get("model_answer"))
        if record.get("image_label") not in ("P", "N") or target != (
            "Yes" if record["image_label"] == "P" else "No"
        ):
            raise ValueError(f"{case_id}, 위치 {position}: 이미지 라벨과 정답이 다릅니다.")
        if answer is None:
            raise ValueError(f"{case_id}, 위치 {position}: 단독 Yes/No 답변으로 해석할 수 없습니다.")
        for field in ("model_path", "image_path", "prompt"):
            if not isinstance(record.get(field), str) or not record[field]:
                raise ValueError(f"{case_id}: {field}가 없습니다.")
        models.add(record["model_path"])
        key = (record["model_path"], Path(record["image_path"]), record["prompt"])
        if key in unique and (
            normalize_answer(unique[key]["gt"]), normalize_answer(unique[key]["model_answer"])
        ) != (target, answer):
            raise ValueError(f"같은 이미지·질문의 정답 또는 예측이 다릅니다: {record['image_path']}")
        unique.setdefault(key, record)
        cases[case_id][position] = record

    if len(models) != 1:
        raise ValueError("한 결과 파일에는 같은 모델의 결과만 들어 있어야 합니다.")
    for case_id, positions in cases.items():
        if set(positions) != {1, 2, 3}:
            raise ValueError(f"{case_id}: 이미지 3장의 결과가 모두 있어야 합니다.")
        first = positions[1]
        if any(
            row["source_result"] != first["source_result"]
            or row["source_error_type"] != first["source_error_type"]
            for row in positions.values()
        ):
            raise ValueError(f"{case_id}: 이미지별 원본 결과 정보가 다릅니다.")
        combined_gt = "Yes" if all(row["image_label"] == "P" for row in positions.values()) else "No"
        if combined_gt != normalize_answer(first["source_result"]["gt"]):
            raise ValueError(f"{case_id}: 단일 이미지 정답과 원본 케이스 정답이 다릅니다.")

    def correct(row: dict) -> bool:
        return normalize_answer(row["model_answer"]) == normalize_answer(row["gt"])

    image_rows = []
    for error_type in ("false-yes", "false-no"):
        for label in ("P", "N"):
            selected = [row for row in records if row["source_error_type"] == error_type and row["image_label"] == label]
            if selected:
                image_rows.append({
                    "source": error_type, "label": label, "count": len(selected),
                    "correct": sum(correct(row) for row in selected),
                })
    image_rows.append({"source": "total", "label": "P + N", "count": len(records), "correct": sum(correct(row) for row in records)})

    case_rows = []
    for error_type in ("false-yes", "false-no", "total"):
        selected = [rows for rows in cases.values() if error_type == "total" or rows[1]["source_error_type"] == error_type]
        case_rows.append({
            "source": error_type, "count": len(selected),
            "all_correct": sum(all(correct(row) for row in rows.values()) for rows in selected),
        })
    return {
        "model_name": Path(next(iter(models))).name,
        "case_count": len(cases),
        "image_metrics": evaluate_results(records),
        "unique_metrics": evaluate_results(list(unique.values())),
        "image_rows": image_rows,
        "case_rows": case_rows,
    }


def draw_table(axis, title: str, headers: list[str], rows: list[list[str]], widths: list[float], bottom: float = 0.02) -> None:
    axis.set_axis_off()
    axis.set_title(title, loc="left", fontsize=14, pad=12, weight="bold")
    table = axis.table(
        cellText=rows, colLabels=headers, colWidths=widths,
        cellLoc="right", colLoc="right", bbox=[0, bottom, 1, 0.91 - bottom],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#dbe3ed")
        cell.PAD = 0.04
        if column == 0:
            cell.set_text_props(ha="left")
        if row == 0:
            cell.set_facecolor("#24486c")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#f0f5fa" if row % 2 else "white")


def save_report(report: dict, output_path: Path) -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR") if name in available), None)
    def text(korean: str, english: str) -> str:
        return korean if font else english
    def source_name(value: str) -> str:
        return {"false-yes": "False-Yes", "false-no": "False-No", "total": text("전체", "Total")}[value]
    def rate(count: int, total: int) -> str:
        return f"{count / total:.2%}" if total else "N/A"

    with plt.rc_context({"font.family": font} if font else {}):
        fig, axes = plt.subplots(3, 1, figsize=(15, 12), gridspec_kw={"height_ratios": [1.2, 1.5, 1.2]})
        try:
            fig.suptitle(
                f"{report['model_name']} | Single-image evaluation\n"
                + text(f"원래 3장 입력의 오류 케이스 {report['case_count']}개 대상", f"Original three-image error cases only: {report['case_count']}"),
                fontsize=17,
            )
            metrics_rows = []
            for label, metrics in (
                (text("케이스 내 이미지 (중복 포함)", "Case-image occurrences"), report["image_metrics"]),
                (text("고유 이미지·질문", "Unique image/question pairs"), report["unique_metrics"]),
            ):
                metrics_rows.append([label, str(metrics["total_samples"])] + [
                    f"{metrics[key]:.2%}" for key in ("accuracy", "precision", "recall", "f1", "yes_response_rate")
                ])
            draw_table(
                axes[0], text("1. 개별 이미지 기본 성능 (양성: Yes)", "1. Single-image metrics (positive: Yes)"),
                [text("집계 단위", "Counting unit"), text("샘플 수", "Samples"), "Accuracy", "Precision", "Recall", "F1", "Yes rate"],
                metrics_rows, [0.28, 0.08, 0.128, 0.128, 0.128, 0.128, 0.128], bottom=0.28,
            )
            matrices = []
            for label, key in ((text("중복 포함", "Occurrences"), "image_metrics"), (text("중복 제거", "Unique pairs"), "unique_metrics")):
                counts = report[key]["confusion_matrix"]
                matrices.append(label + ": " + " / ".join(f"{name} {counts[name]}" for name in ("TP", "TN", "FP", "FN")))
            axes[0].text(0, 0.03, "\n".join(matrices), transform=axes[0].transAxes, fontsize=10, color="#52606d")

            image_rows = [[
                source_name(row["source"]), row["label"], str(row["count"]), str(row["correct"]),
                str(row["count"] - row["correct"]), rate(row["count"] - row["correct"], row["count"]),
            ] for row in report["image_rows"]]
            draw_table(
                axes[1], text("2. 원래 오류 유형별 단일 이미지 결과 (중복 포함)", "2. Single-image results by original error type (occurrences)"),
                [text("원래 오류", "Original error"), text("이미지 라벨", "Image label"), text("샘플 수", "Samples"), text("정답 수", "Correct"), text("오답 수", "Incorrect"), text("오류율", "Error rate")],
                image_rows, [0.22, 0.18, 0.15, 0.15, 0.15, 0.15],
            )
            case_rows = [[
                source_name(row["source"]), str(row["count"]), str(row["all_correct"]),
                str(row["count"] - row["all_correct"]), rate(row["all_correct"], row["count"]),
            ] for row in report["case_rows"]]
            draw_table(
                axes[2], text("3. 원본 케이스 기준: 개별 입력 시 3장 모두 맞혔는가?", "3. Original cases: were all three images correct individually?"),
                [text("원래 오류", "Original error"), text("케이스 수", "Cases"), text("3장 모두 정답", "All 3 correct"), text("1장 이상 오답", "At least 1 incorrect"), text("모두 정답 비율", "All-correct rate")],
                case_rows, [0.22, 0.15, 0.21, 0.22, 0.20],
            )
            fig.text(
                0.5, 0.035,
                text(
                    "중복 포함: 원본 케이스의 각 이미지 위치를 한 번씩 집계. 고유 이미지·질문: 같은 모델·이미지·질문을 한 번만 집계.\n"
                    "P 정답은 Yes, N 정답은 No. 3장 모두 정답은 개별 입력 결과 기준이며, 3장 동시 입력을 재평가한 값이 아닙니다.\n"
                    "원래 오답이었던 케이스만 평가했으므로 전체 데이터셋의 단일 이미지 성능과 구분해야 합니다.",
                    "Occurrences count each original case/image position. Unique pairs count each model/image/question once.\n"
                    "P expects Yes; N expects No. All-correct cases use separate single-image answers, not another joint inference.\n"
                    "Only original error cases are included; these metrics do not represent the full dataset.",
                ), ha="center", fontsize=10, color="#52606d",
            )
            fig.tight_layout(rect=(0.02, 0.13, 0.98, 0.92), h_pad=2.5)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
        finally:
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="단일 이미지 결과의 기본 지표·오류 유형·케이스별 PNG 표 생성")
    parser.add_argument("--input-jsonl", type=Path, help="생략하면 data/single_image_results/*_single_image_results.jsonl 전체")
    parser.add_argument("--output-dir", type=Path, default=EVALUATION_RESULT_DIR)
    args = parser.parse_args()
    inputs = [args.input_jsonl.expanduser().resolve()] if args.input_jsonl else sorted(SINGLE_IMAGE_RESULT_DIR.glob("*_single_image_results.jsonl"))
    if not inputs:
        parser.error("평가할 단일 이미지 결과 파일이 없습니다.")
    for input_path in inputs:
        try:
            report = summarize_results(read_jsonl(input_path))
        except (OSError, ValueError) as error:
            parser.error(f"{input_path}: {error}")
        name = input_path.stem.removesuffix("_single_image_results") + "_single_image_summary.png"
        output_path = args.output_dir.expanduser().resolve() / name
        save_report(report, output_path)
        print(f"Saved: {output_path}")
        for key in ("image_metrics", "unique_metrics", "image_rows", "case_rows"):
            print(f"{key}: {report[key]}")


if __name__ == "__main__":
    main()
