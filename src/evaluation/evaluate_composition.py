# 전체 결과 평가: python src/evaluation/evaluate_composition.py
# 파일 하나 평가: python src/evaluation/evaluate_composition.py --input-jsonl data/inference_results/Qwen2.5-VL-3B-Instruct_results.jsonl
# 저장 위치: data/evaluation_results/<모델명>_composition_errors.png

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from src.common.data import read_jsonl
from src.common.paths import DEFAULT_CASES_JSONL, EVALUATION_RESULT_DIR, RESULT_DIR
from src.evaluation.evaluate_results import normalize_answer


def index_cases(cases: list[dict]) -> dict[str, dict]:
    """case_id로 원본 케이스를 찾을 수 있도록 중복 없이 연결합니다."""
    indexed = {}
    for index, case in enumerate(cases, start=1):
        case_id = case.get("case_id") if isinstance(case, dict) else None
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"{index}번째 케이스에 유효한 case_id가 없습니다.")
        if case_id in indexed:
            raise ValueError(f"원본 케이스의 case_id가 중복됩니다: {case_id}")
        indexed[case_id] = case
    if not indexed:
        raise ValueError("원본 케이스가 없습니다.")
    return indexed


def evaluate_composition(results: list[dict], cases_by_id: dict[str, dict]) -> list[dict]:
    """결과에 있는 샘플만 집계하고, 양성 이미지 수별 오류와 No 소계를 구합니다."""
    if not results:
        raise ValueError("평가할 결과가 없습니다.")

    buckets = [{"sample_count": 0, "error_count": 0} for _ in range(4)]
    seen = set()
    for index, result in enumerate(results, start=1):
        case_id = result.get("case_id") if isinstance(result, dict) else None
        if not isinstance(case_id, str) or case_id not in cases_by_id:
            raise ValueError(f"{index}번째 결과에 대응하는 원본 케이스가 없습니다: {case_id!r}")
        if case_id in seen:
            raise ValueError(f"추론 결과의 case_id가 중복됩니다: {case_id}")
        seen.add(case_id)
        case = cases_by_id[case_id]
        labels = case.get("image_labels")
        if (
            not isinstance(labels, list) or len(labels) != 3
            or any(label not in ("P", "N") for label in labels)
        ):
            raise ValueError(f"{case_id}: image_labels는 P/N 라벨 3개여야 합니다.")
        image_order = case.get("image_order")
        if (
            not isinstance(image_order, list) or len(image_order) != 3
            or result.get("image_order") != image_order
        ):
            raise ValueError(f"{case_id}: 결과와 원본 케이스의 이미지 구성이 다릅니다.")

        positive_count = labels.count("P")
        target = "Yes" if positive_count == 3 else "No"
        if normalize_answer(case.get("gt")) != target or normalize_answer(result.get("gt")) != target:
            raise ValueError(f"{case_id}: 이미지 구성과 원본/결과의 gt가 일치하지 않습니다.")
        prediction = normalize_answer(result.get("model_answer"))
        if prediction is None:
            raise ValueError(f"{case_id}: 단독 Yes/No 답변이 아닙니다: {result.get('model_answer')!r}")

        buckets[positive_count]["sample_count"] += 1
        buckets[positive_count]["error_count"] += int(prediction != target)

    rows = []
    for composition, positive_counts in (
        ("3N", (0,)),
        ("1P+2N", (1,)),
        ("2P+1N", (2,)),
        ("NO total", (0, 1, 2)),
        ("3P", (3,)),
    ):
        sample_count = sum(buckets[count]["sample_count"] for count in positive_counts)
        error_count = sum(buckets[count]["error_count"] for count in positive_counts)
        rows.append({
            "composition": composition,
            "sample_count": sample_count,
            "error_type": "False-No" if positive_counts == (3,) else "False-Yes",
            "error_count": error_count,
            "error_rate": error_count / sample_count if sample_count else None,
        })
    return rows


def save_composition_plot(input_path: Path, rows: list[dict], output_path: Path) -> None:
    """한국어 글꼴을 사용할 수 있으면 한국어로, 없으면 영어로 표를 저장합니다."""
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_name = next((
        name for name in (
            "Malgun Gothic", "AppleGothic", "NanumGothic",
            "Noto Sans CJK KR", "Noto Sans KR",
        ) if name in available_fonts
    ), None)
    headers = (
        ["이미지 구성", "샘플 수", "오류 수", "오류율"] if font_name
        else ["Image composition", "Samples", "Errors", "Error rate"]
    )
    cell_rows = []
    for row in rows:
        composition = row["composition"]
        if font_name and composition == "NO total":
            composition = "NO 전체"
        rate = "N/A" if row["error_rate"] is None else f"{row['error_rate']:.2%}"
        cell_rows.append([
            composition,
            f"{row['sample_count']:,}",
            f"{row['error_type']} {row['error_count']:,}",
            rate,
        ])

    with plt.rc_context({"font.family": font_name} if font_name else {}):
        fig, axis = plt.subplots(figsize=(11, 4.8))
        try:
            axis.set_axis_off()
            model_name = input_path.stem.removesuffix("_results")
            subtitle = "이미지 구성별 오류 분석" if font_name else "Errors by image composition"
            axis.set_title(f"{model_name}\n{subtitle}", fontsize=14, pad=16)
            table = axis.table(
                cellText=cell_rows, colLabels=headers,
                colWidths=[0.28, 0.18, 0.34, 0.20],
                cellLoc="right", colLoc="right",
                bbox=[0, 0.05, 1, 0.85],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            for (row_index, column), cell in table.get_celld().items():
                cell.visible_edges = "B"
                cell.set_edgecolor("#e1e5ea")
                cell.set_linewidth(0.6)
                cell.PAD = 0.06
                if column == 0:
                    cell.set_text_props(ha="left")
                if row_index == 0:
                    cell.set_text_props(weight="bold")
                    cell.set_edgecolor("#9aa5b1")
                elif rows[row_index - 1]["composition"] == "NO total":
                    cell.set_text_props(weight="bold")

            note = (
                "P: 대상 있음, N: 대상 없음 | 오류율 = 오류 수 / 해당 행의 샘플 수\n"
                "NO 전체는 앞의 세 행의 소계입니다. 샘플 수가 0이면 오류율은 N/A입니다."
                if font_name else
                "P: target present, N: target absent | Error rate = errors / samples in each row\n"
                "NO total is the subtotal of the first three rows. N/A means no samples."
            )
            fig.text(0.5, 0.035, note, ha="center", fontsize=9, color="#52606d")
            fig.tight_layout(rect=(0.02, 0.13, 0.98, 1))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
        finally:
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="이미지 구성별 오류 수와 오류율을 PNG 표로 저장")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model-path", type=Path, help="선택 모델의 결과만 평가 (생략하면 전체 모델)")
    source.add_argument(
        "--input-jsonl", type=Path,
        help="평가할 결과 파일 (생략하면 data/inference_results/*_results.jsonl 전체)",
    )
    parser.add_argument(
        "--cases-jsonl", type=Path, default=DEFAULT_CASES_JSONL,
        help="이미지 구성과 정답이 저장된 원본 cases JSONL",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=EVALUATION_RESULT_DIR,
        help="PNG 저장 폴더 (기본: data/evaluation_results)",
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
    cases_path = args.cases_jsonl.expanduser().resolve()
    try:
        cases_by_id = index_cases(read_jsonl(cases_path))
    except (OSError, ValueError) as error:
        parser.error(f"{cases_path}: {error}")

    output_dir = args.output_dir.expanduser().resolve()
    for input_path in input_paths:
        try:
            rows = evaluate_composition(read_jsonl(input_path), cases_by_id)
        except (OSError, ValueError) as error:
            parser.error(f"{input_path}: {error}")
        output_name = input_path.stem.removesuffix("_results") + "_composition_errors.png"
        output_path = output_dir / output_name
        save_composition_plot(input_path, rows, output_path)
        print(f"저장: {output_path}")


if __name__ == "__main__":
    main()
