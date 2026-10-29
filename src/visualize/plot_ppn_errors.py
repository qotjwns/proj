# 3B의 2P+1N False-Yes 사례: python src/visualize/plot_ppn_errors.py
# 다른 모델: python src/visualize/plot_ppn_errors.py --input-jsonl data/inference_results/Qwen2.5-VL-7B-Instruct_results.jsonl
# 저장 위치: data/evaluation_results/<모델명>_2P1N_errors/

import argparse
from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common.data import read_jsonl
from src.common.images import load_image
from src.common.paths import DEFAULT_CASES_JSONL, EVALUATION_RESULT_DIR, RESULT_DIR, project_path
from src.evaluation.evaluate_composition import evaluate_composition, index_cases
from src.evaluation.evaluate_results import normalize_answer


def select_errors(results: list[dict], cases_by_id: dict[str, dict]) -> list[dict]:
    """원본과 결과의 일치를 확인하고 2P+1N의 False-Yes만 선택합니다."""
    evaluate_composition(results, cases_by_id)
    selected = []
    for result in results:
        case = cases_by_id[result["case_id"]]
        if case["image_labels"].count("P") == 2 and normalize_answer(result["model_answer"]) == "Yes":
            if case["image_path_base"] != "project_root":
                raise ValueError(f"지원하지 않는 이미지 경로 기준입니다: {case['case_id']}")
            image_dir = project_path(case["image_dir"])
            for name in case["image_order"]:
                if not (image_dir / name).is_file():
                    raise FileNotFoundError(image_dir / name)
            selected.append({"case": case, "result": result, "number": len(selected) + 1})
    return selected


def plot_cases(records: list[dict], model_name: str, output_path: Path) -> None:
    """입력 이미지 순서를 유지하며 한 사례를 이미지 3열로 표시합니다."""
    height = 4 * len(records) + 1
    fig = plt.figure(figsize=(15, height))
    try:
        grid = fig.add_gridspec(
            2 * len(records), 3,
            height_ratios=[0.45, 3.0] * len(records),
            left=0.025, right=0.975,
            bottom=0.4 / height, top=1 - 0.65 / height,
            hspace=0.28, wspace=0.06,
        )
        fig.suptitle(
            f"{model_name} | 2P+1N False-Yes errors",
            fontsize=16, y=1 - 0.15 / height,
        )
        for row, record in enumerate(records):
            case = record["case"]
            result = record["result"]
            header = fig.add_subplot(grid[2 * row, :])
            header.set_axis_off()
            header.text(
                0, 0.95,
                f"Case {record['number']:02d} | Target: {case['class']} | "
                f"GT: {case['gt']} | Model: {result['model_answer'].strip()} | {case['group_id']}",
                ha="left", va="top", fontsize=12, weight="bold",
            )
            header.text(
                0, 0.38, f"{case['case_id']} | {case['prompt']}",
                ha="left", va="top", fontsize=10, color="#52606d",
            )

            image_dir = project_path(case["image_dir"])
            for column, (name, label) in enumerate(zip(case["image_order"], case["image_labels"])):
                axis = fig.add_subplot(grid[2 * row + 1, column])
                with load_image(image_dir / name) as image:
                    axis.imshow(image)
                color = "#16824b" if label == "P" else "#c73c3c"
                axis.set_title(f"Image {column + 1} | {label} | {name}", fontsize=10, color=color, pad=7)
                axis.set_xticks([])
                axis.set_yticks([])
                for spine in axis.spines.values():
                    spine.set_color(color)
                    spine.set_linewidth(2)

        fig.text(
            0.5, 0.12 / height,
            "P (green): target present | N (red): no target annotation | Images shown in model input order",
            ha="center", va="bottom", fontsize=10, color="#52606d",
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160, facecolor="white")
    finally:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="2P+1N에서 Yes로 잘못 답한 사례의 원본 이미지 플롯")
    parser.add_argument(
        "--input-jsonl", type=Path,
        default=RESULT_DIR / "Qwen2.5-VL-3B-Instruct_results.jsonl",
    )
    parser.add_argument("--cases-jsonl", type=Path, default=DEFAULT_CASES_JSONL)
    parser.add_argument("--output-dir", type=Path, help="모음 및 개별 PNG 저장 폴더")
    args = parser.parse_args()
    input_path = args.input_jsonl.expanduser().resolve()
    try:
        cases_by_id = index_cases(read_jsonl(args.cases_jsonl.expanduser().resolve()))
        selected = select_errors(read_jsonl(input_path), cases_by_id)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not selected:
        print("2P+1N False-Yes 사례가 없습니다.")
        return

    model_name = input_path.stem.removesuffix("_results")
    output_dir = (
        args.output_dir.expanduser().resolve() if args.output_dir
        else EVALUATION_RESULT_DIR / f"{model_name}_2P1N_errors"
    )
    overview_path = output_dir / "overview.png"
    plot_cases(selected, model_name, overview_path)
    print(f"Overview: {overview_path}", flush=True)
    for record in selected:
        category = re.sub(r"[^a-zA-Z0-9_-]+", "_", record["case"]["class"])
        output_path = output_dir / f"{record['number']:02d}_{category}.png"
        plot_cases([record], model_name, output_path)
        print(f"Case {record['number']:02d}: {record['case']['case_id']} -> {output_path}", flush=True)
    print(f"Completed: {len(selected)} cases + overview", flush=True)


if __name__ == "__main__":
    main()
