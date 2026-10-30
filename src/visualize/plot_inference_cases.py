# 전체 그룹: python src/visualize/plot_inference_cases.py
# 그룹 하나: python src/visualize/plot_inference_cases.py --group group_01_cat
# 클래스·조건 지정: python src/visualize/plot_inference_cases.py --category cat --condition B
# 앞의 2그룹만 확인: python src/visualize/plot_inference_cases.py --limit 2
# 저장 위치: data/inference_group_plots/<그룹명>.png

import argparse
from pathlib import Path
import textwrap

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.common.data import normalize_category, read_jsonl
from src.common.images import load_image
from src.common.paths import DEFAULT_CASES_JSONL, GROUP_PLOT_DIR, project_path


DEFAULT_INPUT_JSONL = DEFAULT_CASES_JSONL
DEFAULT_OUTPUT_DIR = GROUP_PLOT_DIR


def select_cases(
    cases: list[dict], group_id: str | None = None, category: str | None = None,
    condition: str | None = None, gt: str | None = None,
) -> list[dict]:
    """샘플 순서를 유지하면서 그룹·클래스·조건·정답 필터를 적용합니다."""
    selected = cases
    if group_id:
        selected = [case for case in selected if case["group_id"] == group_id]
    if category:
        category_key = normalize_category(category)
        selected = [
            case for case in selected
            if normalize_category(case["class"]) == category_key
        ]
    if condition:
        selected = [case for case in selected if case["condition_code"] == condition]
    if gt:
        selected = [case for case in selected if case["gt"] == gt]
    return selected


def group_cases(cases: list[dict]) -> dict[str, list[dict]]:
    """JSONL 순서대로 같은 그룹의 샘플을 모읍니다."""
    groups = {}
    for case in cases:
        groups.setdefault(case["group_id"], []).append(case)
    return groups


def plot_group(
    group_id: str, cases: list[dict], image_dir: Path | None, output_dir: Path,
    output_suffix: str = "",
) -> Path:
    """그룹의 각 샘플을 한 행에 표시하고, 그룹당 PNG 한 장으로 저장합니다."""
    row_count = len(cases)
    figure_height = 4 * row_count + 2
    fig, axes = plt.subplots(row_count, 3, figsize=(15, figure_height), squeeze=False)
    try:
        yes_count = sum(case["gt"] == "Yes" for case in cases)
        fig.suptitle(
            f"{group_id} | class: {cases[0]['class']}\n"
            f"{row_count} samples | Yes: {yes_count} | No: {row_count - yes_count}",
            fontsize=15,
            y=1 - 0.15 / figure_height,
        )
        for row_index, case in enumerate(cases):
            case_image_dir = image_dir or project_path(case["image_dir"])
            row_title = (
                f"Sample {row_index + 1} | {case['condition_code']} ({case['composition']})"
                f" | GT: {case['gt']}"
            )
            for index, (axis, name, label) in enumerate(
                zip(axes[row_index], case["image_order"], case["image_labels"]), start=1
            ):
                color = "forestgreen" if label == "P" else "firebrick"
                label_text = "POS" if label == "P" else "NEG candidate"
                axis.set_axis_off()
                axis.set_title(
                    f"{row_title}\nImage {index} | {label_text}\n{name}",
                    fontsize=10, color=color,
                )
                with load_image(case_image_dir / name) as image:
                    axis.imshow(image)

        fig.text(
            0.5, 0.35 / figure_height, textwrap.fill(cases[0]["prompt"], width=110),
            ha="center", va="center", fontsize=11,
        )
        fig.tight_layout(rect=(0, 0.8 / figure_height, 1, 1 - 0.9 / figure_height), h_pad=3)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{group_id}{output_suffix}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return output_path
    finally:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="balanced3 평가 샘플을 그룹별 PNG로 저장 (샘플당 1행·이미지 3열)")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--image-dir", type=Path, help="이미지 폴더 직접 지정 (기본: JSONL의 image_dir)")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="저장 폴더 (기본: data/inference_group_plots/)")
    parser.add_argument("--group", help="그룹 하나만 표시 (예: group_01_cat)")
    parser.add_argument("--category", help="해당 클래스만 표시 (예: cat)")
    parser.add_argument("--condition", choices=("A", "B", "C", "D"), help="이미지 구성 조건으로 필터")
    parser.add_argument("--gt", choices=("Yes", "No"), help="정답 라벨로 필터")
    parser.add_argument("--limit", type=int, help="필터 적용 후 앞에서부터 저장할 그룹 수")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit는 1 이상이어야 합니다.")

    input_path = args.input_jsonl.expanduser().resolve()
    image_dir = args.image_dir.expanduser().resolve() if args.image_dir else None
    output_dir = args.output_dir.expanduser().resolve()
    selected = select_cases(
        read_jsonl(input_path), group_id=args.group, category=args.category,
        condition=args.condition, gt=args.gt,
    )
    groups = list(group_cases(selected).items())
    if args.limit is not None:
        groups = groups[:args.limit]
    output_suffix = ""
    if args.condition:
        output_suffix += f"__condition_{args.condition}"
    if args.gt:
        output_suffix += f"__gt_{args.gt}"
    for group_id, cases in groups:
        output_path = plot_group(group_id, cases, image_dir, output_dir, output_suffix)
        print(f"저장: {output_path}", flush=True)
    print(f"완료: {len(groups)}개 그룹, {sum(len(cases) for _, cases in groups)}개 샘플")


if __name__ == "__main__":
    main()
