# 실행: python src/visualize/plot_errors.py --experiment a --model-path data/models/Qwen/Qwen2.5-VL-3B-Instruct
# PPN False-Yes만: python src/visualize/plot_errors.py --experiment c --condition C --error-type false-yes

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.common.images import load_image
from src.common.paths import DEFAULT_MODEL_PATH, project_path, report_dir, result_path
from src.common.results import read_results


def plot_page(rows: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(len(rows), 3, figsize=(16, 4.5 * len(rows)), squeeze=False)
    try:
        for row, panels in zip(rows, axes):
            for index, (axis, name, target) in enumerate(zip(panels, row["image_order"], row["image_gt"])):
                with load_image(project_path(row["image_dir"]) / name) as image:
                    axis.imshow(image)
                prediction = row["image_predictions"][index] if row["image_predictions"] else "N/A"
                axis.set_title(f"{row['class']} | Image {index + 1} | GT: {target} | Pred: {prediction}\n{name}", fontsize=10)
                axis.set_xticks([])
                axis.set_yticks([])
            panels[1].text(
                0.5, 1.25,
                f"{row['case_id']}\nFinal: {row['prediction']} | GT: {row['gt']} | Raw: {row['model_answers']}",
                transform=panels[1].transAxes, ha="center", fontsize=9,
            )
        fig.tight_layout(h_pad=7)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="오답 케이스의 원본 이미지 3장을 입력 순서대로 표시")
    parser.add_argument("--experiment", choices=("a", "b", "c"), required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--run", default="full")
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--condition", choices=("A", "B", "C", "D"))
    parser.add_argument("--error-type", choices=("all", "false-yes", "false-no"), default="all")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    model_path = args.model_path.expanduser().resolve()
    source = args.input_jsonl or result_path(args.experiment, model_path, args.run)
    rows = read_results(source)
    if any(row["experiment"] != args.experiment or Path(row["model_path"]).name != model_path.name for row in rows):
        raise ValueError("선택한 실험·모델과 입력 결과가 다릅니다.")
    rows = [row for row in rows if row["prediction"] != row["gt"]]
    if args.condition:
        rows = [row for row in rows if row["image_gt"].count("Yes") == "ABCD".index(args.condition)]
    if args.error_type != "all":
        prediction = "Yes" if args.error_type == "false-yes" else "No"
        rows = [row for row in rows if row["prediction"] == prediction]
    folder = f"errors_{args.condition or 'all'}_{args.error_type}"
    output_dir = args.output_dir or report_dir(args.experiment, model_path, source.parent.name) / folder
    for start in range(0, len(rows), 5):
        plot_page(rows[start:start + 5], output_dir / f"page_{start // 5 + 1:03d}.png")
    print(f"오답 {len(rows)}개 | 저장: {output_dir}")


if __name__ == "__main__":
    main()
