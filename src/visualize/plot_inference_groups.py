# 전체 그룹: python src/visualize/plot_inference_groups.py
# 양성 4장·음성 후보 8장 확인용. 3장 실험 plot은 plot_inference_cases.py를 사용합니다.
# 그룹 하나: python src/visualize/plot_inference_groups.py --group group_01_person
# 특정 클래스: python src/visualize/plot_inference_groups.py --category cat
# 다른 JSON: python src/visualize/plot_inference_groups.py --input-json data/inference_pools/inference_groups_seed123.json
# 저장 위치: data/inference_pool_plots/<그룹명>.png

import argparse
from pathlib import Path

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.common.data import read_json as load_groups, select_groups
from src.common.images import load_image
from src.common.paths import DATA_DIR, DEFAULT_GROUPS_JSON, IMAGE_DIR


DEFAULT_INPUT_JSON = DEFAULT_GROUPS_JSON
DEFAULT_IMAGE_DIR = IMAGE_DIR
DEFAULT_OUTPUT_DIR = DATA_DIR / "inference_pool_plots"


def draw_image(axis, image_path: Path, label: str, color: str) -> None:
    """한 칸에 라벨·파일명·이미지를 표시합니다."""
    axis.set_axis_off()
    axis.set_title(f"{label}\n{image_path.name}", color=color, fontsize=11)
    with load_image(image_path) as image:
        axis.imshow(image)


def plot_group(group_name: str, group: dict, image_dir: Path, output_dir: Path) -> Path:
    """JSON 순서대로 양성 4장·음성 후보 8장을 3행 4열 PNG로 저장합니다."""
    entries = [
        (name, f"POS {index}", "forestgreen")
        for index, name in enumerate(group["pos_image"], start=1)
    ] + [
        (name, f"NEG candidate {index}", "firebrick")
        for index, name in enumerate(group["neg_image"], start=1)
    ]
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), squeeze=False)
    try:
        fig.suptitle(f"{group_name}  |  class: {group['class']}", fontsize=18)
        for axis, (name, label, color) in zip(axes.flat, entries):
            draw_image(axis, image_dir / name, label, color)
        fig.text(
            0.5, 0.015,
            "Top row: POS 1-4 | Middle row: NEG 1-4 | Bottom row: NEG 5-8\n"
            "NEG candidates: no annotation for this class",
            ha="center", fontsize=10,
        )
        fig.tight_layout(rect=(0, 0.055, 1, 0.95))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{group_name}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return output_path
    finally:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="추론용 그룹 JSON의 양성 4장·음성 후보 8장을 그룹별 PNG로 저장"
    )
    parser.add_argument(
        "--input-json", type=Path, default=DEFAULT_INPUT_JSON,
        help="make_inference_pool.py로 생성한 그룹 JSON",
    )
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--group", help="그룹 하나만 출력 (예: group_01_person)")
    parser.add_argument("--category", help='해당 클래스의 그룹만 출력 (예: cat, "traffic light")')
    args = parser.parse_args()

    image_dir = args.image_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    groups = load_groups(args.input_json.expanduser().resolve())
    selected = select_groups(groups, args.group, args.category)
    for group_name, group in selected.items():
        output_path = plot_group(group_name, group, image_dir, output_dir)
        print(f"저장: {output_path}", flush=True)
    print(f"완료: {len(selected)}개 그룹")


if __name__ == "__main__":
    main()
