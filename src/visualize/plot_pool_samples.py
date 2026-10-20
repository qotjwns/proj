# 전체 클래스: python src/data_manipulate/plot_pool_samples.py
# 특정 클래스: python src/data_manipulate/plot_pool_samples.py --category person
# 다른 샘플: python src/data_manipulate/plot_pool_samples.py --seed 123
# 저장 위치: data/pool_samples/<클래스 폴더명>.png

import argparse
import json
from pathlib import Path
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POOL_DIR = PROJECT_ROOT / "data" / "pools"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "pool_samples"
SAMPLES_PER_POOL = 4


def load_pool(pool_path: Path) -> dict:
    """make_pool.py가 생성한 이미지 목록을 읽습니다."""
    with pool_path.open(encoding="utf-8") as source:
        pool = json.load(source)
    if pool["image_path_base"] != "project_root":
        raise ValueError(f"프로젝트 기준 이미지 경로가 필요합니다: {pool_path}")
    return pool


def sample_images(pool: dict, seed: int) -> list[dict]:
    """클래스·pool 종류별로 재현 가능하게 최대 4장을 중복 없이 뽑습니다."""
    rng = random.Random(f"{seed}:{pool['category_id']}:{pool['pool_type']}")
    images = pool["images"]
    return rng.sample(images, k=min(SAMPLES_PER_POOL, len(images)))


def load_image(image_path: str) -> Image.Image:
    """JSON의 상대 경로를 프로젝트 기준으로 해석해 이미지를 읽습니다."""
    path = Path(image_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def plot_category(category_dir: Path, output_dir: Path, seed: int) -> Path:
    """양성 4장과 음성 후보 4장을 2행 4열 PNG 한 장으로 저장합니다."""
    positive = load_pool(category_dir / "positive.json")
    negative = load_pool(category_dir / "negative_candidates.json")
    if (
        positive["category_id"] != negative["category_id"]
        or positive["category_name"] != negative["category_name"]
        or positive["pool_type"] != "positive"
        or negative["pool_type"] != "negative_candidates"
    ):
        raise ValueError(f"양성·음성 후보의 클래스 또는 pool 종류가 다릅니다: {category_dir}")

    fig, axes = plt.subplots(2, SAMPLES_PER_POOL, figsize=(16, 9), squeeze=False)
    try:
        fig.suptitle(
            f"{positive['category_name']} (class ID: {positive['category_id']})\n"
            f"POS pool: {len(positive['images'])}  |  "
            f"NEG candidate pool: {len(negative['images'])}  |  seed: {seed}",
            fontsize=17,
        )

        rows = [
            (positive, "POS", "forestgreen"),
            (negative, "NEG candidate", "firebrick"),
        ]
        for row_index, (pool, label, color) in enumerate(rows):
            samples = sample_images(pool, seed)
            if len(samples) < SAMPLES_PER_POOL:
                print(
                    f"[샘플 부족] {pool['category_name']} / {label}: "
                    f"{len(samples)}장만 표시합니다.",
                    file=sys.stderr,
                )

            for column, axis in enumerate(axes[row_index]):
                axis.set_axis_off()
                if column >= len(samples):
                    axis.set_title(label, color=color)
                    axis.text(
                        0.5, 0.5, "Not enough samples",
                        ha="center", va="center", transform=axis.transAxes,
                        color="gray",
                    )
                    continue

                record = samples[column]
                axis.set_title(
                    f"{label} | image ID: {record['image_id']}",
                    fontsize=11, color=color,
                )
                try:
                    with load_image(record["image_path"]) as image:
                        axis.imshow(image)
                except OSError as error:
                    axis.text(
                        0.5, 0.5, "Image load failed",
                        ha="center", va="center", transform=axis.transAxes,
                        color="firebrick",
                    )
                    print(
                        f"[이미지 오류] {record['image_path']}: {error}",
                        file=sys.stderr,
                    )

        fig.text(
            0.5, 0.02,
            "Top: positive samples | Bottom: negative candidates (no annotation for this class)",
            ha="center", fontsize=11,
        )
        fig.tight_layout(rect=(0, 0.05, 1, 0.91))
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{category_dir.name}.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        return output_path
    finally:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="클래스별 pool에서 양성 4장·음성 후보 4장을 뽑아 PNG로 저장"
    )
    parser.add_argument(
        "--pool-dir", type=Path, default=DEFAULT_POOL_DIR,
        help="클래스별 pool 폴더가 있는 경로",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="PNG 저장 폴더",
    )
    parser.add_argument(
        "--category",
        help='특정 클래스만 출력: person, 01_person, "traffic light" 등 (기본: 전체)',
    )
    parser.add_argument("--seed", type=int, default=42, help="샘플링 시드 (기본: 42)")
    args = parser.parse_args()

    pool_dir = args.pool_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not pool_dir.is_dir():
        parser.error(f"pool 폴더가 없습니다: {pool_dir}")

    category_dirs = sorted(path.parent for path in pool_dir.glob("*/positive.json"))
    if args.category:
        category_key = args.category.strip().casefold().replace(" ", "_")
        category_dirs = [
            path for path in category_dirs
            if category_key in {
                path.name.casefold(),
                path.name.partition("_")[2].casefold(),
            }
        ]
    if not category_dirs:
        parser.error("조건에 맞는 클래스의 positive.json을 찾지 못했습니다.")

    for category_dir in category_dirs:
        output_path = plot_category(category_dir, output_dir, args.seed)
        print(f"저장: {output_path}", flush=True)
    print(f"완료: {len(category_dirs)}개 클래스")


if __name__ == "__main__":
    main()
