# 기본 20클래스 × 5그룹 생성: python src/data_manipulate/make_inference_pool.py
# 클래스 직접 지정: python src/data_manipulate/make_inference_pool.py --category cat dog
# 클래스 수 지정: python src/data_manipulate/make_inference_pool.py --num-classes 10
# 다른 샘플: python src/data_manipulate/make_inference_pool.py --seed 123
# 기본 저장 위치: data/inference_pools/inference_groups_seed42.json

import argparse
from pathlib import Path
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.data import normalize_category, read_json, write_json
from src.common.paths import IMAGE_DIR, INFERENCE_POOL_DIR, POOL_DIR, project_path


DEFAULT_POOL_DIR = POOL_DIR
DEFAULT_OUTPUT_DIR = INFERENCE_POOL_DIR
IMAGES_PER_LABEL = 8
POSITIVE_GROUPS = (
    (0, 1, 2, 3),  # P1, P2, P3, P4
    (4, 5, 6, 7),  # P5, P6, P7, P8
    (0, 2, 4, 6),  # P1, P3, P5, P7
    (1, 3, 5, 7),  # P2, P4, P6, P8
    (0, 3, 4, 7),  # P1, P4, P5, P8
)


def find_category_dir(pool_dir: Path, category: str) -> Path:
    """클래스 이름 또는 폴더명으로 후보 pool 하나를 찾습니다."""
    category_key = normalize_category(category).replace(" ", "_")
    matches = [
        path.parent for path in sorted(pool_dir.glob("*/positive.json"))
        if category_key in {
            path.parent.name.casefold(),
            path.parent.name.partition("_")[2].casefold(),
        }
    ]
    if len(matches) != 1:
        raise ValueError(
            f"클래스 '{category}'에 해당하는 pool이 {len(matches)}개입니다. "
            f"{pool_dir} 아래의 클래스 폴더명을 확인하세요."
        )
    return matches[0]


def load_candidates(path: Path, expected_type: str) -> dict:
    """후보 목록을 읽고 양음성 종류와 이미지 중복을 확인합니다."""
    pool = read_json(path)
    if pool["pool_type"] != expected_type:
        raise ValueError(f"pool 종류가 {expected_type}이어야 합니다: {path}")
    images = pool["images"]
    if len({image["image_id"] for image in images}) != len(images):
        raise ValueError(f"중복된 이미지 ID가 있습니다: {path}")
    if len({image["image_path"] for image in images}) != len(images):
        raise ValueError(f"중복된 이미지 경로가 있습니다: {path}")
    return pool


def select_category_dirs(
    pool_dir: Path, categories: list[str] | None, num_classes: int
) -> list[Path]:
    """지정한 클래스 또는 양성 후보가 많은 순서로 사용 가능한 클래스를 고릅니다."""
    if categories:
        selected = [find_category_dir(pool_dir, category) for category in categories]
        if len(set(selected)) != len(selected):
            raise ValueError("같은 클래스가 중복 지정되었습니다.")
        return selected
    if num_classes < 1:
        raise ValueError("--num-classes는 1 이상이어야 합니다.")

    available = []
    for positive_path in sorted(pool_dir.glob("*/positive.json")):
        category_dir = positive_path.parent
        positive = load_candidates(positive_path, "positive")
        negative = load_candidates(
            category_dir / "negative_candidates.json", "negative_candidates"
        )
        positive_count = len(positive["images"])
        negative_count = len(negative["images"])
        if min(positive_count, negative_count) < IMAGES_PER_LABEL:
            print(
                f"[후보 부족으로 제외] {positive['category_name']}: "
                f"양성 {positive_count}장, 음성 후보 {negative_count}장",
                file=sys.stderr,
            )
            continue
        available.append((positive_count, positive["category_id"], category_dir))

    available.sort(key=lambda item: (-item[0], item[1]))
    if len(available) < num_classes:
        raise ValueError(
            f"양성·음성 후보가 각각 8장 이상인 클래스는 {len(available)}개입니다. "
            f"요청한 {num_classes}개를 만들 수 없습니다."
        )
    return [item[2] for item in available[:num_classes]]


def sample_images(pool: dict, seed: int, prefix: str, label: int) -> list[dict]:
    """중복 없이 8장을 뽑고 P1~P8 또는 N1~N8 이름을 붙입니다."""
    rng = random.Random(f"{seed}:{pool['category_id']}:{pool['pool_type']}")
    candidates = sorted(pool["images"], key=lambda image: image["image_id"])
    selected = rng.sample(candidates, k=IMAGES_PER_LABEL)
    records = []
    for index, image in enumerate(selected, start=1):
        image_path = project_path(image["image_path"])
        if image_path.parent.resolve() != IMAGE_DIR.resolve():
            raise ValueError(
                f"파일명으로 참조할 이미지는 {IMAGE_DIR} 아래에 있어야 합니다: {image_path}"
            )
        records.append({
            "sample_id": f"{prefix}{index}",
            "image_id": image["image_id"],
            "image_path": image["image_path"],
            "label": label,
        })
    return records


def build_inference_pool(category_dir: Path, seed: int) -> dict:
    """한 클래스의 양성 8장·음성 후보 8장을 추론용 묶음으로 만듭니다."""
    positive = load_candidates(category_dir / "positive.json", "positive")
    negative = load_candidates(
        category_dir / "negative_candidates.json", "negative_candidates"
    )
    for key in ("category_id", "category_name", "annotation_source"):
        if positive[key] != negative[key]:
            raise ValueError(f"양성·음성 후보의 {key}가 다릅니다: {category_dir}")
    for key in ("image_id", "image_path"):
        positive_values = {image[key] for image in positive["images"]}
        negative_values = {image[key] for image in negative["images"]}
        if positive_values & negative_values:
            raise ValueError(f"양성·음성 후보에 겹치는 {key}가 있습니다: {category_dir}")

    positive_images = sample_images(positive, seed, prefix="P", label=1)
    negative_images = sample_images(negative, seed, prefix="N", label=0)
    return {
        "pool_id": f"{category_dir.name}_seed{seed}",
        "category_id": positive["category_id"],
        "category_name": positive["category_name"],
        "seed": seed,
        "annotation_source": positive["annotation_source"],
        "image_path_base": "project_root",
        "image_count": len(positive_images) + len(negative_images),
        "positive_count": len(positive_images),
        "negative_candidate_count": len(negative_images),
        "positive_selection_rule": positive["selection_rule"],
        "min_bbox_area_ratio": positive["min_bbox_area_ratio"],
        "negative_selection_rule": negative["selection_rule"],
        "negative_manual_review_completed": negative.get("manual_review_completed", False),
        "label_definition": {"1": "target_present", "0": "no_annotation_for_target"},
        "positive": positive_images,
        "negative_candidates": negative_images,
    }


def build_groups(pool: dict) -> dict:
    """클래스의 양성 8장을 4장씩 묶고 같은 음성 8장을 공유하는 5그룹을 만듭니다."""
    positive_names = [Path(image["image_path"]).name for image in pool["positive"]]
    negative_names = [Path(image["image_path"]).name for image in pool["negative_candidates"]]

    category_name = pool["category_name"].replace(" ", "_")
    groups = {}
    for group_index, indices in enumerate(POSITIVE_GROUPS, start=1):
        groups[f"group_{group_index:02d}_{category_name}"] = {
            "class": pool["category_name"],
            "pos_image": [positive_names[index] for index in indices],
            "neg_image": negative_names.copy(),
        }
    return groups


def save_inference_pool(groups: dict, output_dir: Path, seed: int) -> Path:
    """원본 이미지 복사 없이 그룹별 pos_image·neg_image 파일명 목록을 저장합니다."""
    output_path = output_dir / f"inference_groups_seed{seed}.json"
    write_json(groups, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="클래스별 양성 8장·음성 후보 8장에서 양성 4장·음성 8장의 그룹 5개 생성"
    )
    parser.add_argument(
        "--category", nargs="+",
        help="클래스 이름 또는 폴더명 목록. 지정하면 이 클래스들만 사용합니다.",
    )
    parser.add_argument(
        "--num-classes", type=int, default=20,
        help="자동 선택할 클래스 수 (기본: 20, --category 지정 시 사용하지 않음)",
    )
    parser.add_argument("--seed", type=int, default=42, help="샘플링 시드 (기본: 42)")
    parser.add_argument("--pool-dir", type=Path, default=DEFAULT_POOL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    category_dirs = select_category_dirs(
        args.pool_dir.expanduser().resolve(), args.category, args.num_classes
    )
    groups = {}
    for category_dir in category_dirs:
        pool = build_inference_pool(category_dir, args.seed)
        category_groups = build_groups(pool)
        if groups.keys() & category_groups.keys():
            raise ValueError(f"중복된 그룹 이름이 있습니다: {pool['category_name']}")
        groups.update(category_groups)
        print(
            f"[구성] {pool['category_name']}: "
            f"양성 8장·음성 후보 8장 → {len(category_groups)}그룹",
            flush=True,
        )
    output_path = save_inference_pool(
        groups, args.output_dir.expanduser().resolve(), args.seed
    )

    print(f"전체: {len(category_dirs)}클래스, {len(groups)}그룹")
    print("그룹당 양성 4장·음성 후보 8장. 음성 후보 8장은 클래스 내에서 공유합니다.")
    print(f"저장: {output_path}")


if __name__ == "__main__":
    main()
