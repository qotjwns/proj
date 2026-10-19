# 카테고리별로 bbox 면적이 이미지의 5% 이상인 양성과 대상 객체 주석이 없는
# 음성 후보를 분류하고, 이미지 ID·경로 목록과 개수 집계를 JSON으로 저장합니다.

import json
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
IMAGE_DIR = DATA_DIR / "coco" / "val2017"
ANNOTATION_PATH = DATA_DIR / "coco_rem" / "instances_valrem.json"
OUTPUT_PATH = DATA_DIR / "pool_counts.json"
POOL_DIR = DATA_DIR / "pools"
MIN_BBOX_AREA_RATIO = 0.05


def build_pool_ids(data):
    images = {image["id"]: image for image in data["images"]}
    categories = {category["id"]: category for category in data["categories"]}
    all_image_ids = set(images)

    # 존재 여부는 크기와 관계없이 모든 주석으로 판단합니다.
    present = {category_id: set() for category_id in categories}
    positive = {category_id: set() for category_id in categories}

    for annotation in data["annotations"]:
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        image = images[image_id]
        present[category_id].add(image_id)

        # segmentation의 area가 아니라 bbox의 너비 × 높이를 사용합니다.
        # 여러 작은 bbox의 면적을 합산하지 않습니다.
        _, _, bbox_width, bbox_height = annotation["bbox"]
        image_area = image["width"] * image["height"]
        if bbox_width * bbox_height >= MIN_BBOX_AREA_RATIO * image_area:
            positive[category_id].add(image_id)

    pools = {}
    for category_id in categories:
        positive_ids = positive[category_id]
        negative_candidate_ids = all_image_ids - present[category_id]
        excluded_small_only_ids = present[category_id] - positive_ids

        pools[category_id] = {
            "positive": sorted(positive_ids),
            "negative_candidates": sorted(negative_candidate_ids),
            "excluded_small_only": sorted(excluded_small_only_ids),
        }
    return pools


def summarize_pools(data, pools):
    rows = []
    for category in data["categories"]:
        category_id = category["id"]
        pool = pools[category_id]

        # 한 이미지에 같은 객체가 여러 개 있어도 이미지 수는 한 번만 셉니다.
        row = {
            "category_id": category_id,
            "category_name": category["name"],
            "positive_count": len(pool["positive"]),
            "negative_candidate_count": len(pool["negative_candidates"]),
            "excluded_small_only_count": len(pool["excluded_small_only"]),
        }
        rows.append(row)

    # 양성 이미지가 많은 순서로 출력하고, 동률이면 카테고리 ID순으로 정렬합니다.
    rows.sort(key=lambda row: (-row["positive_count"], row["category_id"]))
    return {
        "image_count": len(data["images"]),
        "category_count": len(data["categories"]),
        "min_bbox_area_ratio": MIN_BBOX_AREA_RATIO,
        "definitions": {
            "positive": "대상 카테고리의 bbox 중 하나 이상이 이미지 면적의 5% 이상",
            "negative_candidate": "크기와 관계없이 대상 카테고리의 주석이 하나도 없음",
            "excluded_small_only": "대상 카테고리의 주석이 있지만 모든 bbox가 이미지 면적의 5% 미만",
        },
        "categories": rows,
    }


def save_pools(data, pools):
    images = {image["id"]: image for image in data["images"]}
    image_records = {}
    for image_id, image in images.items():
        path = IMAGE_DIR / image["file_name"]
        image_records[image_id] = {
            "image_id": image_id,
            "image_path": path.relative_to(PROJECT_DIR).as_posix(),
        }

    for category in sorted(data["categories"], key=lambda item: item["id"]):
        category_id = category["id"]
        folder_name = f"{category_id:02d}_{category['name'].replace(' ', '_')}"
        category_dir = POOL_DIR / folder_name
        category_dir.mkdir(parents=True, exist_ok=True)

        for pool_type in ("positive", "negative_candidates"):
            image_ids = pools[category_id][pool_type]
            manifest = {
                "category_id": category_id,
                "category_name": category["name"],
                "pool_type": pool_type,
                "annotation_source": ANNOTATION_PATH.relative_to(PROJECT_DIR).as_posix(),
                "image_path_base": "project_root",
                "image_count": len(image_ids),
            }
            if pool_type == "positive":
                manifest["selection_rule"] = "at_least_one_bbox_meets_area_ratio"
                manifest["min_bbox_area_ratio"] = MIN_BBOX_AREA_RATIO
            else:
                manifest["selection_rule"] = "no_annotation_for_category_at_any_size"
                manifest["manual_review_completed"] = False
            manifest["images"] = [image_records[image_id] for image_id in image_ids]

            output_path = category_dir / f"{pool_type}.json"
            temporary_path = output_path.with_suffix(".json.tmp")
            with temporary_path.open("w", encoding="utf-8") as output:
                json.dump(manifest, output, ensure_ascii=False, indent=2)
                output.write("\n")
            temporary_path.replace(output_path)


def main():
    with ANNOTATION_PATH.open(encoding="utf-8") as source:
        data = json.load(source)
    pools = build_pool_ids(data)
    summary = summarize_pools(data, pools)
    save_pools(data, pools)

    print(f"전체 이미지: {summary['image_count']:,}장")
    print(f"카테고리: {summary['category_count']:,}개")
    print("양성 기준: bbox 면적 / 이미지 면적 >= 5%")
    print("Positive=양성, Negative candidates=음성 후보, Small-only=작은 객체만 있어 제외\n")
    print(f"{'ID':>3}  {'Category':<20} {'Positive':>8} {'Negative candidates':>19} {'Small-only':>10}")
    print("-" * 68)
    for row in summary["categories"]:
        print(
            f"{row['category_id']:>3}  {row['category_name']:<20} "
            f"{row['positive_count']:>8,} "
            f"{row['negative_candidate_count']:>19,} "
            f"{row['excluded_small_only_count']:>10,}"
        )

    with OUTPUT_PATH.open("w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(f"\n집계 저장: {OUTPUT_PATH}")
    print(f"양성·음성 후보 이미지 목록 저장: {POOL_DIR}")
    print("음성 후보는 주석 기준이며, 실제 객체 부재는 사진을 확인해야 합니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
