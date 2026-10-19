# COCO 검증 이미지와 COCO-ReM 주석의 파일 수, 필수 항목, 중복 ID,
# 이미지·카테고리 참조 관계 및 bbox 형식을 검사하고 결과를 출력합니다.

import json
import math
from pathlib import Path
import sys


DATA_DIR = Path(__file__).resolve().parents[2] / "data"
IMAGE_DIR = DATA_DIR / "coco" / "val2017"
ANNOTATION_PATH = DATA_DIR / "coco_rem" / "instances_valrem.json"
EXPECTED_IMAGE_COUNT = 5000


def report(label, errors):
    if not errors:
        print(f"[PASS] {label}")
        return 0
    print(f"[FAIL] {label}: 문제 {len(errors):,}건")
    for message in errors[:5]:
        print(f"       {message}")
    if len(errors) > 5:
        print(f"       ... 나머지 {len(errors) - 5:,}건")
    return 1


def index_records(records, name, errors):
    indexed = {}
    for position, record in enumerate(records):
        if not isinstance(record, dict) or type(record.get("id")) is not int:
            errors.append(f"{name}[{position}]: 정수 id가 필요합니다.")
            continue
        record_id = record["id"]
        if record_id in indexed:
            errors.append(f"{name}: 중복 id={record_id}")
            continue
        indexed[record_id] = record
    return indexed


def is_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def main():
    print(f"이미지 경로: {IMAGE_DIR}")
    print(f"주석 경로: {ANNOTATION_PATH}\n")

    errors = []
    if not IMAGE_DIR.is_dir():
        errors.append(f"이미지 폴더가 없습니다: {IMAGE_DIR}")
    if not ANNOTATION_PATH.is_file():
        errors.append(f"주석 파일이 없습니다: {ANNOTATION_PATH}")
    if report("데이터 경로", errors):
        return 1

    with ANNOTATION_PATH.open(encoding="utf-8") as source:
        data = json.load(source)

    errors = []
    if not isinstance(data, dict):
        errors.append("JSON 최상위 값은 객체여야 합니다.")
    else:
        for key in ("images", "categories", "annotations"):
            if not isinstance(data.get(key), list) or not data[key]:
                errors.append(f"{key}: 비어 있지 않은 목록이 필요합니다.")
    if report("JSON 읽기 및 필수 목록", errors):
        return 1

    failures = 0
    image_files = {path.name: path for path in IMAGE_DIR.glob("*.jpg") if path.is_file()}
    errors = []
    if len(image_files) != EXPECTED_IMAGE_COUNT:
        errors.append(f"실제 JPG {len(image_files):,}장 / 예상 {EXPECTED_IMAGE_COUNT:,}장")
    if len(data["images"]) != EXPECTED_IMAGE_COUNT:
        errors.append(f"images 항목 {len(data['images']):,}개 / 예상 {EXPECTED_IMAGE_COUNT:,}개")
    failures += report(f"이미지 및 images 항목 수 ({EXPECTED_IMAGE_COUNT:,})", errors)

    errors = []
    images = index_records(data["images"], "images", errors)
    categories = index_records(data["categories"], "categories", errors)
    annotations = index_records(data["annotations"], "annotations", errors)
    failures += report("이미지·카테고리·객체 주석 ID 형식 및 중복", errors)

    errors = []
    filenames = set()
    for image_id, image in images.items():
        filename = image.get("file_name")
        if (
            not isinstance(filename, str)
            or not filename.endswith(".jpg")
            or Path(filename).name != filename
        ):
            errors.append(f"image_id={image_id}: 잘못된 file_name={filename!r}")
        else:
            if filename in filenames:
                errors.append(f"중복 file_name={filename}")
            filenames.add(filename)
        for key in ("width", "height"):
            if type(image.get(key)) is not int or image[key] <= 0:
                errors.append(f"image_id={image_id}: {key}는 양의 정수여야 합니다.")
    for category_id, category in categories.items():
        if not isinstance(category.get("name"), str) or not category["name"].strip():
            errors.append(f"category_id={category_id}: 유효한 name이 필요합니다.")
    failures += report("파일명·이미지 크기·카테고리 이름", errors)

    errors = []
    for filename in sorted(filenames - image_files.keys()):
        errors.append(f"주석에 있지만 파일이 없습니다: {filename}")
    for filename in sorted(image_files.keys() - filenames):
        errors.append(f"파일은 있지만 images 항목에 없습니다: {filename}")
    for filename, path in image_files.items():
        if path.stat().st_size == 0:
            errors.append(f"빈 이미지 파일: {filename}")
    failures += report("이미지 파일과 주석의 일치 및 빈 파일", errors)

    reference_errors = []
    bbox_errors = []
    for annotation_id, annotation in annotations.items():
        for key, indexed in (("image_id", images), ("category_id", categories)):
            reference = annotation.get(key)
            if type(reference) is not int or reference not in indexed:
                reference_errors.append(f"annotation_id={annotation_id}: 잘못된 {key}={reference!r}")
        bbox = annotation.get("bbox")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(is_number(value) for value in bbox)
            or bbox[0] < 0
            or bbox[1] < 0
            or bbox[2] <= 0
            or bbox[3] <= 0
        ):
            bbox_errors.append(f"annotation_id={annotation_id}: 잘못된 bbox={bbox!r}")
    failures += report("객체 주석의 image_id·category_id 연결", reference_errors)
    failures += report("bbox 형식 [x, y, width, height] 및 유효한 크기", bbox_errors)

    print(
        f"\nJPG {len(image_files):,}장 / 카테고리 {len(data['categories']):,}개"
        f" / 객체 주석 {len(data['annotations']):,}개"
    )
    if failures:
        print(f"검증 실패: {failures}개 검사 항목에서 문제가 발견됐습니다.")
        return 1
    print("검증 통과: 검사한 파일 구조와 주석 연결이 정상입니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
