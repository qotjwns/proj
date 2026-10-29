"""그룹 JSON → 이미지 3장짜리 Yes 400개 / No 400개. 모델·이미지는 로드하지 않습니다."""

from collections import Counter
import hashlib
from itertools import combinations, product
import json
from pathlib import Path
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.data import read_json
from src.common.paths import DEFAULT_CASES_JSONL, DEFAULT_GROUPS_JSON, IMAGE_DIR, PROJECT_ROOT


SEED = 42
GROUPS_JSON = DEFAULT_GROUPS_JSON
OUTPUT_JSONL = DEFAULT_CASES_JSONL
PROMPT = "Is there a {object} in all 3 images? Please answer with Yes or No."


def make_case(group_id: str, category: str, images: tuple[str, ...], positive_count: int) -> dict:
    labels = ["P"] * positive_count + ["N"] * (3 - positive_count)
    answer = "Yes" if positive_count == 3 else "No"
    record = {
        "experiment_type": "balanced3",
        "group_id": group_id,
        "class": category,
        "condition": answer.upper(),
        "condition_code": "ABCD"[positive_count],
        "composition": "+".join(labels),
        "image_path_base": "project_root",
        "image_dir": IMAGE_DIR.relative_to(PROJECT_ROOT).as_posix(),
        "image_order": list(images),
        "image_labels": labels,
        "prompt": PROMPT.format(object=category),
        "gt": answer,
        "positive_count": positive_count,
        "negative_count": 3 - positive_count,
        "negative_images": list(images[positive_count:]),
        "negative_positions": list(range(positive_count, 3)),
    }
    # 기존 JSONL과 같은 규칙: case_id를 제외한 레코드의 SHA256 앞 16자리.
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    case_id = f"balanced3__{group_id}__{answer.lower()}__{digest}"
    return {"case_id": case_id, **record}


def build_cases(groups: dict) -> list[dict]:
    """그룹당 Yes 4개·No 4개를 만들고, 같은 클래스의 이미지 3장 조합 중복을 제외합니다."""
    if len(groups) != 100:
        raise ValueError("800개 생성을 위해 원본 100개 그룹이 필요합니다.")
    rng = random.Random(SEED)
    cases = []
    used = set()
    known_labels = {}
    for group_index, (group_id, group) in enumerate(groups.items()):
        positive, negative = group["pos_image"], group["neg_image"]
        if len(positive) != 4 or len(negative) != 8 or len(set(positive + negative)) != 12:
            raise ValueError(f"{group_id}: 겹치지 않는 양성 4장·음성 8장이 필요합니다.")
        category = group["class"]
        for label, images in (("P", positive), ("N", negative)):
            for image in images:
                key = (category, image)
                if key in known_labels and known_labels[key] != label:
                    raise ValueError(f"{category}: 그룹 간 이미지 라벨이 다릅니다: {image}")
                known_labels[key] = label

        # PPP 4개를 먼저 기록. No는 A/B/C 하나씩 + 그룹 순서에 따라 A→B→C 추가.
        for positive_count in (3, 0, 1, 2):
            count = 4 if positive_count == 3 else 1 + int(positive_count == group_index % 3)
            candidates = [
                pos + neg
                for pos, neg in product(combinations(positive, positive_count), combinations(negative, 3 - positive_count))
                if (category, tuple(sorted(pos + neg))) not in used
            ]
            if positive_count != 3:
                rng.shuffle(candidates)
            if len(candidates) < count:
                raise ValueError(f"{group_id}: 중복 없는 {'ABCD'[positive_count]} 조합이 부족합니다.")
            for images in candidates[:count]:
                used.add((category, tuple(sorted(images))))
                cases.append(make_case(group_id, category, images, positive_count))
    return cases


def main() -> None:
    cases = build_cases(read_json(GROUPS_JSON))
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    # 기존 A/B/C 실험 입력을 덮어쓰지 않습니다. 새로 만들려면 OUTPUT_JSONL 이름을 바꾸세요.
    with OUTPUT_JSONL.open("x", encoding="utf-8") as output:
        for case in cases:
            output.write(json.dumps(case, ensure_ascii=False) + "\n")
    counts = Counter(case["condition_code"] for case in cases)
    print(f"전체 {len(cases)}개 | Yes {counts['D']} / No {counts['A'] + counts['B'] + counts['C']}")
    print(f"A(NNN) {counts['A']} / B(PNN) {counts['B']} / C(PPN) {counts['C']} / D(PPP) {counts['D']}")
    print(f"저장: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
