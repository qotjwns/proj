# JSON 입출력·필터 공통 모듈. 사용: python src/data_manipulate/make_pool.py

import json
from pathlib import Path


def read_json(path: Path):
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_json(data, output_path: Path) -> None:
    """임시 파일에 모두 쓴 다음 최종 JSON을 교체합니다."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        json.dump(data, output, ensure_ascii=False, indent=2)
        output.write("\n")
    temporary_path.replace(output_path)


def normalize_category(category: str) -> str:
    return category.strip().casefold().replace("_", " ")


def select_groups(groups: dict, group_id: str | None, category: str | None) -> dict:
    """입력 순서를 유지하면서 그룹 하나 또는 특정 클래스를 선택합니다."""
    selected = groups
    if group_id:
        selected = {group_id: selected[group_id]}
    if category:
        category_key = normalize_category(category)
        selected = {
            name: group for name, group in selected.items()
            if normalize_category(group["class"]) == category_key
        }
    return selected
