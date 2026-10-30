# 공통 결과 형식·파서. 사용: python src/experiments/c/run.py --limit 20

from pathlib import Path
import re

from src.common.data import read_jsonl


C_PROMPT = (
    "For each of the 3 images, determine whether there is at least one {object} in that image. "
    "Answer with exactly three labels (Yes or No) in image order, separated by commas. "
    "Do not provide explanations or a final combined answer."
)


def read_results(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    if any(row.get("schema_version") != 1 for row in rows):
        raise ValueError("구형 결과입니다. src/data_manipulate/convert_results.py로 먼저 변환하세요.")
    return rows


def normalize_answer(answer: str) -> str | None:
    return {"yes": "Yes", "no": "No"}.get(answer.strip().rstrip(".!?。！？").strip().casefold())


def parse_labels(answer: str) -> list[str] | None:
    match = re.fullmatch(r"\s*(Yes|No)\s*,\s*(Yes|No)\s*,\s*(Yes|No)\s*", answer)
    return list(match.groups()) if match else None


def combine_labels(labels: list | None) -> str | None:
    if labels is None or len(labels) != 3 or any(label is None for label in labels):
        return None
    return "Yes" if all(label == "Yes" for label in labels) else "No"


def make_result(
    case: dict, experiment: str, model_path: Path, prompt: str,
    outputs: list[dict], max_new_tokens: int | None,
) -> dict:
    answers = [output["model_answer"] for output in outputs]
    image_predictions = None
    if experiment == "a":
        prediction = normalize_answer(answers[0])
    else:
        if experiment == "b":
            image_predictions = [normalize_answer(answer) for answer in answers]
        else:
            image_predictions = parse_labels(answers[0])
        prediction = combine_labels(image_predictions)
    return {
        "schema_version": 1, "experiment": experiment,
        "case_id": case["case_id"], "group_id": case["group_id"], "class": case["class"],
        "model_path": str(model_path), "image_dir": case["image_dir"],
        "image_order": case["image_order"], "gt": case["gt"],
        "image_gt": ["Yes" if label == "P" else "No" for label in case["image_labels"]],
        "prompt": prompt, "model_answers": answers, "image_predictions": image_predictions,
        "prediction": prediction, "correct": prediction == case["gt"],
        "parse_valid": prediction is not None,
        "yes_logits": [output["yes_logit"] for output in outputs],
        "no_logits": [output["no_logit"] for output in outputs],
        "max_new_tokens": max_new_tokens,
    }


def index_cases(cases: list[dict]) -> dict[str, dict]:
    indexed = {case["case_id"]: case for case in cases}
    if len(indexed) != len(cases):
        raise ValueError("원본 case_id가 중복됩니다.")
    return indexed


def align_results(cases: list[dict], records: list[dict], *, complete: bool = False) -> list[dict]:
    """비교 대상의 중복·누락·이미지 순서 불일치만 확인합니다."""
    indexed = {row["case_id"]: row for row in records}
    source = index_cases(cases)
    if len(indexed) != len(records) or not set(indexed) <= set(source):
        raise ValueError("결과에 중복 또는 원본에 없는 case_id가 있습니다.")
    if complete and set(indexed) != set(source):
        raise ValueError("전체 비교에는 원본 케이스가 모두 필요합니다. 일부 교집합만 비교하지 않습니다.")
    ordered = []
    for case in cases:
        if case["case_id"] not in indexed:
            continue
        row = indexed[case["case_id"]]
        targets = ["Yes" if label == "P" else "No" for label in case["image_labels"]]
        if row["image_order"] != case["image_order"] or row["gt"] != case["gt"] or row["image_gt"] != targets:
            raise ValueError(f"{case['case_id']}: 이미지 순서 또는 정답이 원본과 다릅니다.")
        ordered.append(row)
    return ordered
