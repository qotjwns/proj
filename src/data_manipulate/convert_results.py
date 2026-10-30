# A 변환: python src/data_manipulate/convert_results.py --experiment a --input-jsonl path/to/old_results.jsonl
# B 변환: python src/data_manipulate/convert_results.py --experiment b --input-jsonl path/to/single_image_results.jsonl
# 모델 지정: 위 명령에 --model-path data/models/Qwen/Qwen2.5-VL-7B-Instruct 추가

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.data import read_jsonl
from src.common.paths import DEFAULT_CASES_JSONL, DEFAULT_MODEL_PATH
from src.common.results import C_PROMPT, align_results, index_cases, make_result


def convert(records: list[dict], cases: list[dict], experiment: str, model_path: Path, max_new_tokens: int | None) -> list[dict]:
    cases_by_id = index_cases(cases)
    converted = []
    if experiment == "b":
        grouped = defaultdict(dict)
        for row in records:
            positions = grouped[row["case_id"]]
            if row["image_position"] in positions:
                raise ValueError("기존 B 결과에 중복된 케이스·이미지 위치가 있습니다.")
            positions[row["image_position"]] = row
        for case_id, positions in grouped.items():
            if set(positions) != {1, 2, 3}:
                raise ValueError(f"{case_id}: 이미지 3개의 결과가 모두 필요합니다.")
            rows = [positions[index] for index in (1, 2, 3)]
            case = cases_by_id[case_id]
            if [row["image_order"][0] for row in rows] != case["image_order"] or len({row["prompt"] for row in rows}) != 1:
                raise ValueError(f"{case_id}: 원본 이미지 순서 또는 단일 이미지 질문이 다릅니다.")
            targets = ["Yes" if label == "P" else "No" for label in case["image_labels"]]
            if [row["gt"] for row in rows] != targets:
                raise ValueError(f"{case_id}: 단일 이미지 정답이 원본과 다릅니다.")
            if any(Path(row["model_path"].replace("\\", "/")).name != model_path.name for row in rows):
                raise ValueError("변환할 B 결과의 모델이 --model-path와 다릅니다.")
            outputs = [{key: row.get(key) for key in ("model_answer", "yes_logit", "no_logit")} for row in rows]
            stored_limit = rows[0].get("max_new_tokens")
            token_limit = max_new_tokens if max_new_tokens is not None else stored_limit
            converted.append(make_result(
                case, experiment, model_path, rows[0]["prompt"], outputs, token_limit,
            ))
    else:
        for row in records:
            case = cases_by_id[row["case_id"]]
            if "model_path" in row and Path(row["model_path"].replace("\\", "/")).name != model_path.name:
                raise ValueError("변환할 결과의 모델이 --model-path와 다릅니다.")
            if row["image_order"] != case["image_order"] or row["gt"] != case["gt"]:
                raise ValueError(f"{case['case_id']}: 원본 이미지 순서 또는 정답이 다릅니다.")
            prompt = case["prompt"] if experiment == "a" else row["prompt"]
            if experiment == "c" and prompt != C_PROMPT.format(object=case["class"]):
                raise ValueError("기존 C 결과의 질문이 확정 프롬프트와 다릅니다.")
            output = {
                "model_answer": row["model_answer"],
                "yes_logit": row.get("yes_logit") if experiment == "a" else None,
                "no_logit": row.get("no_logit") if experiment == "a" else None,
            }
            token_limit = max_new_tokens if max_new_tokens is not None else row.get("max_new_tokens")
            converted.append(make_result(case, experiment, model_path, prompt, [output], token_limit))
    return align_results(cases, converted)


def main() -> None:
    parser = argparse.ArgumentParser(description="구형 A/B/C 결과를 공통 JSONL 형식의 새 파일로 변환")
    parser.add_argument("--experiment", choices=("a", "b", "c"), required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--cases-jsonl", type=Path, default=DEFAULT_CASES_JSONL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-new-tokens", type=int, help="기존 실행 설정을 아는 경우만 지정; 모르면 null로 보존")
    args = parser.parse_args()
    source = args.input_jsonl.expanduser().resolve()
    output = args.output.expanduser().resolve() if args.output else source.with_name(source.stem + "_v1.jsonl")
    rows = convert(
        read_jsonl(source), read_jsonl(args.cases_jsonl), args.experiment,
        args.model_path.expanduser().resolve(), args.max_new_tokens,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"변환: {len(rows)}개 → {output}")


if __name__ == "__main__":
    main()
