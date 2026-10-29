# False-Yes와 False-No의 각 이미지를 한 장씩 추론: python src/inference/run_single_image_errors.py
# False-Yes만: python src/inference/run_single_image_errors.py --error-type false-yes
# False-No만: python src/inference/run_single_image_errors.py --error-type false-no
# 7B: python src/inference/run_single_image_errors.py --model-path data/models/Qwen/Qwen2.5-VL-7B-Instruct
# 저장 위치: data/single_image_results/<모델명>_<오류종류>_single_image_results.jsonl

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tqdm import tqdm

from src.common.data import read_jsonl
from src.common.paths import (
    DEFAULT_CASES_JSONL, DEFAULT_MODEL_PATH, RESULT_DIR, SINGLE_IMAGE_RESULT_DIR, project_path,
)
from src.evaluation.evaluate_composition import evaluate_composition, index_cases
from src.evaluation.evaluate_results import normalize_answer
from src.inference.inference_test import load_model, select_device
from src.inference.run_inference import infer_case, yes_no_token_ids


def select_error_cases(
    results: list[dict], cases_by_id: dict[str, dict], error_type: str, limit: int | None,
) -> list[dict]:
    """기존 3장 입력에서 틀린 케이스만 원래 결과 순서로 선택합니다."""
    evaluate_composition(results, cases_by_id)
    selected = []
    for result in results:
        target = normalize_answer(result["gt"])
        prediction = normalize_answer(result["model_answer"])
        if target == prediction:
            continue
        actual_type = "false-yes" if prediction == "Yes" else "false-no"
        if error_type != "all" and actual_type != error_type:
            continue
        selected.append({
            "case": cases_by_id[result["case_id"]],
            "source_result": result,
            "error_type": actual_type,
        })
    return selected[:limit] if limit is not None else selected


def build_single_image_tasks(selected: list[dict]) -> list[dict]:
    """이미지별 정답과 질문을 만들고 원래 케이스·입력 위치를 보존합니다."""
    tasks = []
    known_labels = {}
    for selected_case in selected:
        case = selected_case["case"]
        if case.get("image_path_base") != "project_root":
            raise ValueError(f"지원하지 않는 이미지 경로 기준입니다: {case['case_id']}")
        source_prompt = case.get("prompt")
        if not isinstance(source_prompt, str) or "in all 3 images" not in source_prompt:
            raise ValueError(f"{case['case_id']}: 'in all 3 images'가 들어 있는 원본 질문이 필요합니다.")
        # 대상 클래스와 답변 지시는 유지하고 3장 전체에 대한 조건만 한 장으로 바꿉니다.
        prompt = source_prompt.replace("in all 3 images", "in this image")
        image_dir = project_path(case["image_dir"]).resolve()
        for position, (name, label) in enumerate(zip(case["image_order"], case["image_labels"]), start=1):
            image_path = (image_dir / name).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            target = "Yes" if label == "P" else "No"
            cache_key = (image_path, prompt)
            if cache_key in known_labels and known_labels[cache_key] != target:
                raise ValueError(f"동일 이미지·질문의 정답이 다릅니다: {image_path}")
            known_labels[cache_key] = target
            tasks.append({
                **selected_case,
                "image_dir": image_dir,
                "image_path": image_path,
                "image_position": position,
                "image_name": name,
                "image_label": label,
                "prompt": prompt,
                "gt": target,
                "cache_key": cache_key,
            })
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="False-Yes/False-No 케이스의 이미지를 한 장씩 재추론")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="로컬 모델 폴더 (기본: 3B)")
    parser.add_argument(
        "--input-jsonl", type=Path,
        help="원래 3장 입력의 추론 결과 (기본: data/inference_results/<모델명>_results.jsonl)",
    )
    parser.add_argument("--cases-jsonl", type=Path, default=DEFAULT_CASES_JSONL)
    parser.add_argument("--error-type", choices=("all", "false-yes", "false-no"), default="all")
    parser.add_argument("--limit", type=int, help="선택할 원본 오류 케이스 수 (케이스당 이미지 3장)")
    parser.add_argument("--output", type=Path, help="개별 이미지 결과 JSONL 경로")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit는 1 이상이어야 합니다.")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens는 1 이상이어야 합니다.")

    model_path = args.model_path.expanduser().resolve()
    input_path = (
        args.input_jsonl.expanduser().resolve() if args.input_jsonl
        else RESULT_DIR / f"{model_path.name}_results.jsonl"
    )
    cases_path = args.cases_jsonl.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve() if args.output else
        SINGLE_IMAGE_RESULT_DIR / f"{model_path.name}_{args.error_type}_single_image_results.jsonl"
    )
    if output_path in (input_path, cases_path):
        parser.error("원본 추론 결과나 cases 파일에 덮어쓸 수 없습니다.")
    try:
        cases_by_id = index_cases(read_jsonl(cases_path))
        selected = select_error_cases(read_jsonl(input_path), cases_by_id, args.error_type, args.limit)
        tasks = build_single_image_tasks(selected)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not tasks:
        print("선택한 유형의 오류 케이스가 없습니다.")
        return

    unique_count = len({task["cache_key"] for task in tasks})
    device = select_device(args.device)
    print(f"입력: {input_path}\n모델: {model_path}\n장치: {device}", flush=True)
    print(
        f"오류 케이스 {len(selected)}개 → 이미지별 결과 {len(tasks)}행 "
        f"(중복을 제외한 실제 추론 {unique_count}회)\n출력: {output_path}", flush=True,
    )
    model, processor = load_model(model_path, device)
    token_ids = yes_no_token_ids(processor.tokenizer)
    cache = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for task in tqdm(tasks, desc="Single-image inference", unit="image"):
            case = task["case"]
            if task["cache_key"] not in cache:
                single_case = {
                    "case_id": case["case_id"],
                    "image_order": [task["image_name"]],
                    "prompt": task["prompt"],
                    "gt": task["gt"],
                }
                prediction = infer_case(
                    model, processor, single_case, token_ids, args.max_new_tokens,
                    image_dir=task["image_dir"],
                )
                cache[task["cache_key"]] = {
                    key: prediction[key] for key in ("model_answer", "yes_logit", "no_logit")
                }
            prediction = cache[task["cache_key"]]
            answer = normalize_answer(prediction["model_answer"])
            record = {
                "case_id": case["case_id"],
                "group_id": case["group_id"],
                "class": case["class"],
                "source_error_type": task["error_type"],
                "source_results_path": str(input_path),
                "source_prompt": case["prompt"],
                "source_result": task["source_result"],
                "model_path": str(model_path),
                "image_position": task["image_position"],
                "image_order": [task["image_name"]],
                "image_path": str(task["image_path"]),
                "image_label": task["image_label"],
                "prompt": task["prompt"],
                "gt": task["gt"],
                **prediction,
                "normalized_answer": answer,
                "is_correct": answer == task["gt"] if answer is not None else None,
            }
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
    print(f"완료: {len(tasks)}개 이미지별 결과 저장 → {output_path}")


if __name__ == "__main__":
    main()
