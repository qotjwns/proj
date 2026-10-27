# 전체 추론: python src/inference/run_inference.py
# 한 그룹: python src/inference/run_inference.py --group group_01_cat
# 소규모 확인: python src/inference/run_inference.py --limit 8
# 저장 형식: 샘플마다 case_id, image_order, model_answer, answer, yes_logit, no_logit

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from tqdm import tqdm

from src.common.data import normalize_category, read_jsonl as load_cases
from src.common.images import load_image
from src.common.paths import DEFAULT_CASES_JSONL, DEFAULT_MODEL_PATH, IMAGE_DIR, RESULT_DIR
from src.inference.inference_test import load_model, prepare_inputs, select_device


def select_cases(
    cases: list[dict], group_id: str | None, category: str | None,
    condition: str | None, limit: int | None,
) -> list[dict]:
    """저장 순서를 유지하며 선택한 샘플만 추론합니다."""
    selected = cases
    if group_id:
        selected = [case for case in selected if case.get("group_id") == group_id]
    if category:
        category_key = normalize_category(category)
        selected = [
            case for case in selected
            if normalize_category(case.get("class", "")) == category_key
        ]
    if condition:
        selected = [case for case in selected if case.get("condition_code") == condition]
    return selected[:limit] if limit is not None else selected


def yes_no_token_ids(tokenizer) -> tuple[int, int]:
    """공백 없는 Yes/No의 단일 토큰 ID를 얻습니다."""
    yes_id, = tokenizer.encode("Yes", add_special_tokens=False)
    no_id, = tokenizer.encode("No", add_special_tokens=False)
    return yes_id, no_id


@torch.inference_mode()
def infer_case(
    model, processor, case: dict, token_ids: tuple[int, int], max_new_tokens: int,
) -> dict:
    """한 번의 생성에서 답변 원문과 첫 출력 위치의 원시 Yes/No logits를 얻습니다."""
    with ExitStack() as stack:
        images = [stack.enter_context(load_image(IMAGE_DIR / name)) for name in case["image_order"]]
        inputs = prepare_inputs(processor, images, case["prompt"]).to(model.device)
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            num_return_sequences=1,
            temperature=1.0,
            repetition_penalty=1.0,
            return_dict_in_generate=True,
            output_logits=True,
            output_scores=False,
        )
    # 실제로 생성한 토큰 종류와 관계없이 동일한 첫 위치의 두 후보를 측정합니다.
    first_logits = generated.logits[0]
    yes_logit, no_logit = first_logits[0, list(token_ids)].float().cpu().tolist()
    answer_ids = generated.sequences[:, inputs["input_ids"].shape[1]:]
    model_answer = processor.batch_decode(
        answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    return {
        "case_id": case["case_id"],
        "image_order": list(case["image_order"]),
        "model_answer": model_answer,
        "gt": case["gt"],
        "yes_logit": yes_logit,
        "no_logit": no_logit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="로컬 Qwen 모델로 3장 입력을 추론하고 답변·Yes/No logits 저장")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_CASES_JSONL)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="로컬 Qwen2.5-VL 모델 폴더")
    parser.add_argument("--output", type=Path, help="결과 JSONL 경로 (기본: data/inference_results/<입력 파일명>_results.jsonl)")
    parser.add_argument("--group", help="그룹 하나만 추론 (예: group_01_cat)")
    parser.add_argument("--category", help="해당 클래스만 추론 (예: cat)")
    parser.add_argument("--condition", choices=("A", "B", "C", "D"))
    parser.add_argument("--limit", type=int, help="필터 적용 후 앞에서부터 추론할 샘플 수")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = parser.parse_args()

    input_path = args.input_jsonl.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve() if args.output
        else RESULT_DIR / f"{model_path.name}_results.jsonl"
    )
    if output_path == input_path:
        raise ValueError("추론 결과를 입력 JSONL에 덮어쓸 수 없습니다.")
    selected = select_cases(load_cases(input_path), args.group, args.category, args.condition, args.limit)
    if not selected:
        print("추론할 샘플이 없습니다.")
        return

    device = select_device(args.device)
    print(f"입력: {input_path}\n모델: {model_path}\n장치: {device}", flush=True)
    print(f"추론 대상: {len(selected)}개 샘플\n출력: {output_path}", flush=True)
    model, processor = load_model(model_path, device)
    token_ids = yes_no_token_ids(processor.tokenizer)
    print(f"첫 생성 위치의 원시 logits: Yes={token_ids[0]}, No={token_ids[1]}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for case in tqdm(selected, desc="Inference", unit="sample"):
            result = infer_case(model, processor, case, token_ids, args.max_new_tokens)
            output.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
    print(f"완료: {len(selected)}개 결과 저장 → {output_path}")


if __name__ == "__main__":
    main()
