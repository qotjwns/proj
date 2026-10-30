# 공통 추론 실행부. 사용: python src/experiments/b/run.py --model-path data/models/Qwen/Qwen2.5-VL-3B-Instruct

import argparse
import json
from pathlib import Path

from src.common.data import normalize_category, read_jsonl
from src.common.paths import DEFAULT_CASES_JSONL, DEFAULT_MODEL_PATH, project_path, result_path
from src.common.results import C_PROMPT, make_result, read_results


def make_parser(experiment: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"실험 {experiment.upper()}: 케이스별 결과 JSONL 저장")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--cases-jsonl", type=Path, default=DEFAULT_CASES_JSONL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int, help="앞에서부터 지정한 케이스만 실행 (예: 20)")
    parser.add_argument("--group")
    parser.add_argument("--category")
    parser.add_argument("--condition", choices=("A", "B", "C", "D"))
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=32 if experiment == "c" else 8)
    if experiment == "b":
        parser.add_argument("--reuse-jsonl", type=Path, help="같은 모델·설정의 새 공통 형식 B 결과 재사용")
    return parser


def select_cases(args, cases: list[dict]) -> list[dict]:
    if args.group:
        cases = [case for case in cases if case["group_id"] == args.group]
    if args.category:
        category = normalize_category(args.category)
        cases = [case for case in cases if normalize_category(case["class"]) == category]
    if args.condition:
        cases = [case for case in cases if case["condition_code"] == args.condition]
    return cases[:args.limit] if args.limit is not None else cases


def load_cache(path: Path | None, model_path: Path, max_new_tokens: int) -> dict:
    cache = {}
    if path is None:
        return cache
    for row in read_results(path):
        if (
            row["experiment"] != "b"
            or Path(row["model_path"]).resolve() != model_path
            or row["max_new_tokens"] != max_new_tokens
        ):
            raise ValueError("재사용할 B 결과의 모델 또는 생성 설정이 다릅니다.")
        for index, name in enumerate(row["image_order"]):
            key = ((project_path(row["image_dir"]) / name).resolve(), row["prompt"])
            output = {
                "model_answer": row["model_answers"][index],
                "yes_logit": row["yes_logits"][index],
                "no_logit": row["no_logits"][index],
            }
            if key in cache and cache[key] != output:
                raise ValueError(f"같은 이미지·질문의 저장된 예측이 다릅니다: {name}")
            cache[key] = output
    return cache


def run(experiment: str, args) -> None:
    from tqdm import tqdm
    from src.common.model import generate, load_model, select_device

    cases_path = args.cases_jsonl.expanduser().resolve()
    if (args.limit is not None and args.limit < 1) or args.max_new_tokens < 1:
        raise ValueError("limit와 max-new-tokens는 1 이상이어야 합니다.")
    selected = select_cases(args, read_jsonl(cases_path))
    if not selected:
        print("선택한 케이스가 없습니다.")
        return
    model_path = args.model_path.expanduser().resolve()
    scope = "full"
    filters = [value for value in (args.group, args.category, args.condition) if value]
    if filters:
        suffix = "_".join(filters).replace(" ", "_").replace("/", "_")
        scope = f"filtered_{suffix}"
    if args.limit is not None:
        scope = f"pilot_{args.limit}" if scope == "full" else f"{scope}_{args.limit}"
    output_path = (
        args.output.expanduser().resolve() if args.output
        else result_path(experiment, model_path, scope)
    )
    if output_path == cases_path:
        raise ValueError("결과를 원본 케이스 파일에 덮어쓸 수 없습니다.")
    cache = load_cache(args.reuse_jsonl, model_path, args.max_new_tokens) if experiment == "b" else {}
    device = select_device(args.device)
    print(
        f"실험 {experiment.upper()} | {model_path} | {device} | {len(selected)}개\n저장: {output_path}",
        flush=True,
    )
    needs_model = experiment != "b" or any(
        (
            (project_path(case["image_dir"]) / name).resolve(),
            case["prompt"].replace("in all 3 images", "in this image"),
        ) not in cache
        for case in selected for name in case["image_order"]
    )
    model, processor = load_model(model_path, device) if needs_model else (None, None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as destination:
        for case in tqdm(selected, desc=f"Experiment {experiment.upper()}", unit="case"):
            images = [
                (project_path(case["image_dir"]) / name).resolve()
                for name in case["image_order"]
            ]
            if experiment == "b":
                prompt = case["prompt"].replace("in all 3 images", "in this image")
                outputs = []
                for image in images:
                    key = (image, prompt)
                    if key not in cache:
                        cache[key] = generate(model, processor, [image], prompt, args.max_new_tokens)
                    outputs.append(cache[key])
            else:
                prompt = case["prompt"] if experiment == "a" else C_PROMPT.format(object=case["class"])
                outputs = [generate(
                    model, processor, images, prompt, args.max_new_tokens,
                    logits=experiment == "a",
                )]
            row = make_result(case, experiment, model_path, prompt, outputs, args.max_new_tokens)
            destination.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            destination.flush()
            if not row["parse_valid"]:
                tqdm.write(f"형식 실패: {row['case_id']} | {row['model_answers']}")
    print(f"완료: {output_path}")
