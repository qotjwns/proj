# 전체 800개 Single + AND: python src/inference/run_single_image_and.py
# 기존 단일 이미지 결과 재사용:
# python src/inference/run_single_image_and.py --reuse-jsonl data/single_image_results/Qwen2.5-VL-3B-Instruct_all_single_image_results.jsonl
# 저장 위치: data/single_image_and/<모델명>/

import argparse
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from tqdm import tqdm

from src.common.data import read_jsonl
from src.common.paths import DEFAULT_CASES_JSONL, DEFAULT_MODEL_PATH, RESULT_DIR, SINGLE_IMAGE_AND_DIR
from src.evaluation.compare_single_image_and import combine_predictions, save_case_csv
from src.evaluation.evaluate_composition import evaluate_composition, index_cases
from src.evaluation.evaluate_results import evaluate_results, normalize_answer
from src.evaluation.evaluate_single_image_results import draw_table
from src.inference.inference_test import load_model, select_device
from src.inference.run_inference import infer_case, yes_no_token_ids
from src.inference.run_single_image_errors import build_single_image_tasks


def prepare_all_tasks(results: list[dict], cases: list[dict], expected_cases: int) -> list[dict]:
    """기존 정답 여부로 필터링하지 않고 전체 케이스의 단일 이미지 작업을 만듭니다."""
    cases_by_id = index_cases(cases)
    evaluate_composition(results, cases_by_id)
    result_ids = {result["case_id"] for result in results}
    if result_ids != set(cases_by_id):
        raise ValueError("원본 cases와 기존 추론 결과의 case_id 목록이 다릅니다. 전체 결과를 지정하세요.")
    if len(results) != expected_cases:
        raise ValueError(f"전체 {expected_cases}개 케이스가 필요하지만 {len(results)}개입니다.")
    selected = []
    for result in results:
        target = normalize_answer(result["gt"])
        answer = normalize_answer(result["model_answer"])
        error_type = None if target == answer else ("false-yes" if answer == "Yes" else "false-no")
        if target == "Yes":
            outcome = "TP" if answer == "Yes" else "FN"
        else:
            outcome = "FP" if answer == "Yes" else "TN"
        selected.append({
            "case": cases_by_id[result["case_id"]], "source_result": result,
            "error_type": error_type, "source_outcome": outcome,
        })
    return build_single_image_tasks(selected)


def load_cached_predictions(
    path: Path | None, model_path: Path, tasks: list[dict], max_new_tokens: int,
) -> dict:
    """같은 모델·이미지·질문 결과만 재사용하고 정답이나 예측의 모순을 확인합니다."""
    if path is None:
        return {}
    expected = {task["cache_key"]: task["gt"] for task in tasks}
    cache = {}
    for index, record in enumerate(read_jsonl(path), start=1):
        if not isinstance(record, dict):
            raise ValueError(f"재사용 파일 {index}행은 JSON 객체여야 합니다.")
        for field in ("model_path", "image_path", "prompt"):
            if not isinstance(record.get(field), str) or not record[field]:
                raise ValueError(f"재사용 파일 {index}행에 {field}가 없습니다.")
        if Path(record["model_path"]).expanduser().resolve() != model_path:
            raise ValueError(f"재사용 파일 {index}행의 모델 경로가 현재 모델과 다릅니다.")
        key = (Path(record["image_path"]).expanduser().resolve(), record["prompt"])
        if key not in expected:
            continue
        if normalize_answer(record.get("gt")) != expected[key]:
            raise ValueError(f"재사용 파일 {index}행의 개별 이미지 정답이 다릅니다.")
        if normalize_answer(record.get("model_answer")) is None:
            # 유효한 답변이 없는 캐시는 다시 추론합니다.
            continue
        stored_limit = record.get("max_new_tokens")
        if stored_limit is not None and stored_limit != max_new_tokens:
            raise ValueError(f"재사용 파일 {index}행의 max_new_tokens가 현재 설정과 다릅니다.")
        for field in ("yes_logit", "no_logit"):
            value = record.get(field)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"재사용 파일 {index}행의 {field}가 유효하지 않습니다.")
        prediction = {field: record[field] for field in ("model_answer", "yes_logit", "no_logit")}
        prediction["max_new_tokens"] = stored_limit
        if key in cache and any(cache[key][field] != prediction[field] for field in ("model_answer", "yes_logit", "no_logit")):
            raise ValueError(f"같은 이미지·질문의 재사용 결과가 다릅니다: {key[0]}")
        cache[key] = prediction
    return cache


def make_single_record(task: dict, prediction: dict, input_path: Path, model_path: Path, reuse_path: Path | None) -> dict:
    case = task["case"]
    answer = normalize_answer(prediction["model_answer"])
    return {
        "case_id": case["case_id"], "group_id": case["group_id"], "class": case["class"],
        "source_outcome": task["source_outcome"], "source_error_type": task["error_type"],
        "source_results_path": str(input_path), "source_prompt": case["prompt"],
        "source_result": task["source_result"], "model_path": str(model_path),
        "image_position": task["image_position"], "image_order": [task["image_name"]],
        "image_path": str(task["image_path"]), "image_label": task["image_label"],
        "prompt": task["prompt"], "gt": task["gt"], **prediction,
        "normalized_answer": answer, "is_correct": answer == task["gt"] if answer is not None else None,
        "reused_from": str(reuse_path) if reuse_path else None,
    }


def make_comparison(tasks: list[dict], predictions: list[dict]) -> dict:
    """예측만 AND로 결합한 뒤 원래 케이스의 정답과 비교합니다."""
    if len(tasks) != 3 or len(predictions) != 3 or [task["image_position"] for task in tasks] != [1, 2, 3]:
        raise ValueError("한 케이스의 이미지 위치 1·2·3이 필요합니다.")
    case = tasks[0]["case"]
    source = tasks[0]["source_result"]
    if any(task["case"]["case_id"] != case["case_id"] for task in tasks):
        raise ValueError("서로 다른 케이스의 이미지를 결합할 수 없습니다.")
    answers = [normalize_answer(prediction["model_answer"]) for prediction in predictions]
    final_answer = combine_predictions(answers)
    target = normalize_answer(source["gt"])
    original_answer = normalize_answer(source["model_answer"])
    original_correct = original_answer == target
    and_correct = final_answer == target
    return {
        "case_id": case["case_id"], "class": case["class"],
        "composition": "".join(case["image_labels"]), "source_outcome": tasks[0]["source_outcome"],
        **{f"image_{index}": task["image_name"] for index, task in enumerate(tasks, start=1)},
        **{f"single_prediction_{index}": answer for index, answer in enumerate(answers, start=1)},
        "gt": target, "original_prediction": original_answer, "and_prediction": final_answer,
        "original_correct": original_correct, "and_correct": and_correct,
        "transition": f"{'correct' if original_correct else 'wrong'}_to_{'correct' if and_correct else 'wrong'}",
    }


def save_comparison_plot(comparisons: list[dict], model_name: str, output_path: Path) -> None:
    metrics_by_method = [evaluate_results([
        {"gt": row["gt"], "model_answer": row[key]} for row in comparisons
    ]) for key in ("original_prediction", "and_prediction")]
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in (
        "Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR",
    ) if name in available), None)

    def text(korean: str, english: str) -> str:
        return korean if font else english

    with plt.rc_context({"font.family": font} if font else {}):
        fig, axes = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={"height_ratios": [1.2, 1.3]})
        try:
            fig.suptitle(
                f"{model_name} | Single + Python AND\n"
                + text(f"전체 {len(comparisons)}개 케이스: 기존 정답·오답 모두 포함", f"All {len(comparisons)} cases: original correct and incorrect cases included"),
                fontsize=16,
            )
            rows = []
            for label, metrics in zip(
                [text("기존: 3장 동시 입력", "Original: joint inference"), text("개별 이미지 + Python AND", "Single images + Python AND")],
                metrics_by_method,
            ):
                rows.append([label, str(metrics["total_samples"])] + [
                    f"{metrics[key]:.2%}" for key in ("accuracy", "precision", "recall", "f1", "yes_response_rate")
                ])
            draw_table(
                axes[0], text("1. 같은 전체 케이스에서의 성능 비교 (양성: Yes)", "1. Metrics on the same full set (positive: Yes)"),
                [text("방법", "Method"), text("케이스 수", "Cases"), "Accuracy", "Precision", "Recall", "F1", "Yes rate"],
                rows, [0.28, 0.08, 0.128, 0.128, 0.128, 0.128, 0.128], bottom=0.28,
            )
            lines = []
            for label, metrics in zip([text("기존", "Original"), "AND"], metrics_by_method):
                counts = metrics["confusion_matrix"]
                lines.append(label + ": " + " / ".join(f"{key} {counts[key]}" for key in ("TP", "TN", "FP", "FN")))
            axes[0].text(0, 0.04, "\n".join(lines), transform=axes[0].transAxes, fontsize=10, color="#52606d")
            transitions = []
            for key, label in (
                ("correct_to_correct", text("기존 정답 → AND 정답", "Correct to correct")),
                ("correct_to_wrong", text("기존 정답 → AND 오답", "Correct to wrong")),
                ("wrong_to_correct", text("기존 오답 → AND 정답", "Wrong to correct")),
                ("wrong_to_wrong", text("기존 오답 → AND 오답", "Wrong to wrong")),
            ):
                count = sum(row["transition"] == key for row in comparisons)
                transitions.append([label, str(count), f"{count / len(comparisons):.2%}"])
            draw_table(
                axes[1], text("2. 케이스별 정오 변화", "2. Changes in case correctness"),
                [text("변화", "Transition"), text("케이스 수", "Cases"), text("전체 대비 비율", "Share of all cases")],
                transitions, [0.56, 0.20, 0.24],
            )
            fig.text(
                0.5, 0.04,
                text(
                    "AND: 개별 예측 3개가 모두 Yes일 때만 Yes. 정답 라벨과 기존 예측은 AND 계산에 사용하지 않습니다.\n"
                    "동일 모델·이미지·질문은 재사용하되, 각 원본 케이스의 3개 예측을 결합해 케이스당 한 번 평가합니다.",
                    "AND returns Yes only when all three predictions are Yes. Ground truth and original predictions are not inputs to AND.\n"
                    "Identical model/image/question results are reused; each original case is evaluated once after combining its three predictions.",
                ), ha="center", fontsize=10, color="#52606d",
            )
            fig.tight_layout(rect=(0.02, 0.14, 0.98, 0.90), h_pad=2.5)
            fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
        finally:
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="전체 800개 케이스의 단일 이미지 추론 + Python AND + 기존 결과 비교")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input-jsonl", type=Path, help="기존 3장 입력 결과 (기본: 선택 모델의 결과 파일)")
    parser.add_argument("--cases-jsonl", type=Path, default=DEFAULT_CASES_JSONL)
    parser.add_argument("--expected-cases", type=int, default=800, help="전체 케이스 수 검증 (기본: 800)")
    parser.add_argument("--reuse-jsonl", type=Path, help="같은 모델·생성 설정의 기존 단일 이미지 결과 (선택)")
    parser.add_argument("--output-dir", type=Path, help="기본: data/single_image_and/<모델명>")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    if args.expected_cases < 1 or args.max_new_tokens < 1:
        parser.error("--expected-cases와 --max-new-tokens는 1 이상이어야 합니다.")
    model_path = args.model_path.expanduser().resolve()
    input_path = args.input_jsonl.expanduser().resolve() if args.input_jsonl else RESULT_DIR / f"{model_path.name}_results.jsonl"
    cases_path = args.cases_jsonl.expanduser().resolve()
    reuse_path = args.reuse_jsonl.expanduser().resolve() if args.reuse_jsonl else None
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else SINGLE_IMAGE_AND_DIR / model_path.name
    single_path = output_dir / "single_image_results.jsonl"
    and_path = output_dir / "and_results.jsonl"
    csv_path = output_dir / "comparison.csv"
    png_path = output_dir / "comparison.png"
    outputs = {single_path, and_path, csv_path, png_path}
    if input_path in outputs or cases_path in outputs or (reuse_path in outputs and reuse_path != single_path):
        parser.error("입력 파일을 출력 파일로 덮어쓸 수 없습니다.")
    try:
        tasks = prepare_all_tasks(read_jsonl(input_path), read_jsonl(cases_path), args.expected_cases)
        cache = load_cached_predictions(reuse_path, model_path, tasks, args.max_new_tokens)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    reused_keys = set(cache)
    unique_keys = {task["cache_key"] for task in tasks}
    print(f"전체 {len(tasks) // 3}개 케이스 / 이미지별 결과 {len(tasks)}행", flush=True)
    print(f"고유 이미지·질문 {len(unique_keys)}개 / 재사용 {len(cache)}개 / 새 추론 {len(unique_keys - reused_keys)}개", flush=True)
    model = processor = token_ids = None
    if unique_keys - reused_keys:
        device = select_device(args.device)
        print(f"모델: {model_path}\n장치: {device}", flush=True)
        model, processor = load_model(model_path, device)
        token_ids = yes_no_token_ids(processor.tokenizer)

    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = []
    with single_path.open("w", encoding="utf-8") as single_output, and_path.open("w", encoding="utf-8") as and_output:
        for start in tqdm(range(0, len(tasks), 3), desc="Single + AND", unit="case"):
            case_tasks = tasks[start:start + 3]
            predictions = []
            for task in case_tasks:
                key = task["cache_key"]
                if key not in cache:
                    single_case = {
                        "case_id": task["case"]["case_id"], "image_order": [task["image_name"]],
                        "prompt": task["prompt"], "gt": task["gt"],
                    }
                    result = infer_case(model, processor, single_case, token_ids, args.max_new_tokens, image_dir=task["image_dir"])
                    cache[key] = {field: result[field] for field in ("model_answer", "yes_logit", "no_logit")}
                    cache[key]["max_new_tokens"] = args.max_new_tokens
                prediction = cache[key]
                record = make_single_record(task, prediction, input_path, model_path, reuse_path if key in reused_keys else None)
                single_output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                single_output.flush()
                predictions.append(prediction)
            try:
                comparison = make_comparison(case_tasks, predictions)
            except ValueError as error:
                raise ValueError(f"{case_tasks[0]['case']['case_id']}: {error} 개별 이미지의 원문 답변은 저장했습니다.") from error
            and_output.write(json.dumps(comparison, ensure_ascii=False, allow_nan=False) + "\n")
            and_output.flush()
            comparisons.append(comparison)
    if len(comparisons) != args.expected_cases:
        raise ValueError("전체 케이스의 AND 예측이 완료되지 않았습니다.")
    save_case_csv(comparisons, csv_path)
    save_comparison_plot(comparisons, model_path.name, png_path)
    print(f"완료: {single_path}\n{and_path}\n{csv_path}\n{png_path}")


if __name__ == "__main__":
    main()
