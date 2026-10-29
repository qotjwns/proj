"""C 실험: 3장 동시 입력 → 이미지별 답변 → Python AND. 평가·A/B/C 비교도 이 파일에서 수행합니다."""

import argparse
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.data import read_jsonl, write_json
from src.common.paths import (
    DATA_DIR, DEFAULT_CASES_JSONL, DEFAULT_MODEL_PATH, EVALUATION_RESULT_DIR,
    IMAGE_DIR, RESULT_DIR, SINGLE_IMAGE_AND_DIR,
)


PILOT_CASES = 20
MAX_NEW_TOKENS = 32
PROMPT = (
    "For each of the 3 images, determine whether there is at least one {object} in that image. "
    "Answer with exactly three labels (Yes or No) in image order, separated by commas. "
    "Do not provide explanations or a final combined answer."
)


def parse_labels(answer: str) -> list[str] | None:
    """周囲の空白だけ許容し、説明・追加ラベルを含む回答は失敗として扱います。"""
    match = re.fullmatch(r"\s*(Yes|No)\s*,\s*(Yes|No)\s*,\s*(Yes|No)\s*", answer)
    return list(match.groups()) if match else None


def combine_labels(labels: list[str] | None) -> str | None:
    if labels is None:
        return None
    if len(labels) != 3 or any(label not in ("Yes", "No") for label in labels):
        raise ValueError("AND 입력은 Yes/No 예측 3개여야 합니다.")
    return "Yes" if all(label == "Yes" for label in labels) else "No"


def image_targets(case: dict) -> list[str]:
    return ["Yes" if label == "P" else "No" for label in case["image_labels"]]


def validate_cases(cases: list[dict]) -> None:
    if len(cases) != 800 or len({case["case_id"] for case in cases}) != 800:
        raise ValueError("중복 없는 원본 800개 케이스가 필요합니다.")
    for case in cases:
        if len(case["image_order"]) != 3 or len(case["image_labels"]) != 3:
            raise ValueError(f"{case['case_id']}: 이미지와 라벨이 각각 3개여야 합니다.")
        if any(label not in ("P", "N") for label in case["image_labels"]):
            raise ValueError(f"{case['case_id']}: P/N 라벨이 필요합니다.")
        if combine_labels(image_targets(case)) != case["gt"]:
            raise ValueError(f"{case['case_id']}: 이미지 라벨과 최종 정답이 다릅니다.")


def align_results(cases: list[dict], records: list[dict]) -> list[dict]:
    """같은 전체 케이스인지 확인하고 원본 순서로 정렬합니다."""
    indexed = {record["case_id"]: record for record in records}
    if len(indexed) != len(records) or set(indexed) != {case["case_id"] for case in cases}:
        raise ValueError("결과에 중복·누락·추가 case_id가 있습니다.")
    ordered = [indexed[case["case_id"]] for case in cases]
    for case, record in zip(cases, ordered):
        if record["image_order"] != case["image_order"] or record["gt"] != case["gt"]:
            raise ValueError(f"{case['case_id']}: 이미지 순서 또는 정답이 원본과 다릅니다.")
    return ordered


def make_record(case: dict, answer: str) -> dict:
    labels = parse_labels(answer)
    prediction = combine_labels(labels)
    targets = image_targets(case)
    correct = [label == target for label, target in zip(labels, targets)] if labels else None
    return {
        "case_id": case["case_id"], "group_id": case["group_id"], "class": case["class"],
        "image_order": case["image_order"], "image_gt": targets,
        "prompt": PROMPT.format(object=case["class"]), "model_answer": answer,
        "parse_valid": labels is not None, "image_predictions": labels,
        "image_correct": correct, "all_images_correct": all(correct) if correct is not None else False,
        "and_prediction": prediction, "gt": case["gt"], "correct": prediction == case["gt"],
    }


def evaluate_predictions(cases: list[dict], predictions: list, image_predictions: list | None = None) -> dict:
    """형식 실패도 분모에 포함합니다. 이미지 지표는 케이스 내 출현 횟수 기준입니다."""
    total = len(cases)
    if not total or len(predictions) != total or (image_predictions is not None and len(image_predictions) != total):
        raise ValueError("케이스와 예측 개수가 같고 1개 이상이어야 합니다.")
    correct = sum(prediction == case["gt"] for case, prediction in zip(cases, predictions))
    invalid = sum(prediction is None for prediction in predictions)
    image_correct = all_correct = 0
    if image_predictions is not None:
        for case, labels in zip(cases, image_predictions):
            if labels is None:
                continue
            matches = [label == target for label, target in zip(labels, image_targets(case))]
            image_correct += sum(matches)
            all_correct += all(matches)
    return {
        "cases": total, "valid_answers": total - invalid, "invalid_answers": invalid,
        "valid_answer_rate": (total - invalid) / total,
        "image_accuracy": image_correct / (3 * total) if image_predictions is not None else None,
        "all_images_correct_rate": all_correct / total if image_predictions is not None else None,
        "accuracy": correct / total,
        "false_yes": sum(case["gt"] == "No" and pred == "Yes" for case, pred in zip(cases, predictions)),
        "false_no": sum(case["gt"] == "Yes" and pred == "No" for case, pred in zip(cases, predictions)),
    }


def infer_labels(model, processor, case: dict) -> str:
    import torch
    from src.common.images import load_image
    from src.inference.inference_test import prepare_inputs

    # A와 같은 이미지 로더·전처리·입력 순서·greedy 설정을 사용합니다.
    with torch.inference_mode(), ExitStack() as stack:
        images = [stack.enter_context(load_image(IMAGE_DIR / name)) for name in case["image_order"]]
        inputs = prepare_inputs(processor, images, PROMPT.format(object=case["class"])).to(model.device)
        generated = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False, num_beams=1, num_return_sequences=1,
            temperature=1.0, repetition_penalty=1.0,
            return_dict_in_generate=False, output_logits=False, output_scores=False,
        )
    answer_ids = generated[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def run_inference(cases: list[dict], model_path: Path, mode: str) -> None:
    from tqdm import tqdm
    from src.inference.inference_test import load_model, select_device

    selected = cases[:PILOT_CASES] if mode == "pilot" else cases
    folder = f"pilot_{PILOT_CASES}" if mode == "pilot" else "full"
    output_dir = DATA_DIR / "joint_image_and" / model_path.name / folder
    device = select_device()
    print(f"C | {mode} | {len(selected)}개 | {device}\n모델: {model_path}\n저장: {output_dir}", flush=True)
    model, processor = load_model(model_path, device)
    records = []
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as output:
        for case in tqdm(selected, desc="C: joint labels + AND", unit="case"):
            record = make_record(case, infer_labels(model, processor, case))
            record.update(model_path=str(model_path), max_new_tokens=MAX_NEW_TOKENS)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            records.append(record)
            if mode == "pilot" or not record["parse_valid"]:
                tqdm.write(f"{case['case_id']} | {record['model_answer']!r} → {record['image_predictions']}")
    metrics = evaluate_predictions(
        selected, [row["and_prediction"] for row in records], [row["image_predictions"] for row in records],
    )
    write_json({
        "experiment": "C", "mode": mode, "model_path": str(model_path),
        "cases_path": str(DEFAULT_CASES_JSONL), "prompt_template": PROMPT,
        "max_new_tokens": MAX_NEW_TOKENS,
        "accuracy_definition": "Python AND accuracy; invalid answers count as incorrect",
        **metrics,
    }, output_dir / "metrics.json")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def compare(cases: list[dict], a_rows: list[dict], b_rows: list[dict], c_rows: list[dict]) -> list[dict]:
    from src.evaluation.evaluate_results import normalize_answer

    a_rows = align_results(cases, a_rows)
    b_rows = align_results(cases, [
        {**row, "image_order": [row[f"image_{index}"] for index in (1, 2, 3)]} for row in b_rows
    ])
    c_rows = align_results(cases, c_rows)
    a_predictions = [normalize_answer(row["model_answer"]) for row in a_rows]
    b_labels = [[normalize_answer(row[f"single_prediction_{index}"]) for index in (1, 2, 3)] for row in b_rows]
    b_predictions = [combine_labels(labels) for labels in b_labels]
    c_labels = [parse_labels(row["model_answer"]) for row in c_rows]
    c_predictions = [combine_labels(labels) for labels in c_labels]
    for case, a_pred, b_pred, b_row, c_row in zip(cases, a_predictions, b_predictions, b_rows, c_rows):
        if normalize_answer(b_row["original_prediction"]) != a_pred:
            raise ValueError(f"{case['case_id']}: B가 참조한 기존 예측과 A 결과가 다릅니다.")
        if normalize_answer(b_row["and_prediction"]) != b_pred:
            raise ValueError(f"{case['case_id']}: B의 저장된 AND와 개별 예측이 다릅니다.")
        if c_row["prompt"] != PROMPT.format(object=case["class"]):
            raise ValueError(f"{case['case_id']}: C의 프롬프트가 확정된 문구와 다릅니다.")
    return [
        {"method": "A: joint final answer", **evaluate_predictions(cases, a_predictions)},
        {"method": "B: single images + AND", **evaluate_predictions(cases, b_predictions, b_labels)},
        {"method": "C: joint labels + AND", **evaluate_predictions(cases, c_predictions, c_labels)},
    ]


def save_table(rows: list[dict], model_name: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def percentage(value):
        return "N/A" if value is None else f"{value:.2%}"

    cells = [[
        row["method"], str(row["cases"]), percentage(row["valid_answer_rate"]),
        percentage(row["image_accuracy"]), percentage(row["all_images_correct_rate"]),
        percentage(row["accuracy"]), str(row["false_yes"]), str(row["false_no"]), str(row["invalid_answers"]),
    ] for row in rows]
    fig, axis = plt.subplots(figsize=(17, 4))
    try:
        axis.set_axis_off()
        axis.set_title(f"{model_name} | A / B / C | Same {rows[0]['cases']} cases", fontsize=15, pad=18)
        table = axis.table(
            cellText=cells,
            colLabels=["Method", "Cases", "Valid", "Image acc.", "All 3 correct", "Final acc.", "False-Yes", "False-No", "Invalid"],
            colWidths=[0.25, 0.06, 0.09, 0.10, 0.12, 0.11, 0.09, 0.09, 0.09],
            cellLoc="center", bbox=[0, 0.25, 1, 0.65],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        fig.text(
            0.5, 0.05,
            "Final accuracy: A's answer; Python AND for B/C. A has no per-image predictions.\n"
            "Invalid C outputs count as incorrect for all three images and the case; they are separate from False-Yes/False-No.\n"
            "Image metrics count case-image occurrences. Different prompts prevent isolating a single cause.",
            ha="center", fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.15, 1, 1))
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)


def save_comparison(cases: list[dict], model_path: Path) -> None:
    sources = {
        "A": RESULT_DIR / f"{model_path.name}_results.jsonl",
        "B": SINGLE_IMAGE_AND_DIR / model_path.name / "and_results.jsonl",
        "C": DATA_DIR / "joint_image_and" / model_path.name / "full" / "results.jsonl",
    }
    rows = compare(cases, read_jsonl(sources["A"]), read_jsonl(sources["B"]), read_jsonl(sources["C"]))
    output_dir = EVALUATION_RESULT_DIR / model_path.name / "abc"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json({"sources": {key: str(path) for key, path in sources.items()}, "metrics": rows}, output_dir / "comparison.json")
    with (output_dir / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save_table(rows, model_path.name, output_dir / "comparison.png")
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"저장: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="C 실험: 3장 동시 입력의 개별 답변 + Python AND")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="로컬 Qwen2.5-VL 모델 폴더 (기본: 3B)")
    parser.add_argument("--mode", choices=("pilot", "full", "compare"), default="pilot", help="20개 확인 / 전체 800개 / 저장된 A·B·C 비교")
    args = parser.parse_args()
    model_path = args.model_path.expanduser().resolve()
    cases = read_jsonl(DEFAULT_CASES_JSONL)
    validate_cases(cases)
    if args.mode == "compare":
        save_comparison(cases, model_path)
    else:
        run_inference(cases, model_path, args.mode)


if __name__ == "__main__":
    main()
