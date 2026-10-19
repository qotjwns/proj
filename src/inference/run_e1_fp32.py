# .venv/bin/python src/inference/run_e1_fp32.py configs/model.json
# 기존 E1 결과에 없는 샘플만 FP32로 추론하고 별도 파일에 이어서 저장합니다.
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
import transformers
from transformers import AutoProcessor, GenerationConfig, Qwen2_5_VLForConditionalGeneration

from run_e1 import IMAGE_SIZE, PROJECT_ROOT, build_evaluations, predict, prepare_inputs


def read_trial_ids(path):
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as source:
        return {json.loads(line)["trial_id"] for line in source if line.strip()}


def main():
    parser = argparse.ArgumentParser(description="E1 미완료 샘플만 FP32로 재추론합니다.")
    parser.add_argument("config", type=Path, help="모델 설정 JSON 경로")
    args = parser.parse_args()
    config = json.loads(args.config.expanduser().read_text(encoding="utf-8"))
    model_path = (PROJECT_ROOT / Path(config["save_dir"]).expanduser()).resolve()
    log_dir = PROJECT_ROOT / "logs"
    source_meta_bytes = (log_dir / "baseline_e1_meta.json").read_bytes()
    source_meta = json.loads(source_meta_bytes)
    data_bytes = (PROJECT_ROOT / source_meta["dataset"]).read_bytes()

    # 원본 실행의 질문과 생성 설정을 사용해 정밀도 이외의 조건을 유지합니다.
    comparison_settings = {
        "model": config,
        "dataset_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "image_preprocessing": {
            "method": "aspect-ratio-preserving resize with white padding",
            "size": [IMAGE_SIZE, IMAGE_SIZE],
            "processor_do_resize": False,
        },
    }
    changed = [key for key, value in comparison_settings.items() if source_meta[key] != value]
    if changed:
        print(f"원본 실행과 조건이 달라 재추론하지 않았습니다: {', '.join(changed)}", flush=True)
        return

    evaluations = build_evaluations(json.loads(data_bytes), source_meta["prompt_config"])
    targets = {}
    completed_by_template = {}
    source_result_hashes = {}
    for template_id, trials in evaluations.items():
        source_path = log_dir / f"baseline_e1_{template_id}.jsonl"
        source_bytes = source_path.read_bytes() if source_path.exists() else b""
        source_ids = {json.loads(line)["trial_id"] for line in source_bytes.splitlines() if line.strip()}
        source_result_hashes[template_id] = hashlib.sha256(source_bytes).hexdigest()
        targets[template_id] = [sample for sample in trials if sample["trial_id"] not in source_ids]
        result_path = log_dir / f"baseline_e1_{template_id}_fp32_retry.jsonl"
        completed = read_trial_ids(result_path)
        completed_by_template[template_id] = completed
        pending_count = sum(sample["trial_id"] not in completed for sample in targets[template_id])
        print(
            f"[{template_id}] 원본 미완료 {len(targets[template_id])}개 / "
            f"FP32 남은 추론 {pending_count}개", flush=True,
        )

    total_targets = sum(len(trials) for trials in targets.values())
    device = source_meta["device"]
    dtype = torch.float32
    metadata = {
        **source_meta,
        "dtype": str(dtype),
        "source_dtype": source_meta["dtype"],
        "source_meta_sha256": hashlib.sha256(source_meta_bytes).hexdigest(),
        "source_results_sha256": source_result_hashes,
        "group_count": len({s["group_id"] for trials in targets.values() for s in trials}),
        "planned_trials": total_targets,
        "trials_per_template": {key: len(value) for key, value in targets.items()},
        "target_trial_ids": [s["trial_id"] for trials in targets.values() for s in trials],
    }
    meta_path = log_dir / "baseline_e1_fp32_retry_meta.json"
    if meta_path.exists():
        previous_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        changed = [key for key, value in metadata.items() if previous_meta.get(key) != value]
        if changed:
            print(f"이전 FP32 실행과 조건이 달라 재개하지 않았습니다: {', '.join(changed)}", flush=True)
            return

    pending_total = sum(
        sample["trial_id"] not in completed_by_template[template_id]
        for template_id, trials in targets.items() for sample in trials
    )
    if pending_total == 0:
        print("최종 추론 미완료: 0개", flush=True)
        return
    if (device == "mps" and not torch.backends.mps.is_available()) or (
        device.startswith("cuda") and not torch.cuda.is_available()
    ):
        print(f"원본과 같은 장치 {device}를 사용할 수 없어 재추론하지 않았습니다.", flush=True)
        return

    torch.manual_seed(source_meta["seed"])
    print(f"재추론: {pending_total}개 / 장치: {device} / 정밀도: {dtype}", flush=True)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    yes_id, = processor.tokenizer.encode("Yes", add_special_tokens=False)
    no_id, = processor.tokenizer.encode("No", add_special_tokens=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
        attn_implementation=source_meta["attn_implementation"],
    ).to(device).eval()
    generation_config = GenerationConfig(**source_meta["generation"])
    if not meta_path.exists():
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    incomplete_total = 0
    for template_id, trials in targets.items():
        if not trials:
            continue
        result_path = log_dir / f"baseline_e1_{template_id}_fp32_retry.jsonl"
        failed_path = log_dir / f"baseline_e1_{template_id}_fp32_retry_failed.jsonl"
        completed = completed_by_template[template_id]
        with result_path.open("a", encoding="utf-8") as output:
            for index, sample in enumerate(trials, start=1):
                if sample["trial_id"] in completed:
                    continue
                inputs = prepare_inputs(processor, sample["images"], sample["question"]).to(device)
                prediction = predict(model, processor, inputs, generation_config, yes_id, no_id)
                invalid_values = {
                    key: repr(value) for key, value in prediction.items()
                    if isinstance(value, float) and not math.isfinite(value)
                }
                if invalid_values:
                    failure = {
                        "sample_id": sample["sample_id"],
                        "trial_id": sample["trial_id"],
                        "template_id": template_id,
                        "dtype": str(dtype),
                        "reason": "non_finite_scores",
                        "invalid_values": invalid_values,
                    }
                    with failed_path.open("a", encoding="utf-8") as failed:
                        failed.write(json.dumps(failure, ensure_ascii=False, allow_nan=False) + "\n")
                    print(f"[{template_id} FP32 {index}/{len(trials)}] {sample['sample_id']} 수치 오류로 건너뜀", flush=True)
                    continue
                correct = prediction["prediction"] == sample["answer"]
                base_prediction = prediction["prediction"]
                if sample["answer_inverted"] and base_prediction is not None:
                    base_prediction = {"Yes": "No", "No": "Yes"}[base_prediction]
                result = {
                    "sample_id": sample["sample_id"],
                    "trial_id": sample["trial_id"],
                    "template_id": template_id,
                    "task": sample["task"],
                    "group_id": sample["group_id"],
                    "class": sample["class"],
                    "num_images": sample["num_images"],
                    "images": sample["images"],
                    "question": sample["question"],
                    **prediction,
                    "answer": sample["answer"],
                    "answer_inverted": sample["answer_inverted"],
                    "base_answer": sample["base_answer"],
                    "base_prediction": base_prediction,
                    "correct": correct,
                    "dtype": str(dtype),
                }
                output.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                completed.add(sample["trial_id"])
                print(
                    f"[{template_id} FP32 {index}/{len(trials)}] {sample['sample_id']} "
                    f"예측={prediction['prediction']} 정답={sample['answer']}", flush=True,
                )
        incomplete_total += sum(sample["trial_id"] not in completed for sample in trials)
    print(f"최종 추론 미완료: {incomplete_total}개", flush=True)


if __name__ == "__main__":
    main()
