# .venv/bin/python src/inference/run_e1.py configs/model.json
# E1 전체 그룹을 configs/e1.json에서 선택한 템플릿으로 추론해 저장합니다.
# 기존 결과는 유지하고, 아직 저장되지 않은 샘플만 추론합니다.
import argparse
import hashlib
import json
import math
from pathlib import Path
from PIL import Image, ImageOps
import torch
import transformers
from transformers import AutoProcessor, GenerationConfig, Qwen2_5_VLForConditionalGeneration

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data/eval_e1.json"
PROMPT_CONFIG_PATH = PROJECT_ROOT / "configs/e1.json"
IMAGE_SIZE = 448
MAX_NEW_TOKENS = 8
SEED = 42


def build_evaluations(samples, prompt_config):
    templates = [
        template for template in prompt_config["templates"]
        if template["template_id"] in prompt_config["enabled_templates"]
    ]
    single_samples = {}
    if any(template["task"] == "single_image" for template in templates):
        for sample in samples:
            for image_path, label in zip(sample["images"], sample["image_labels"]):
                key = (sample["class"], image_path)
                single_samples.setdefault(key, {
                    "sample_id": f"{sample['group_id']}_single_{Path(image_path).stem}",
                    "group_id": sample["group_id"],
                    "class": sample["class"],
                    "images": [image_path],
                    "image_labels": [label],
                    "answer": "Yes" if label else "No",
                })

    evaluations = {}
    for template in templates:
        template_id = template["template_id"]
        source = single_samples.values() if template["task"] == "single_image" else samples
        trials = []
        for sample in source:
            num_images = len(sample["images"])
            question = template["question"].format(
                N=num_images, **prompt_config["objects"][sample["class"]]
            ).strip()
            answer = sample["answer"]
            if template["invert_answer"]:
                answer = {"Yes": "No", "No": "Yes"}[answer]
            trials.append({
                **sample,
                "trial_id": f"{sample['sample_id']}__{template_id}",
                "template_id": template_id,
                "task": template["task"],
                "num_images": num_images,
                "question": question,
                "base_answer": sample["answer"],
                "answer": answer,
                "answer_inverted": template["invert_answer"],
            })
        evaluations[template_id] = trials
    return evaluations


def prepare_inputs(processor, image_paths, question):
    images = []
    for image_path in image_paths:
        with Image.open(PROJECT_ROOT / image_path) as source:
            images.append(ImageOps.pad(
                source.convert("RGB"),
                (IMAGE_SIZE, IMAGE_SIZE),
                method=Image.Resampling.LANCZOS,
                color="white",
            ))

    content = [{"type": "image"} for _ in images]
    content.append({"type": "text", "text": question.strip()})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return processor(
        text=[text], images=images, do_resize=False, return_tensors="pt"
    )

@torch.inference_mode()
def predict(model, processor, inputs, generation_config, yes_id, no_id):
    outputs = model.generate(
        **inputs,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_logits=True,
    )
    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs.sequences[0, input_length:]
    raw_output = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
    prediction = {"yes": "Yes", "no": "No"}.get(raw_output.strip().rstrip(".!").lower())
    first_logits = outputs.logits[0][0].float()
    answer_logits = first_logits[[yes_id, no_id]]

    return {
        "raw_output": raw_output,
        "prediction": prediction,
        "yes_logit": answer_logits[0].item(),
        "no_logit": answer_logits[1].item(),
        "p_yes_given_yes_no": answer_logits.softmax(dim=0)[0].item(),
        "input_token_count": input_length,
        "image_token_count": (inputs["input_ids"] == model.config.image_token_id).sum().item(),
    }

def main():
    parser = argparse.ArgumentParser(description="E1 전체 그룹을 템플릿별로 추론합니다.")
    parser.add_argument("config", type=Path, help="모델 설정 JSON 경로")
    args = parser.parse_args()
    config = json.loads(args.config.expanduser().read_text(encoding="utf-8"))
    model_path = (PROJECT_ROOT / Path(config["save_dir"]).expanduser()).resolve()
    data_bytes = DATA_PATH.read_bytes()
    samples = json.loads(data_bytes)
    prompt_bytes = PROMPT_CONFIG_PATH.read_bytes()
    prompt_config = json.loads(prompt_bytes)
    evaluations = build_evaluations(samples, prompt_config)
    group_count = len({sample["group_id"] for sample in samples})
    total_trials = sum(len(trials) for trials in evaluations.values())
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    meta_path = log_dir / "baseline_e1_meta.json"
    previous_metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else None
    completed_by_template = {}
    for template_id, trials in evaluations.items():
        result_path = log_dir / f"baseline_e1_{template_id}.jsonl"
        completed = set()
        if result_path.exists():
            with result_path.open(encoding="utf-8") as saved:
                completed = {json.loads(line)["trial_id"] for line in saved if line.strip()}
        completed_by_template[template_id] = completed
        pending_count = sum(sample["trial_id"] not in completed for sample in trials)
        print(f"[{template_id}] 기존 결과 {len(trials) - pending_count}개 / 남은 추론 {pending_count}개", flush=True)
    pending_total = sum(
        sample["trial_id"] not in completed_by_template[template_id]
        for template_id, trials in evaluations.items() for sample in trials
    )

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    dtype = torch.float32 if device == "cpu" else torch.float16
    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    resume_settings = {
        "model": config,
        "dataset_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "prompt_config_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "device": device,
        "dtype": str(dtype),
    }
    if previous_metadata is not None:
        changed = [key for key, value in resume_settings.items() if previous_metadata[key] != value]
        if changed:
            print(f"기존 실행과 설정이 달라 이어서 실행하지 않았습니다: {', '.join(changed)}", flush=True)
            return
    if pending_total == 0:
        print("최종 추론 미완료: 0개", flush=True)
        return
    torch.manual_seed(SEED)
    print(f"그룹: {group_count}개 / 전체 추론: {total_trials}회 / 장치: {device}", flush=True)

    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )
    yes_id, = processor.tokenizer.encode("Yes", add_special_tokens=False)
    no_id, = processor.tokenizer.encode("No", add_special_tokens=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, dtype=dtype, attn_implementation="sdpa"
    ).to(device).eval()
    generation_config = GenerationConfig(
        do_sample=False,
        num_beams=1,
        max_new_tokens=MAX_NEW_TOKENS,
        repetition_penalty=1.0,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    metadata = {
        "model": config,
        "dataset": DATA_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "dataset_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "prompt_config": prompt_config,
        "prompt_config_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "evaluation_role": prompt_config["evaluation_role"],
        "group_count": group_count,
        "base_sample_count": len(samples),
        "planned_trials": total_trials,
        "trials_per_template": {key: len(value) for key, value in evaluations.items()},
        "input_format": {
            "order": "images_first_text_last",
            "image_numbering": False,
            "extra_text_separators": False,
            "ground_truth_in_model_input": False,
        },
        "device": device,
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "seed": SEED,
        "image_preprocessing": {
            "method": "aspect-ratio-preserving resize with white padding",
            "size": [IMAGE_SIZE, IMAGE_SIZE],
            "processor_do_resize": False,
        },
        "attn_implementation": "sdpa",
        "generation": generation_config.to_diff_dict(),
        "logit_position": "first_generated_token",
        "answer_token_ids": {"Yes": yes_id, "No": no_id},
    }
    if previous_metadata is not None:
        changed = [key for key, value in metadata.items() if previous_metadata.get(key) != value]
        if changed:
            print(f"기존 실행과 설정이 달라 이어서 실행하지 않았습니다: {', '.join(changed)}", flush=True)
            return
    else:
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    incomplete_total = 0
    for template_id, trials in evaluations.items():
        result_path = log_dir / f"baseline_e1_{template_id}.jsonl"
        failed_path = log_dir / f"baseline_e1_{template_id}_failed.jsonl"
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
                        "reason": "non_finite_scores",
                        "invalid_values": invalid_values,
                    }
                    with failed_path.open("a", encoding="utf-8") as failed:
                        failed.write(json.dumps(failure, ensure_ascii=False, allow_nan=False) + "\n")
                    print(f"[{template_id} {index}/{len(trials)}] {sample['sample_id']} 수치 오류로 건너뜀", flush=True)
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
                }
                output.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                output.flush()
                completed.add(sample["trial_id"])
                print(
                    f"[{template_id} {index}/{len(trials)}] {sample['sample_id']} "
                    f"예측={prediction['prediction']} 정답={sample['answer']}",
                    flush=True,
                )
        incomplete_total += sum(sample["trial_id"] not in completed for sample in trials)
    print(f"최종 추론 미완료: {incomplete_total}개", flush=True)


if __name__ == "__main__":
    main()
