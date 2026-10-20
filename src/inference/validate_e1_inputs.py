# .venv/bin/python src/inference/validate_e1_inputs.py configs/model.json
# 모델 가중치와 추론 없이 E1 16개 샘플의 실제 입력 수·순서·토큰·잘림을 검증합니다.
import argparse
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, ImageOps
import torch
import transformers
from transformers import AutoProcessor

from run_e1 import (
    DATA_PATH, IMAGE_SIZE, MAX_NEW_TOKENS, PROJECT_ROOT, PROMPT_CONFIG_PATH,
    build_evaluations, prepare_inputs,
)

# 정상 추론 사례와 FP16 수치 오류 사례를 함께 검사합니다.
GROUP_IDS = ("cat_001", "bus_002")
OUTPUT_PATH = PROJECT_ROOT / "logs/e1_input_validation.json"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def read_results(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_bytes().splitlines(keepends=True)
            if line.endswith(b"\n") and line.strip()]


def reference_image(processor, path):
    # 다중 이미지 입력과 별개로 원본 파일 한 장만 전처리해 비교 기준을 만듭니다.
    with Image.open(PROJECT_ROOT / path) as source:
        image = ImageOps.pad(
            source.convert("RGB"), (IMAGE_SIZE, IMAGE_SIZE),
            method=Image.Resampling.LANCZOS, color="white",
        )
    return processor.image_processor(images=[image], do_resize=False, return_tensors="pt")


def check_sample(processor, sample, inputs, references, model_config):
    n = len(sample["images"])
    ids = inputs["input_ids"][0]
    grids = inputs["image_grid_thw"]
    pixels = inputs["pixel_values"]
    image_id = model_config["image_token_id"]
    merge_size = processor.image_processor.merge_size
    ref_grids = torch.cat([references[path]["image_grid_thw"] for path in sample["images"]])
    token_counts = (ref_grids.prod(-1) // merge_size**2).tolist()
    content = [{"type": "image"} for _ in range(n)]
    content.append({"type": "text", "text": sample["question"].strip()})
    text = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True,
    )
    # truncation=False인 독립 토큰화 결과와 전체 토큰열을 비교합니다.
    parts = text.split(processor.image_token)
    expanded = parts[0] + "".join(
        processor.image_token * count + part for count, part in zip(token_counts, parts[1:])
    )
    full_ids = processor.tokenizer(
        expanded, add_special_tokens=False, truncation=False, return_tensors="pt",
    )["input_ids"]
    starts = (ids == model_config["vision_start_token_id"]).nonzero().flatten().tolist()
    ends = (ids == model_config["vision_end_token_id"]).nonzero().flatten().tolist()
    blocks_ok = len(starts) == len(ends) == n
    if blocks_ok:
        blocks_ok = all(
            end - start - 1 == count and bool((ids[start + 1:end] == image_id).all())
            for start, end, count in zip(starts, ends, token_counts)
        )
    image_checks, hashes, offset = [], [], 0
    for path in sample["images"]:
        reference = references[path]["pixel_values"]
        block = pixels[offset:offset + len(reference)]
        image_checks.append(torch.equal(block, reference))
        hashes.append(digest(block.contiguous().numpy().tobytes()))
        offset += len(reference)
    checks = {
        "declared_image_count": sample["num_images"] == n,
        "image_files_exist": all((PROJECT_ROOT / path).is_file() for path in sample["images"]),
        "distinct_images": len(set(sample["images"])) == len(set(hashes)) == n,
        "grid_count_and_shape": torch.equal(grids, ref_grids),
        "patch_count": len(pixels) == offset,
        "image_content_and_order": all(image_checks),
        "image_blocks_and_delimiters": blocks_ok,
        "image_token_count": int((ids == image_id).sum()) == sum(token_counts),
        "no_truncation_or_extra_text": torch.equal(inputs["input_ids"], full_ids),
        "attention_mask": inputs["attention_mask"].shape == inputs["input_ids"].shape
                          and bool((inputs["attention_mask"] == 1).all()),
        "modality_token_types": torch.equal(inputs["mm_token_type_ids"], (inputs["input_ids"] == image_id).long()),
        "finite_pixels": bool(torch.isfinite(pixels).all()),
        "finite_pixels_after_fp16_cast": bool(torch.isfinite(pixels.half()).all()),
        "within_context_limit": len(ids) + MAX_NEW_TOKENS <= model_config["max_position_embeddings"],
        "question_N": f"each of the {n} images?" in sample["question"],
    }
    return {
        "sample_id": sample["sample_id"], "group_id": sample["group_id"],
        "num_images": n, "answer": sample["answer"], "images": sample["images"],
        "question": sample["question"], "rendered_chat_before_image_expansion": text,
        "input_token_count": len(ids), "image_token_count": int((ids == image_id).sum()),
        "image_grid_thw": grids.tolist(), "pixel_values_shape": list(pixels.shape),
        "image_token_counts": token_counts, "vision_start_positions": starts,
        "vision_end_positions": ends, "image_tensor_sha256": hashes,
        "single_image_tensor_matches": image_checks,
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser(description="E1 16개 샘플의 Processor 입력을 검증합니다.")
    parser.add_argument("config", type=Path, help="모델 설정 JSON 경로")
    args = parser.parse_args()
    config = json.loads(args.config.expanduser().read_text())
    model_path = (PROJECT_ROOT / config["save_dir"]).expanduser().resolve()
    model_config = json.loads((model_path / "config.json").read_text())
    data_bytes, prompt_bytes = DATA_PATH.read_bytes(), PROMPT_CONFIG_PATH.read_bytes()
    prompt_config = json.loads(prompt_bytes)
    trials = build_evaluations(
        json.loads(data_bytes), {**prompt_config, "enabled_templates": ["main"]},
    )["main"]
    samples = [sample for sample in trials if sample["group_id"] in GROUP_IDS]
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    references = {
        path: reference_image(processor, path)
        for path in dict.fromkeys(path for sample in samples for path in sample["images"])
    }
    saved = {}
    for filename in ("baseline_e1_main.jsonl", "baseline_e1_main_fp32_retry.jsonl"):
        for row in read_results(PROJECT_ROOT / "logs" / filename):
            if row["template_id"] == "main":
                saved.setdefault(row["sample_id"], row)
    records = []
    for sample in samples:
        inputs = prepare_inputs(processor, sample["images"], sample["question"])
        record = check_sample(processor, sample, inputs, references, model_config)
        previous = saved.get(sample["sample_id"])
        record["saved_result_available"] = previous is not None
        if previous is not None:
            record["checks"]["saved_input_matches"] = all(
                previous[key] == record[key]
                for key in ("images", "question", "num_images", "input_token_count", "image_token_count")
            )
        records.append(record)
        failed = [key for key, passed in record["checks"].items() if not passed]
        print(f"[{'FAIL' if failed else 'PASS'}] {sample['sample_id']} "
              f"입력 {record['input_token_count']} / 이미지 토큰 {record['image_token_count']}"
              + (f" / {', '.join(failed)}" if failed else ""), flush=True)

    by_pair = {(r["group_id"], r["num_images"], r["answer"]): r for r in records}
    pairs = []
    for group_id in GROUP_IDS:
        for n in (2, 4, 6, 8):
            a, b = by_pair.get((group_id, n, "Yes")), by_pair.get((group_id, n, "No"))
            changed = [i for i, (x, y) in enumerate(zip(a["image_tensor_sha256"], b["image_tensor_sha256"])) if x != y] if a and b else []
            pairs.append({"group_id": group_id, "num_images": n,
                          "changed_image_indices": changed, "passed": bool(a and b) and changed == [1]})
    checks = {
        "sample_coverage": len(samples) == 16 and len(by_pair) == 16 and all(p["passed"] for p in pairs),
        "all_sample_checks": all(all(r["checks"].values()) for r in records),
    }
    meta_path = PROJECT_ROOT / "logs/baseline_e1_meta.json"
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        checks["baseline_configuration_matches"] = all(
            metadata[key] == value for key, value in {
                "model": config, "dataset_sha256": digest(data_bytes),
                "prompt_config_sha256": digest(prompt_bytes),
                "torch_version": torch.__version__, "transformers_version": transformers.__version__,
                "image_preprocessing": {
                    "method": "aspect-ratio-preserving resize with white padding",
                    "size": [IMAGE_SIZE, IMAGE_SIZE], "processor_do_resize": False,
                },
            }.items()
        )
    report = {
        "passed": all(checks.values()), "checks": checks, "sample_count": len(records),
        "selection": {"groups": GROUP_IDS, "N": [2, 4, 6, 8], "answers": ["Yes", "No"]},
        "scope": "CPU Processor inputs only; no model weights, forward pass, or generation.",
        "model": config, "dataset_sha256": digest(data_bytes),
        "run_e1_sha256": digest((PROJECT_ROOT / "src/inference/run_e1.py").read_bytes()),
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "context_limit": model_config["max_position_embeddings"],
        "minimal_pairs": pairs, "samples": records,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"최종: {'PASS' if report['passed'] else 'FAIL'} / 보고서: {OUTPUT_PATH}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
