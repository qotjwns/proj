# 공통 모델 모듈. 사용: python src/experiments/a/run.py --model-path data/models/Qwen/Qwen2.5-VL-3B-Instruct

from contextlib import ExitStack
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.common.images import load_image


def select_device(name: str = "auto") -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(name)


def load_model(model_path: Path, device: torch.device):
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, max_pixels=1280 * 28 * 28)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, dtype=dtype, attn_implementation="sdpa",
    ).to(device).eval()
    return model, processor


def prepare_inputs(processor, images: list, prompt: str):
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=images, padding=True, return_tensors="pt")


@torch.inference_mode()
def generate(
    model, processor, images: list[Path], prompt: str, max_new_tokens: int,
    *, logits: bool = True,
) -> dict:
    with ExitStack() as stack:
        loaded = [stack.enter_context(load_image(path)) for path in images]
        inputs = prepare_inputs(processor, loaded, prompt).to(model.device)
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            num_beams=1, num_return_sequences=1, temperature=1.0, repetition_penalty=1.0,
            return_dict_in_generate=True, output_logits=logits, output_scores=False,
        )
    answer_ids = generated.sequences[:, inputs["input_ids"].shape[1]:]
    answer = processor.batch_decode(
        answer_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    yes_logit = no_logit = None
    if logits:
        yes_id, = processor.tokenizer.encode("Yes", add_special_tokens=False)
        no_id, = processor.tokenizer.encode("No", add_special_tokens=False)
        yes_logit, no_logit = generated.logits[0][0, [yes_id, no_id]].float().cpu().tolist()
    return {"model_answer": answer, "yes_logit": yes_logit, "no_logit": no_logit}
