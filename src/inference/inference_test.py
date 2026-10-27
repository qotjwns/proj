# python src/inference/inference_test.py
# python src/inference/inference_test.py --image /path/to/image1.jpg /path/to/image2.jpg
# python src/inference/inference_test.py --image /path/to/image1.jpg /path/to/image2.jpg --prompt "두 이미지의 차이를 설명해줘"

import argparse
from pathlib import Path

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.common.images import load_image
from src.common.paths import DEFAULT_MODEL_PATH, IMAGE_DIR


DEFAULT_IMAGE_PATHS = [
    IMAGE_DIR / "000000124798.jpg",
    IMAGE_DIR / "000000377393.jpg",
]
DEFAULT_PROMPT = "두 이미지의 차이를 설명해줘"


def select_device(device_name: str = "auto") -> torch.device:

    if device_name == "auto":
        if torch.cuda.is_available():
            device_name = "cuda"
        elif torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"

    return torch.device(device_name)


def load_model(model_path: str | Path, device: torch.device):
    model_path = Path(model_path).expanduser().resolve()
    if device.type == "cpu":
        dtype = torch.float32
    elif device.type == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        max_pixels=1280 * 28 * 28,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    model = model.to(device).eval()
    return model, processor


def prepare_inputs(processor, images: list[Image.Image], prompt: str):
    """이미지들을 입력 순서대로 한 메시지에 넣습니다."""
    messages = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in images]
            + [{"type": "text", "text": prompt}],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return processor(
        text=[text],
        images=images,
        padding=True,
        return_tensors="pt",
    )


@torch.inference_mode()
def infer_images(
    model,
    processor,
    images: list[Image.Image],
    prompt: str = DEFAULT_PROMPT,
    max_new_tokens: int = 256,
) -> str:
    """입력 이미지들의 정보를 함께 사용해 답변 하나를 생성합니다."""
    inputs = prepare_inputs(processor, images, prompt).to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    answer_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(
        answer_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen2.5-VL로 한 장 이상의 이미지에 대한 답변 생성")
    parser.add_argument(
        "--image",
        type=Path,
        nargs="+",
        default=DEFAULT_IMAGE_PATHS,
        metavar="IMAGE",
        help="입력 순서대로 이미지 경로 한 개 이상 (기본값: %(default)s)",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH,help="다운로드된 Qwen2.5-VL 모델 폴더")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="이미지에 대한 질문")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="최대 생성 토큰 수")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto", help="추론 장치 (auto: CUDA → MPS → CPU 순서로 선택)")
    args = parser.parse_args()

    images = [load_image(image_path) for image_path in args.image]
    device = select_device(args.device)
    for index, image_path in enumerate(args.image, start=1):
        print(f"입력 이미지 {index}: {image_path}", flush=True)
    print(f"프롬프트: {args.prompt}", flush=True)
    print(f"모델 경로: {args.model_path}", flush=True)
    print(f"추론 장치: {device}", flush=True)
    model, processor = load_model(args.model_path, device)
    answer = infer_images(
        model, processor, images, prompt=args.prompt, max_new_tokens=args.max_new_tokens
    )
    print(answer)


if __name__ == "__main__":
    main()
