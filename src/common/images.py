# 이미지 로딩 공통 모듈. 사용: python src/visualize/plot_inference_cases.py

from pathlib import Path

from PIL import Image, ImageOps


def load_image(image_path: str | Path) -> Image.Image:
    """파일을 닫기 전에 EXIF 회전 정보를 반영한 RGB 이미지를 만듭니다."""
    with Image.open(Path(image_path).expanduser()) as image:
        return ImageOps.exif_transpose(image).convert("RGB")
