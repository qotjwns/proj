"""Pillow 이미지 로딩. 데이터 생성 코드는 이 모듈에 의존하지 않습니다."""

from pathlib import Path

from PIL import Image, ImageOps


def load_image(image_path: str | Path) -> Image.Image:
    """파일을 닫기 전에 EXIF 회전 정보를 반영한 RGB 이미지를 만듭니다."""
    with Image.open(Path(image_path).expanduser()) as image:
        return ImageOps.exif_transpose(image).convert("RGB")
