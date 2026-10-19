# 카테고리별 양성·음성 후보를 각각 최대 5장씩 골라 이미지 ID와 양성의 대상 객체
# bbox를 표시한 example.png를 각 카테고리 폴더에 저장합니다.

from collections import defaultdict
import json
from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_DIR = Path(__file__).resolve().parents[2]
POOL_DIR = PROJECT_DIR / "data" / "pools"
ANNOTATION_PATH = PROJECT_DIR / "data" / "coco_rem" / "instances_valrem.json"
EXAMPLES_PER_POOL = 5
SEED = 42

MARGIN = 28
GAP = 16
CARD_WIDTH = 328
CARD_HEIGHT = 274
IMAGE_HEIGHT = 222
HEADER_HEIGHT = 120
ROW_LABEL_HEIGHT = 38
ROW_GAP = 30
CANVAS_WIDTH = MARGIN * 2 + CARD_WIDTH * EXAMPLES_PER_POOL + GAP * (EXAMPLES_PER_POOL - 1)
CANVAS_HEIGHT = HEADER_HEIGHT + (ROW_LABEL_HEIGHT + CARD_HEIGHT) * 2 + ROW_GAP + 54
GREEN = "#16845b"
AMBER = "#b7791f"
INK = "#182536"
MUTED = "#64748b"

FONT_PATH = next(
    (path for path in (
        Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("C:/Windows/Fonts/malgun.ttf"),
    ) if path.is_file()),
    None,
)
KOREAN = FONT_PATH is not None and FONT_PATH.name != "DejaVuSans.ttf"


def font(size):
    if FONT_PATH is not None:
        return ImageFont.truetype(str(FONT_PATH), size)
    return ImageFont.load_default(size=size)


def sample_images(manifest, pool_index):
    records = sorted(manifest["images"], key=lambda record: record["image_id"])
    rng = random.Random(SEED + manifest["category_id"] * 2 + pool_index)
    return rng.sample(records, min(EXAMPLES_PER_POOL, len(records)))


def draw_card(canvas, draw, x, y, record, category_id, positive, boxes, image_metadata):
    accent = GREEN if positive else AMBER
    draw.rounded_rectangle(
        (x, y, x + CARD_WIDTH, y + CARD_HEIGHT), radius=12,
        fill="white", outline="#dce3eb", width=1,
    )
    if record is None:
        draw.text(
            (x + CARD_WIDTH / 2, y + CARD_HEIGHT / 2),
            "추가 이미지 없음" if KOREAN else "No more images",
            font=font(21), fill=MUTED, anchor="mm",
        )
        return

    with Image.open(PROJECT_DIR / record["image_path"]) as source:
        thumbnail = ImageOps.contain(
            source.convert("RGB"), (CARD_WIDTH - 20, IMAGE_HEIGHT),
            method=Image.Resampling.LANCZOS,
        )
    image_x = x + (CARD_WIDTH - thumbnail.width) // 2
    image_y = y + 10 + (IMAGE_HEIGHT - thumbnail.height) // 2
    draw.rectangle(
        (x + 10, y + 10, x + CARD_WIDTH - 10, y + 10 + IMAGE_HEIGHT),
        fill="#edf1f5",
    )
    canvas.paste(thumbnail, (image_x, image_y))

    ratio_label = "검수 전" if KOREAN else "Unreviewed"
    if positive:
        bbox = max(boxes[(category_id, record["image_id"])], key=lambda box: box[2] * box[3])
        metadata = image_metadata[record["image_id"]]
        scale_x = thumbnail.width / metadata["width"]
        scale_y = thumbnail.height / metadata["height"]
        bx, by, bw, bh = bbox
        draw.rectangle(
            (
                image_x + max(0, bx) * scale_x,
                image_y + max(0, by) * scale_y,
                image_x + min(metadata["width"], bx + bw) * scale_x,
                image_y + min(metadata["height"], by + bh) * scale_y,
            ),
            outline=GREEN, width=3,
        )
        ratio_label = f"bbox {bw * bh / (metadata['width'] * metadata['height']):.1%}"

    draw.text((x + 12, y + CARD_HEIGHT - 29), f"ID {record['image_id']:012d}", font=font(16), fill=INK)
    draw.text(
        (x + CARD_WIDTH - 12, y + CARD_HEIGHT - 29), ratio_label,
        font=font(16), fill=accent, anchor="ra",
    )


def create_sheet(folder, positive, negative, boxes, image_metadata):
    canvas = Image.new("RGB", (CANVAS_WIDTH, CANVAS_HEIGHT), "#f3f6fa")
    draw = ImageDraw.Draw(canvas)
    category_id = positive["category_id"]
    draw.text((MARGIN, 22), positive["category_name"], font=font(40), fill=INK)
    draw.text(
        (CANVAS_WIDTH - MARGIN, 36), f"CLASS {category_id:02d}",
        font=font(20), fill=MUTED, anchor="ra",
    )
    subtitle = (
        "위: 양성  |  아래: 음성 후보  |  초록 박스: 대상 객체의 가장 큰 bbox"
        if KOREAN else "Top: positive | Bottom: negative candidates | Green box: largest target bbox"
    )
    draw.text((MARGIN, 78), subtitle, font=font(20), fill=MUTED)

    for row_index, manifest in enumerate((positive, negative)):
        is_positive = row_index == 0
        selected = sample_images(manifest, row_index)
        row_y = HEADER_HEIGHT + row_index * (ROW_LABEL_HEIGHT + CARD_HEIGHT + ROW_GAP)
        if KOREAN:
            label = "양성 POS · bbox 면적 5% 이상" if is_positive else "음성 후보 NEG · 대상 객체 주석 없음 / 사진 검수 전"
            count = f"전체 {manifest['image_count']:,}장 중 {len(selected)}장"
        else:
            label = "POS · bbox area >= 5%" if is_positive else "NEG candidates · No target annotation / Unreviewed"
            count = f"{len(selected)} of {manifest['image_count']:,} images"
        draw.text((MARGIN, row_y), label, font=font(23), fill=GREEN if is_positive else AMBER)
        draw.text((CANVAS_WIDTH - MARGIN, row_y + 2), count, font=font(19), fill=MUTED, anchor="ra")
        for column in range(EXAMPLES_PER_POOL):
            draw_card(
                canvas, draw, MARGIN + column * (CARD_WIDTH + GAP), row_y + ROW_LABEL_HEIGHT,
                selected[column] if column < len(selected) else None,
                category_id, is_positive, boxes, image_metadata,
            )

    footer = (
        f"고정 시드 {SEED} · 원본 전체 영역을 축소해 표시 · 음성 후보는 실제 객체 부재 확인이 필요합니다."
        if KOREAN else f"Seed {SEED} · Full images shown without cropping · Negative candidates need visual review."
    )
    draw.text((MARGIN, CANVAS_HEIGHT - 33), footer, font=font(17), fill=MUTED)
    output_path = folder / "example.png"
    temporary_path = folder / "example.png.tmp"
    canvas.save(temporary_path, format="PNG")
    temporary_path.replace(output_path)
    print(f"{folder.name}: POS {min(5, positive['image_count'])} / NEG candidates {min(5, negative['image_count'])}")


def main():
    with ANNOTATION_PATH.open(encoding="utf-8") as source:
        data = json.load(source)
    image_metadata = {image["id"]: image for image in data["images"]}
    boxes = defaultdict(list)
    for annotation in data["annotations"]:
        boxes[(annotation["category_id"], annotation["image_id"])].append(annotation["bbox"])

    manifests = sorted(POOL_DIR.glob("*/positive.json"))
    for positive_path in manifests:
        with positive_path.open(encoding="utf-8") as source:
            positive = json.load(source)
        with (positive_path.parent / "negative_candidates.json").open(encoding="utf-8") as source:
            negative = json.load(source)
        create_sheet(positive_path.parent, positive, negative, boxes, image_metadata)
    print(f"\n완료: {len(manifests)}개 클래스의 example.png 저장")


if __name__ == "__main__":
    main()
