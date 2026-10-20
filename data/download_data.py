from pathlib import Path
import shutil
import sys
import time
from urllib.request import Request, urlopen
import zipfile
import zlib


CHUNK_SIZE = 1024 * 1024
DATA_DIR = Path(__file__).resolve().parent
DATASETS = {
    "images": {
        "url": "http://images.cocodataset.org/zips/val2017.zip",
        "archive": "val2017.zip",
        "destination": "coco",
        "expected": "val2017/",
    },
    "annotations": {
        "url": (
            "https://huggingface.co/datasets/kdexd/coco-rem/"
            "resolve/main/instances_valrem.json.zip"
        ),
        "archive": "instances_valrem.json.zip",
        "destination": "coco_rem",
        "expected": "instances_valrem.json",
    },
}


def download(url: str, archive_path: Path) -> None:
    if archive_path.exists():
        if not zipfile.is_zipfile(archive_path):
            raise ValueError(f"Invalid existing ZIP; move it aside and rerun: {archive_path}")
        print(f"Reusing {archive_path}")
        return

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = archive_path.with_suffix(archive_path.suffix + ".part")
    request = Request(url, headers={"User-Agent": "COCO-ReM-data-downloader/1.0"})
    print(f"Downloading {url}", flush=True)
    with urlopen(request, timeout=60) as response, partial_path.open("wb") as output:
        total = int(response.headers.get("Content-Length", "0"))
        received = 0
        last_update = 0.0
        while True:
            chunk = response.read(CHUNK_SIZE)
            if not chunk:
                break
            output.write(chunk)
            received += len(chunk)
            now = time.monotonic()
            if now - last_update >= 1:
                progress = f"{received / CHUNK_SIZE:.1f} MiB"
                if total:
                    progress += f" / {total / CHUNK_SIZE:.1f} MiB ({received / total:.1%})"
                print(f"\r  {progress}", end="", flush=True)
                last_update = now
    print(f"\r  Downloaded {received / CHUNK_SIZE:.1f} MiB.                 ")
    if total and received != total:
        raise OSError(f"Incomplete download: expected {total} bytes, got {received}")
    if not zipfile.is_zipfile(partial_path):
        raise ValueError(f"Downloaded response is not a ZIP: {partial_path}")
    partial_path.replace(archive_path)


def matches_member(path: Path, member: zipfile.ZipInfo) -> bool:
    if not path.is_file() or path.stat().st_size != member.file_size:
        return False
    checksum = 0
    with path.open("rb") as existing:
        while True:
            chunk = existing.read(CHUNK_SIZE)
            if not chunk:
                break
            checksum = zlib.crc32(chunk, checksum)
    return checksum == member.CRC


def extract(archive_path: Path, destination: Path, expected: str) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        names = {member.filename for member in members if not member.is_dir()}
        if expected.endswith("/"):
            present = any(name.startswith(expected) and name.endswith(".jpg") for name in names)
        else:
            present = expected in names
        if not present:
            raise ValueError(f"Expected {expected} inside {archive_path}")

        # Validate every output path before writing any archive content.
        for member in members:
            target = (destination / member.filename).resolve()
            target.relative_to(destination)

        written = 0
        reused = 0
        print(f"Extracting and checking {archive_path.name} ...", flush=True)
        for member in members:
            target = destination / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if matches_member(target, member):
                reused += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # Write atomically so interrupted extraction can be safely repeated.
            temporary = target.with_name(target.name + ".extracting")
            with archive.open(member) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=CHUNK_SIZE)
            temporary.replace(target)
            written += 1
    print(f"  {destination}: {written} extracted, {reused} already verified")


def main() -> int:
    for name, dataset in DATASETS.items():
        archive_path = DATA_DIR / "archives" / dataset["archive"]
        destination = DATA_DIR / dataset["destination"]
        print(f"\n[{name}]")
        download(dataset["url"], archive_path)
        extract(archive_path, destination, dataset["expected"])

    print(f"\nDone. Data directory: {DATA_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
