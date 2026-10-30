# 표·CSV 저장 모듈. 사용: python src/visualize/compare.py

import csv
from pathlib import Path


def percentage(value) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_table(headers: list[str], rows: list[list], title: str, path: Path, note: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(max(8, len(headers) * 1.7), max(3.5, len(rows) * 0.45 + 2)))
    try:
        axis.set_axis_off()
        axis.set_title(title, fontsize=14, pad=18)
        table = axis.table(cellText=rows, colLabels=headers, cellLoc="center", bbox=[0, 0.1, 1, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        for (row, column), cell in table.get_celld().items():
            cell.set_edgecolor("#dbe3ed")
            if row == 0:
                cell.set_facecolor("#24486c")
                cell.set_text_props(color="white", weight="bold")
            if column == 0:
                cell.set_text_props(ha="left")
        fig.text(0.5, 0.02, note, ha="center", fontsize=9)
        fig.tight_layout(rect=(0.02, 0.1, 0.98, 0.95))
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)
