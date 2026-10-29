"""기본 성능 지표와 혼동행렬을 PNG 한 장에 표시합니다."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_evaluation_plot(input_path: Path, metrics: dict, output_path: Path) -> None:
    """평가 코드에서 계산한 지표를 받아 모델별 요약 이미지를 저장합니다."""
    metric_rows = [
        [label, f"{metrics[key]:.2%}"]
        for key, label in (
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1", "F1"),
            ("yes_response_rate", "Yes response rate"),
        )
    ]
    confusion = metrics["confusion_matrix"]
    cell_names = (("TP", "FN"), ("FP", "TN"))
    counts = [[confusion[name] for name in row] for row in cell_names]
    max_count = max(max(row) for row in counts)

    fig, (metrics_axis, matrix_axis) = plt.subplots(1, 2, figsize=(12, 6))
    try:
        model_name = input_path.stem.removesuffix("_results")
        fig.suptitle(
            f"{model_name}\n"
            f"Basic evaluation | Samples: {metrics['total_samples']:,} | Positive: Yes",
            fontsize=15,
        )

        metrics_axis.set_axis_off()
        metrics_axis.set_title("Metrics", fontsize=13, pad=14)
        table = metrics_axis.table(
            cellText=metric_rows,
            colLabels=["Metric", "Value"],
            colWidths=[0.72, 0.28],
            cellLoc="left",
            colLoc="left",
            bbox=[0.02, 0.12, 0.96, 0.76],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        for (row, column), cell in table.get_celld().items():
            cell.set_edgecolor("#dbe3ed")
            if row == 0:
                cell.set_facecolor("#24486c")
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor("#f0f5fa" if row % 2 else "white")
                if column == 1:
                    cell.set_text_props(weight="bold")

        matrix_axis.imshow(
            counts, cmap="Blues", vmin=0, vmax=max(max_count, 1),
            interpolation="nearest",
        )
        matrix_axis.set_title("Confusion matrix (counts)", fontsize=13, pad=14)
        matrix_axis.set_xticks([0, 1], labels=["Yes", "No"])
        matrix_axis.set_yticks([0, 1], labels=["Yes", "No"])
        matrix_axis.set_xlabel("Predicted label")
        matrix_axis.set_ylabel("Ground truth")
        for row in range(2):
            for column in range(2):
                count = counts[row][column]
                matrix_axis.text(
                    column, row, f"{cell_names[row][column]}\n{count:,}",
                    ha="center", va="center", fontsize=18,
                    color="white" if count > max_count / 2 else "#152b40",
                )

        fig.text(
            0.5, 0.035,
            "Yes response rate = predicted Yes / all samples. "
            "Zero-denominator metrics are set to 0.",
            ha="center", fontsize=10, color="#52606d",
        )
        fig.tight_layout(rect=(0, 0.09, 1, 0.88), w_pad=3)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    finally:
        plt.close(fig)
