# .venv/bin/python src/visualization/plot_e1.py
# E1 main 결과에서 정확도, 오류율, 반례 민감도와 logit 분포를 6개 패널로 그립니다.
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INCLUDE_FP32 = True
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
SEED = 42
BOOTSTRAP_REPEATS = 2000
BLUE, RED, GREEN, PURPLE = "#2455d6", "#e63c38", "#237a3d", "#9146cc"


def read_jsonl(path):
    content = path.read_bytes()
    # 실행 중인 로그를 읽을 때 마지막 줄이 아직 기록 중이면 다음 실행에서 읽습니다.
    rows = [json.loads(line) for line in content.splitlines(keepends=True)
            if line.endswith(b"\n") and line.strip()]
    return rows, hashlib.sha256(content).hexdigest()


def load_results():
    metadata = json.loads((PROJECT_ROOT / "logs/baseline_e1_meta.json").read_text())
    data_bytes = (PROJECT_ROOT / metadata["dataset"]).read_bytes()
    if hashlib.sha256(data_bytes).hexdigest() != metadata["dataset_sha256"]:
        print("평가 데이터가 원본 추론 때와 달라 그래프를 생성하지 않았습니다.")
        return None
    expected = {sample["sample_id"]: sample for sample in json.loads(data_bytes)}
    paths = [(PROJECT_ROOT / "logs/baseline_e1_main.jsonl", metadata["dtype"])]
    if INCLUDE_FP32:
        retry_path = PROJECT_ROOT / "logs/baseline_e1_main_fp32_retry.jsonl"
        if retry_path.exists():
            retry_meta = json.loads((PROJECT_ROOT / "logs/baseline_e1_fp32_retry_meta.json").read_text())
            for key in ["model", "dataset_sha256", "prompt_config_sha256", "device", "generation"]:
                if retry_meta[key] != metadata[key]:
                    print(f"FP32 재추론의 {key} 설정이 달라 그래프를 생성하지 않았습니다.")
                    return None
            paths.append((retry_path, retry_meta["dtype"]))

    records, hashes = {}, {}
    for path, dtype in paths:
        rows, digest = read_jsonl(path)
        hashes[path.relative_to(PROJECT_ROOT).as_posix()] = digest
        for row in rows:
            sample_id = row["sample_id"]
            if row.get("template_id") != "main" or sample_id not in expected:
                continue
            sample = expected[sample_id]
            if any(row[key] != sample[key] for key in ["group_id", "class", "num_images", "images", "question", "answer"]):
                print(f"평가 데이터와 저장 결과가 일치하지 않습니다: {sample_id}")
                return None
            if not all(isinstance(row.get(key), (int, float)) and math.isfinite(row[key])
                       for key in ["yes_logit", "no_logit"]):
                continue
            # 기존 FP16 성공 결과를 우선하며, FP32는 누락된 샘플만 보완합니다.
            if sample_id not in records:
                records[sample_id] = {
                    **row, "dtype": dtype,
                    "margin": row["yes_logit"] - row["no_logit"],
                }
    return metadata, list(expected.values()), list(records.values()), hashes


def rate_summary(successes, total):
    if total == 0:
        return {"count": 0, "total": 0, "rate": None, "ci95": [None, None]}
    p = successes / total
    z = 1.959963984540054
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return {"count": successes, "total": total, "rate": p,
            "ci95": [max(0.0, center - radius), min(1.0, center + radius)]}


def summarize(expected, records):
    ns = sorted({sample["num_images"] for sample in expected})
    classes = list(dict.fromkeys(sample["class"] for sample in expected))
    group_classes = {sample["group_id"]: sample["class"] for sample in expected}
    by_key = {(row["group_id"], row["num_images"], row["answer"]): row for row in records}
    by_n = {}
    for n in ns:
        yes = [row for row in records if row["num_images"] == n and row["answer"] == "Yes"]
        no = [row for row in records if row["num_images"] == n and row["answer"] == "No"]
        by_n[str(n)] = {
            "yes_accuracy": rate_summary(sum(row["prediction"] == "Yes" for row in yes), len(yes)),
            "no_accuracy": rate_summary(sum(row["prediction"] == "No" for row in no), len(no)),
            "false_no_rate": rate_summary(sum(row["prediction"] == "No" for row in yes), len(yes)),
            "false_yes_rate": rate_summary(sum(row["prediction"] == "Yes" for row in no), len(no)),
            "invalid_predictions": sum(row["prediction"] not in ["Yes", "No"] for row in yes + no),
            "yes_margins": [row["margin"] for row in yes],
            "no_margins": [row["margin"] for row in no],
        }

    sensitivities = {}
    for group in group_classes:
        values = {}
        for n in ns:
            yes, no = by_key.get((group, n, "Yes")), by_key.get((group, n, "No"))
            if yes is not None and no is not None:
                values[str(n)] = yes["margin"] - no["margin"]
        sensitivities[group] = values
    complete_groups = [group for group, values in sensitivities.items() if len(values) == len(ns)]
    sensitivity_by_n = {}
    changes = []
    if complete_groups:
        matrix = np.array([[sensitivities[group][str(n)] for n in ns] for group in complete_groups])
        rng = np.random.default_rng(SEED)
        # 그룹 단위로 재표집하고 동일한 인덱스를 모든 N에 적용합니다.
        indices = rng.integers(0, len(complete_groups), size=(BOOTSTRAP_REPEATS, len(complete_groups)))
        means = matrix[indices].mean(axis=1)
        for index, n in enumerate(ns):
            sensitivity_by_n[str(n)] = {
                "mean": float(matrix[:, index].mean()),
                "ci95": np.quantile(means[:, index], [0.025, 0.975]).tolist() if len(complete_groups) > 1 else [None, None],
                "groups": len(complete_groups),
            }
        for index, group in enumerate(complete_groups):
            changes.append({"group_id": group, "class": group_classes[group],
                            "change": float(matrix[index, -1] - matrix[index, 0])})

    class_rates = []
    for category in classes:
        rows = [row for row in records if row["class"] == category and row["num_images"] == ns[-1] and row["answer"] == "No"]
        class_rates.append({"class": category, **rate_summary(sum(row["prediction"] == "Yes" for row in rows), len(rows))})
    class_rates.sort(key=lambda item: -(item["rate"] if item["rate"] is not None else -1))
    ids = {row["sample_id"] for row in records}
    return {
        "num_images": ns,
        "expected_trials": len(expected),
        "included_trials": len(records),
        "missing_sample_ids": [s["sample_id"] for s in expected if s["sample_id"] not in ids],
        "dtype_counts": {dtype: sum(row["dtype"] == dtype for row in records) for dtype in sorted({row["dtype"] for row in records})},
        "by_num_images": by_n,
        "sensitivity_by_group": sensitivities,
        "complete_groups": complete_groups,
        "expected_groups": len(group_classes),
        "sensitivity_by_num_images": sensitivity_by_n,
        "group_changes": changes,
        "mean_group_change": float(np.mean([item["change"] for item in changes])) if changes else None,
        "groups_with_decrease": sum(item["change"] < 0 for item in changes),
        "class_false_yes_at_max_n": class_rates,
    }


def plot_rate(ax, ns, items, color, marker, label=None):
    valid = [(n, item) for n, item in zip(ns, items) if item["rate"] is not None]
    if not valid:
        return
    x, rates = zip(*[(n, item["rate"]) for n, item in valid])
    ax.plot(x, rates, color=color, marker=marker, linewidth=1.7, markersize=5, label=label)
    ax.fill_between(x, [item["ci95"][0] for _, item in valid],
                    [item["ci95"][1] for _, item in valid], color=color, alpha=0.12)


def draw_figure(summary, metadata, stem):
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10,
                         "axes.titlesize": 11.5, "axes.titleweight": "bold",
                         "axes.labelsize": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    fig.subplots_adjust(left=0.065, right=0.98, top=0.87, bottom=0.19, hspace=0.52, wspace=0.30)
    ns = summary["num_images"]
    by_n = summary["by_num_images"]
    for ax in axes.flat:
        ax.set_axisbelow(True)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
        for spine in ax.spines.values():
            spine.set_color("#777777")
            spine.set_linewidth(0.7)
    for ax in axes[0]:
        ax.set_xticks(ns)
        ax.set_xlabel("Number of images (N)")
    for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
        ax.set_ylim(0, 1.04)
        ax.set_yticks(np.linspace(0, 1, 6))

    ax = axes[0, 0]
    ax.set_title("(a) Yes / No Accuracy")
    plot_rate(ax, ns, [by_n[str(n)]["yes_accuracy"] for n in ns], BLUE, "o", "Yes (all-positive)")
    plot_rate(ax, ns, [by_n[str(n)]["no_accuracy"] for n in ns], RED, "s", "No (single-negative)")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="best", fontsize=9)
    ax.set_xticklabels([f"{n}\n(n={by_n[str(n)]['yes_accuracy']['total']}/{by_n[str(n)]['no_accuracy']['total']})" for n in ns], fontsize=8.5)

    ax = axes[0, 1]
    ax.set_title("(b) False-Yes Rate")
    plot_rate(ax, ns, [by_n[str(n)]["false_yes_rate"] for n in ns], RED, "s")
    ax.set_ylabel("False-Yes rate")
    ax.set_xticklabels([f"{n}\n(n={by_n[str(n)]['false_yes_rate']['total']})" for n in ns], fontsize=8.5)

    ax = axes[0, 2]
    ax.set_title("(c) Counterexample Sensitivity")
    items = summary["sensitivity_by_num_images"]
    if items:
        ax.plot(ns, [items[str(n)]["mean"] for n in ns], color=GREEN, marker="o", linewidth=1.7, markersize=5)
        if len(summary["complete_groups"]) > 1:
            ax.fill_between(ns, [items[str(n)]["ci95"][0] for n in ns],
                            [items[str(n)]["ci95"][1] for n in ns], color=GREEN, alpha=0.13)
    else:
        ax.text(0.5, 0.5, "No complete groups", transform=ax.transAxes, ha="center")
    ax.axhline(0, color="#888888", linestyle=":", linewidth=0.9)
    ax.set_ylabel(r"$\Delta_N = s(A_N) - s(B_N)$")
    ax.text(0.96, 0.95, f"Matched groups: {len(summary['complete_groups'])}", transform=ax.transAxes, ha="right", va="top", fontsize=9)

    ax = axes[1, 0]
    ax.set_title(f"(d) Class-wise False-Yes Rate (N = {ns[-1]})")
    classes = summary["class_false_yes_at_max_n"]
    for index, item in enumerate(classes):
        if item["rate"] is None:
            ax.text(index, 0.025, "NA", ha="center", fontsize=8)
            continue
        rate, (lower, upper) = item["rate"], item["ci95"]
        ax.bar(index, rate, color="#8bbcf2", edgecolor=BLUE, linewidth=0.65, width=0.72)
        ax.errorbar(index, rate, yerr=[[max(0, rate-lower)], [max(0, upper-rate)]], color="#555555", capsize=2, linewidth=0.8)
    overall = by_n[str(ns[-1])]["false_yes_rate"]["rate"]
    if overall is not None:
        ax.axhline(overall, color=RED, linestyle="--", linewidth=1)
        ax.text(0.98, 0.96, f"Micro avg: {overall:.3f}", color=RED, ha="right", va="top", transform=ax.transAxes, fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"})
    ax.set_xticks(range(len(classes)), [f"{item['class']} ({item['total']})" for item in classes], rotation=65, ha="right", fontsize=8)
    ax.set_xlim(-0.7, len(classes)-0.3)
    ax.set_ylabel("False-Yes rate")
    ax.set_xlabel("Object class (available No samples)")

    ax = axes[1, 1]
    ax.set_title("(e) Yes / No Logit Margin Distribution")
    for offset, key, color in [(-0.27, "yes_margins", BLUE), (0.27, "no_margins", RED)]:
        for n in ns:
            values = by_n[str(n)][key]
            if values:
                ax.boxplot([values], positions=[n+offset], widths=0.42, patch_artist=True, manage_ticks=False,
                           boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.30},
                           medianprops={"color": color, "linewidth": 1.5},
                           whiskerprops={"color": "#777777"}, capprops={"color": "#777777"},
                           flierprops={"marker": ".", "markersize": 3, "markerfacecolor": color, "markeredgecolor": color, "alpha": 0.5})
    ax.axhline(0, color="#888888", linestyle=":", linewidth=0.9)
    ax.set_xticks(ns)
    ax.set_xlabel("Number of images (N)")
    ax.set_ylabel(r"$s(X) = \mathrm{logit}(Yes) - \mathrm{logit}(No)$")
    ax.legend(handles=[Patch(facecolor=BLUE, alpha=0.3, label="All-positive (Yes)"),
                       Patch(facecolor=RED, alpha=0.3, label="Single-negative (No)")], fontsize=8.5, loc="best")

    ax = axes[1, 2]
    ax.set_title("(f) Group-level Sensitivity Change")
    values = np.array([item["change"] for item in summary["group_changes"]])
    if len(values):
        if len(values) > 1 and np.ptp(values) > 0:
            violin = ax.violinplot([values], positions=[1], widths=0.70, showextrema=False)
            violin["bodies"][0].set_facecolor(PURPLE)
            violin["bodies"][0].set_edgecolor(PURPLE)
            violin["bodies"][0].set_alpha(0.17)
        rng = np.random.default_rng(SEED)
        ax.scatter(1+rng.uniform(-0.13, 0.13, len(values)), values, s=10, color=PURPLE, alpha=0.65, edgecolors="none")
        mean = summary["mean_group_change"]
        ax.axhline(mean, color=RED, linestyle="--", linewidth=1.1)
        ax.text(0.97, 0.05, f"Mean: {mean:+.3f}\nDecreased: {summary['groups_with_decrease']}/{len(values)}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=RED,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
    else:
        ax.text(0.5, 0.5, "No complete groups", transform=ax.transAxes, ha="center")
    ax.axhline(0, color="#777777", linestyle=":", linewidth=0.9)
    ax.set_xlim(0.5, 1.5)
    ax.set_xticks([1], [f"Matched groups (n = {len(values)})"])
    ax.set_ylabel(rf"$\Delta_{{{ns[-1]}}} - \Delta_{{{ns[0]}}}$")

    precision = "FP16 + FP32 recovery (mixed precision)" if INCLUDE_FP32 else metadata["dtype"].replace("torch.", "").upper()
    model_name = metadata["model"]["model_id"].split("/")[-1]
    fig.suptitle(f"{model_name} | E1 main | {precision}", y=0.97, fontsize=15, fontweight="bold")
    dtype_text = ", ".join(f"{key.replace('torch.', '')}: {value}" for key, value in summary["dtype_counts"].items())
    fig.text(0.5, 0.925, f"Included trials: {summary['included_trials']}/{summary['expected_trials']}  |  "
             f"Missing: {len(summary['missing_sample_ids'])}  |  {dtype_text}", ha="center", fontsize=10)
    fig.text(0.5, 0.035,
             "A: all-positive; B: one negative replacing image 2. Positive margin favors Yes.\n"
             "(a,b,d): 95% Wilson intervals; (c): pointwise 95% group-bootstrap intervals. "
             "(c,f): same groups complete at every N.\n"
             "Missing results are excluded; unparseable predictions count as incorrect accuracy, "
             "but not as explicit false-Yes/false-No. Pilot development data.", ha="center", va="bottom", fontsize=8.5, linespacing=1.6)
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    loaded = load_results()
    if loaded is None:
        return
    metadata, expected, records, hashes = loaded
    if not records:
        print("시각화할 유효한 결과가 없습니다.")
        return
    summary = summarize(expected, records)
    summary.update({
        "include_fp32": INCLUDE_FP32, "source_sha256": hashes,
        "dataset_sha256": metadata["dataset_sha256"], "template_id": "main",
        "seed": SEED, "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "definitions": {
            "margin": "logit(Yes) - logit(No)",
            "sensitivity": "margin(all-positive) - margin(single-negative)",
            "group_change": "sensitivity(max N) - sensitivity(min N); negative = decrease",
            "rate_ci": "95% Wilson", "sensitivity_ci": "pointwise 95% percentile bootstrap over matched groups",
            "sensitivity_cohort": "groups with both answers available at every N",
        },
    })
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = "e1_main_fp16_fp32" if INCLUDE_FP32 else "e1_main_fp16"
    (OUTPUT_DIR / f"{stem}_metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    draw_figure(summary, metadata, stem)
    print(f"분석 결과: {summary['included_trials']}/{summary['expected_trials']}개 / "
          f"공통 완성 그룹: {len(summary['complete_groups'])}/{summary['expected_groups']}개")
    print(f"그림: {OUTPUT_DIR / (stem + '.png')}")
    print(f"PDF: {OUTPUT_DIR / (stem + '.pdf')}")
    print(f"지표: {OUTPUT_DIR / (stem + '_metrics.json')}")


if __name__ == "__main__":
    main()
