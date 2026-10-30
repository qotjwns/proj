# 지표 계산 모듈. 사용: python src/visualize/evaluate.py --experiment a


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    if total == 0:
        raise ValueError("평가할 결과가 없습니다.")
    tp = sum(row["gt"] == "Yes" and row["prediction"] == "Yes" for row in rows)
    tn = sum(row["gt"] == "No" and row["prediction"] == "No" for row in rows)
    fp = sum(row["gt"] == "No" and row["prediction"] == "Yes" for row in rows)
    fn = sum(row["gt"] == "Yes" and row["prediction"] == "No" for row in rows)
    invalid = sum(row["prediction"] is None for row in rows)
    positive = sum(row["gt"] == "Yes" for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / positive if positive else 0.0
    position_correct = [0, 0, 0]
    all_correct = 0
    has_image_predictions = rows[0]["experiment"] != "a"
    if has_image_predictions:
        for row in rows:
            if row["image_predictions"] is None:
                continue
            matches = [
                prediction == target
                for prediction, target in zip(row["image_predictions"], row["image_gt"])
            ]
            for index, correct in enumerate(matches):
                position_correct[index] += correct
            all_correct += all(matches)
    return {
        "cases": total, "accuracy": (tp + tn) / total,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "yes_response_rate": (tp + fp) / total,
        "true_yes": tp, "true_no": tn, "false_yes": fp, "false_no": fn,
        "invalid_answers": invalid, "valid_answer_rate": (total - invalid) / total,
        "image_accuracy": sum(position_correct) / (3 * total) if has_image_predictions else None,
        "image_accuracy_by_position": [count / total for count in position_correct] if has_image_predictions else None,
        "all_images_correct_rate": all_correct / total if has_image_predictions else None,
    }


def by_composition(rows: list[dict]) -> list[dict]:
    summary = []
    for positive_count, label in enumerate(("NNN", "PNN", "PPN", "PPP")):
        selected = [row for row in rows if row["image_gt"].count("Yes") == positive_count]
        errors = sum(row["prediction"] != row["gt"] for row in selected)
        summary.append({
            "composition": label, "cases": len(selected), "errors": errors,
            "error_rate": errors / len(selected) if selected else None,
            "false_yes": sum(row["gt"] == "No" and row["prediction"] == "Yes" for row in selected),
            "false_no": sum(row["gt"] == "Yes" and row["prediction"] == "No" for row in selected),
            "invalid_answers": sum(row["prediction"] is None for row in selected),
        })
    return summary
