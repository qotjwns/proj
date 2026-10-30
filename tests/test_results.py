# 실행: python -m unittest discover -s tests -p 'test_results.py'

from copy import deepcopy
from pathlib import Path
import unittest

from src.common.results import align_results, make_result, parse_labels
from src.visualize.metrics import summarize


def case(name="case_1", labels=("P", "P", "N")):
    return {
        "case_id": name, "group_id": "group_cat", "class": "cat",
        "image_dir": "data/coco/val2017", "image_order": ["1.jpg", "2.jpg", "3.jpg"],
        "image_labels": list(labels), "gt": "Yes" if all(label == "P" for label in labels) else "No",
    }


def result(experiment, answers, source=None):
    outputs = [{"model_answer": answer, "yes_logit": None, "no_logit": None} for answer in answers]
    return make_result(source or case(), experiment, Path("model"), "prompt", outputs, 32 if experiment == "c" else 8)


class ResultTests(unittest.TestCase):
    def test_schema_is_identical_across_experiments(self):
        rows = [result("a", ["No."]), result("b", ["Yes", "Yes", "No"]), result("c", ["Yes, Yes, No"])]
        self.assertEqual(set(rows[0]), set(rows[1]))
        self.assertEqual(set(rows[1]), set(rows[2]))
        for row in rows:
            self.assertEqual(row["prediction"], "No")
            self.assertTrue(row["correct"])
            self.assertTrue(row["parse_valid"])
            self.assertEqual(len(row["model_answers"]), len(row["yes_logits"]))
            self.assertEqual(len(row["yes_logits"]), len(row["no_logits"]))
        self.assertIsNone(rows[0]["image_predictions"])
        self.assertEqual(rows[1]["image_predictions"], rows[2]["image_predictions"])

    def test_and_is_different_from_all_images_correct(self):
        row = result("b", ["No", "Yes", "No"])
        metrics = summarize([row])
        self.assertEqual(row["prediction"], "No")
        self.assertEqual(metrics["accuracy"], 1)
        self.assertEqual(metrics["image_accuracy"], 2 / 3)
        self.assertEqual(metrics["all_images_correct_rate"], 0)

    def test_c_rejects_explanations_and_wrong_label_count(self):
        self.assertEqual(parse_labels(" Yes, No , Yes\n"), ["Yes", "No", "Yes"])
        for answer in ("", "Yes, No", "Yes, No, Yes, No", "Yes, No, Yes.", "yes, no, yes", "Yes, Yes, No\nFinal: No"):
            with self.subTest(answer=answer):
                self.assertIsNone(parse_labels(answer))

    def test_invalid_answers_are_not_silently_dropped(self):
        positive = case("positive", ("P", "P", "P"))
        rows = [result("c", ["Yes, Yes, Yes"], positive), result("c", ["explanation"], positive)]
        metrics = summarize(rows)
        for field in ("accuracy", "image_accuracy", "all_images_correct_rate", "recall"):
            self.assertEqual(metrics[field], 0.5)
        self.assertEqual(metrics["invalid_answers"], 1)
        self.assertEqual(metrics["false_no"], 0)

    def test_a_does_not_get_an_invented_image_accuracy(self):
        metrics = summarize([result("a", ["Yes"])])
        self.assertIsNone(metrics["image_accuracy"])
        self.assertIsNone(metrics["all_images_correct_rate"])
        self.assertEqual(metrics["false_yes"], 1)

    def test_b_counts_each_image_even_when_one_answer_is_invalid(self):
        row = result("b", ["Yes", "explanation", "No"])
        metrics = summarize([row])
        self.assertIsNone(row["prediction"])
        self.assertFalse(row["correct"])
        self.assertEqual(metrics["image_accuracy"], 2 / 3)
        self.assertEqual(metrics["all_images_correct_rate"], 0)
        self.assertEqual(metrics["invalid_answers"], 1)

    def test_comparison_checks_entire_set_and_image_order(self):
        cases = [case("one"), case("two")]
        rows = [result("a", ["No"], source) for source in cases]
        self.assertEqual(align_results(cases, rows[::-1], complete=True), rows)
        with self.assertRaises(ValueError):
            align_results(cases, rows[:1], complete=True)
        with self.assertRaises(ValueError):
            align_results(cases, [rows[0], rows[0]])
        changed = deepcopy(rows)
        changed[0]["image_order"].reverse()
        with self.assertRaises(ValueError):
            align_results(cases, changed)


if __name__ == "__main__":
    unittest.main()
