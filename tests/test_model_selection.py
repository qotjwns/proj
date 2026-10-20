import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/inference"))
import download_model
import model_config
import run_e1


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.models = model_config.load_model_configs()
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        registry = self.root / "models.json"
        registry.write_text(json.dumps(self.models), encoding="utf-8")
        self.enterContext(patch.object(model_config, "PROJECT_ROOT", self.root))
        self.enterContext(patch.object(model_config, "MODEL_CONFIG_PATH", registry))
        self.enterContext(patch.object(run_e1, "PROJECT_ROOT", self.root))
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))

    def test_selected_model_downloads_and_runs_with_separate_results(self):
        data_path = self.root / "eval.json"
        data_path.write_text(json.dumps([{
            "sample_id": "cat_001_yes", "group_id": "cat_001", "class": "cat",
            "images": ["cat.jpg"], "image_labels": [True], "answer": "Yes",
        }]), encoding="utf-8")
        prompt_path = self.root / "prompt.json"
        prompt_path.write_text(json.dumps({
            "evaluation_role": "test", "enabled_templates": ["main"],
            "templates": [{
                "template_id": "main", "task": "all_images", "invert_answer": False,
                "question": "Is there a {obj} in each of the {N} images?",
            }],
            "objects": {"cat": {"obj": "cat"}},
        }), encoding="utf-8")
        self.enterContext(patch.object(run_e1, "DATA_PATH", data_path))
        self.enterContext(patch.object(run_e1, "PROMPT_CONFIG_PATH", prompt_path))
        self.enterContext(patch.object(run_e1.torch.cuda, "is_available", return_value=False))
        self.enterContext(patch.object(run_e1.torch.backends.mps, "is_available", return_value=False))
        processor = MagicMock()
        processor.tokenizer.encode.side_effect = lambda answer, **kwargs: [1 if answer == "Yes" else 2]
        processor.tokenizer.pad_token_id = 0
        processor_loader = self.enterContext(patch.object(
            run_e1.AutoProcessor, "from_pretrained", return_value=processor,
        ))
        model = MagicMock()
        model.to.return_value = model
        model.eval.return_value = model
        model.generation_config.eos_token_id = 3
        model_loader = self.enterContext(patch.object(
            run_e1.AutoModelForImageTextToText, "from_pretrained", return_value=model,
        ))
        self.enterContext(patch.object(run_e1, "prepare_inputs"))
        predict = self.enterContext(patch.object(run_e1, "predict", return_value={
            "raw_output": "Yes", "prediction": "Yes", "yes_logit": 2.0,
            "no_logit": 1.0, "p_yes_given_yes_no": 0.73,
            "input_token_count": 10, "image_token_count": 8,
        }))
        download = self.enterContext(patch.object(download_model, "snapshot_download"))

        for name, config in self.models.items():
            with self.subTest(model=name), patch.object(sys, "argv", ["command", name]):
                path = self.root / config["save_dir"]
                download_model.main()
                download.assert_called_with(
                    repo_id=config["model_id"], repo_type="model",
                    revision=config["revision"], local_dir=str(path),
                )
                run_e1.main()
                processor_loader.assert_called_with(path, local_files_only=True)
                model_loader.assert_called_with(
                    path, local_files_only=True, dtype=run_e1.torch.float32,
                    attn_implementation="sdpa",
                )
                logs = self.root / "logs" / name
                metadata = json.loads((logs / "baseline_e1_meta.json").read_text())
                self.assertEqual(metadata["model"], config)
                result_path = logs / "baseline_e1_main.jsonl"
                saved = result_path.read_bytes()
                self.assertEqual(len(saved.splitlines()), 1)
                self.assertTrue(json.loads(saved)["correct"])

                # A repeat skips this model's completed trial; another model must run it.
                run_e1.main()
                self.assertEqual(result_path.read_bytes(), saved)

        self.assertEqual(predict.call_count, len(self.models))
        self.assertEqual(model_loader.call_count, len(self.models))
        self.assertFalse((self.root / "logs/baseline_e1_meta.json").exists())

    def test_legacy_results_are_only_used_by_the_matching_model(self):
        (self.root / "logs").mkdir()
        name, config = next(iter(self.models.items()))
        (self.root / "logs/baseline_e1_meta.json").write_text(json.dumps({"model": config}))
        self.assertEqual(model_config.get_log_dir(name, config), self.root / "logs")
        for other_name, other_config in self.models.items():
            if other_name != name:
                self.assertEqual(
                    model_config.get_log_dir(other_name, other_config),
                    self.root / "logs" / other_name,
                )
        new_logs = self.root / "logs" / name
        new_logs.mkdir()
        self.assertEqual(model_config.get_log_dir(name, config), new_logs)

    def test_unknown_model_is_rejected_before_downloading(self):
        with patch.object(sys, "argv", ["download_model.py", "unknown-model"]), \
                patch.object(download_model, "snapshot_download") as download, \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                download_model.main()
        self.assertEqual(error.exception.code, 2)
        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
