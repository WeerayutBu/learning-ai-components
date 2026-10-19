"""Regression checks for empty predictions and alternate dev files."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from infrastructure.evaluation import official_evaluate
from infrastructure.training import evaluate, report


class EvaluationTests(unittest.TestCase):
    def test_no_predictions(self):
        args = SimpleNamespace(data_dir="unused", dev_file="alternate.json")
        with patch("infrastructure.training.report", return_value=[]):
            self.assertEqual(evaluate(args, None, []),
                             (0.0, {"dev_F1": 0.0, "dev_F1_ign": 0.0}))
        self.assertEqual(official_evaluate([], "unused"), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(report(args, None, []), [])

    def test_custom_dev_file(self):
        prediction = {"title": "sample", "h_idx": 0, "t_idx": 1, "r": "P17"}
        document = {"title": "sample", "vertexSet": [[{"name": "Paris"}], [{"name": "France"}]],
                    "labels": [{"h": 0, "t": 1, "r": "P17", "evidence": [0]}]}
        with TemporaryDirectory() as directory:
            Path(directory, "alternate.json").write_text(json.dumps([document]))
            args = SimpleNamespace(data_dir=directory, dev_file="alternate.json")
            with patch("infrastructure.training.report", return_value=[prediction]), \
                    patch("infrastructure.evaluation.gen_train_facts", return_value=set()):
                score, metrics = evaluate(args, None, [])
            self.assertEqual(score, 1.0)
            self.assertEqual(metrics["dev_F1"], 100.0)
            self.assertAlmostEqual(metrics["dev_F1_ign"], 99.9995000025)
