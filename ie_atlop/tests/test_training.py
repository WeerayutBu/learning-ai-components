"""Run with PYTHONPATH=ie_atlop/src python -m unittest discover -s ie_atlop/tests."""
import ast
from pathlib import Path
import unittest
from unittest.mock import Mock, call

from application.training import TrainingBackend, run_training


class TrainingTests(unittest.TestCase):
    def test_training_workflow(self):
        backend = Mock(spec=TrainingBackend)
        self.assertIsNone(run_training(backend))
        self.assertEqual(backend.mock_calls, [call.prepare(), call.fit()])

    def test_checkpoint_workflow(self):
        backend = Mock(spec=TrainingBackend)
        backend.evaluate.return_value = {"dev_F1": 60.0}
        self.assertEqual(run_training(backend, "saved.pt"), {"dev_F1": 60.0})
        self.assertEqual(backend.mock_calls, [call.prepare(), call.restore("saved.pt"),
                         call.evaluate(), call.export_predictions()])

    def test_backend_failure_propagates(self):
        for operation in ("prepare", "fit", "restore", "evaluate", "export_predictions"):
            with self.subTest(operation=operation):
                backend = Mock(spec=TrainingBackend)
                getattr(backend, operation).side_effect = RuntimeError("failed")
                checkpoint = "" if operation == "fit" else "saved.pt"
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    run_training(backend, checkpoint)
                self.assertEqual(backend.mock_calls[-1][0], operation)

    def test_dependency_boundaries(self):
        source = Path(__file__).resolve().parents[1] / "src"
        forbidden = {
            "application": {"infrastructure", "interfaces", "bootstrap", "torch", "transformers", "apex", "wandb"},
            "infrastructure": {"interfaces", "bootstrap"},
            "interfaces": {"infrastructure", "bootstrap", "torch", "transformers", "apex", "wandb"},
        }
        for layer, names in forbidden.items():
            for path in (source / layer).glob("*.py"):
                for node in ast.walk(ast.parse(path.read_text())):
                    imports = []
                    if isinstance(node, ast.Import):
                        imports = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.level == 0:
                        imports = [node.module or ""]
                    for name in imports:
                        self.assertNotIn(name.split(".")[0], names, str(path))
