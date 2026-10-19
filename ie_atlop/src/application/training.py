"""Choose training or checkpoint evaluation without depending on ML libraries."""
from typing import Dict, Optional, Protocol


class TrainingBackend(Protocol):
    def prepare(self) -> None: ...
    def fit(self) -> None: ...
    def restore(self, checkpoint: str) -> None: ...
    def evaluate(self) -> Dict[str, float]: ...
    def export_predictions(self) -> None: ...


def run_training(backend: TrainingBackend, checkpoint: str = "") -> Optional[Dict[str, float]]:
    backend.prepare()
    if not checkpoint:
        backend.fit()
        return None
    backend.restore(checkpoint)
    metrics = backend.evaluate()
    backend.export_predictions()
    return metrics
