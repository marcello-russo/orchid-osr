import torch
from sklearn.metrics import accuracy_score, f1_score


class MulticlassAccuracy:
    def __init__(self, num_classes: int, average: str = "macro") -> None:
        self.num_classes: int = num_classes
        self.average: str = average
        self._preds: list[int] = []
        self._labels: list[int] = []
        self.reset()

    def to(self, device: torch.device) -> "MulticlassAccuracy":  # type: ignore[no-untyped-def]
        return self

    def update(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        self._preds.extend(predictions.cpu().tolist())  # type: ignore[no-untyped-call]
        self._labels.extend(labels.cpu().tolist())  # type: ignore[no-untyped-call]

    def compute(self) -> torch.Tensor:
        return torch.tensor(accuracy_score(self._labels, self._preds))

    def reset(self) -> None:
        self._preds = []
        self._labels = []


class MulticlassF1Score:
    def __init__(self, num_classes: int, average: str = "macro") -> None:
        self.num_classes: int = num_classes
        self.average: str = average
        self._preds: list[int] = []
        self._labels: list[int] = []
        self.reset()

    def to(self, device: torch.device) -> "MulticlassF1Score":  # type: ignore[no-untyped-def]
        return self

    def update(self, predictions: torch.Tensor, labels: torch.Tensor) -> None:
        self._preds.extend(predictions.cpu().tolist())  # type: ignore[no-untyped-call]
        self._labels.extend(labels.cpu().tolist())  # type: ignore[no-untyped-call]

    def compute(self) -> torch.Tensor:
        return torch.tensor(f1_score(self._labels, self._preds, average=self.average, zero_division=0))

    def reset(self) -> None:
        self._preds = []
        self._labels = []
