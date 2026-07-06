import torch
from sklearn.metrics import accuracy_score, f1_score


class MulticlassAccuracy:
    def __init__(self, num_classes, average="macro"):
        self.num_classes = num_classes
        self.average = average
        self.reset()

    def to(self, device):
        return self

    def update(self, predictions, labels):
        self._preds.extend(predictions.cpu().tolist())
        self._labels.extend(labels.cpu().tolist())

    def compute(self):
        return torch.tensor(accuracy_score(self._labels, self._preds))

    def reset(self):
        self._preds = []
        self._labels = []


class MulticlassF1Score:
    def __init__(self, num_classes, average="macro"):
        self.num_classes = num_classes
        self.average = average
        self.reset()

    def to(self, device):
        return self

    def update(self, predictions, labels):
        self._preds.extend(predictions.cpu().tolist())
        self._labels.extend(labels.cpu().tolist())

    def compute(self):
        return torch.tensor(f1_score(self._labels, self._preds, average=self.average, zero_division=0))

    def reset(self):
        self._preds = []
        self._labels = []
