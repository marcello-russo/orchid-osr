"""OSRDetector registry + 4 implementazioni OSR.

Mahalanobis, KNN, Energy, Ensemble.
Tutti i detector seguono: higher scores = more novel.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import rankdata
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class OSRDetector(ABC):
    """Base per detector OSR. Ogni istanza e' stateless prima di fit()."""

    name: ClassVar[str]
    requires_logits: ClassVar[bool] = False

    @abstractmethod
    def fit(self, embeddings: NDArray, labels: NDArray) -> None:
        """Fit su training embeddings + labels. Non vede test data."""

    @abstractmethod
    def score(self, embeddings: NDArray, logits: NDArray | None = None) -> NDArray[np.float32]:
        """Restituisce (N,) con higher = more novel."""


# --- Registry ---

_DETECTORS: dict[str, type[OSRDetector]] = {}


def register(cls):
    assert hasattr(cls, "name"), f"{cls.__name__} deve avere ClassVar 'name'"
    _DETECTORS[cls.name] = cls
    return cls


def list_detectors() -> list[str]:
    return list(_DETECTORS)


def get_detector(name: str) -> OSRDetector:
    if name not in _DETECTORS:
        raise ValueError(f"Unknown detector: {name}. Available: {list(_DETECTORS)}")
    return _DETECTORS[name]()


# --- Mahalanobis ---


@register
class MahalanobisDetector(OSRDetector):
    name = "Mahalanobis"

    def fit(self, embeddings, labels):
        unique = np.unique(labels)
        if len(unique) < 1:
            raise ValueError(f"{self.name}: training labels vuoti")
        if np.any(np.isnan(embeddings)):
            raise ValueError(f"{self.name}: embeddings contengono NaN")

        self.class_means_ = {}
        centered_list = []
        for c in unique:
            cls_emb = embeddings[labels == c]
            if len(cls_emb) < 2:
                logger.warning(
                    "%s: classe %s ha solo %d campioni — cov stima imprecisa",
                    self.name,
                    int(c),
                    len(cls_emb),
                )
            mean = cls_emb.mean(axis=0)
            self.class_means_[int(c)] = mean
            centered_list.append(cls_emb - mean)
        centered = np.vstack(centered_list)
        lw = LedoitWolf().fit(centered)
        self.pooled_precision_ = lw.precision_
        logger.debug(
            "%s: fit completato, %d classi, embedding dim %d",
            self.name,
            len(unique),
            embeddings.shape[1],
        )

    def score(self, embeddings, logits=None):
        scores = np.full(len(embeddings), np.inf)
        for c, mean in self.class_means_.items():
            diff = embeddings - mean
            dist = np.sum(diff @ self.pooled_precision_ * diff, axis=1)
            np.minimum(scores, dist, out=scores)
        return scores


# --- KNN ---


@register
class KNNDetector(OSRDetector):
    name = "KNN"

    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, embeddings, labels):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.train_emb_ = embeddings / (norms + 1e-12)
        k = min(self.k, len(self.train_emb_))
        self.nn_ = NearestNeighbors(n_neighbors=k).fit(self.train_emb_)

    def score(self, embeddings, logits=None):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb = embeddings / (norms + 1e-12)
        return self.nn_.kneighbors(emb)[0].mean(axis=1)


# --- Energy ---


@register
class EnergyDetector(OSRDetector):
    name = "Energy"
    requires_logits = True

    def __init__(self, temperature: float = 1.0):
        assert temperature >= 0.1, "Temperature < 0.1 causa overflow numerico"
        self.temperature = temperature

    def fit(self, embeddings, labels):
        pass

    def score(self, embeddings, logits=None):
        if logits is None:
            raise ValueError("EnergyDetector requires logits")
        if np.any(np.isnan(logits)):
            logger.warning("EnergyDetector: logits contengono NaN")
            return np.full(len(logits), float("nan"))
        return -self.temperature * logsumexp(logits / self.temperature, axis=1)


# --- Ensemble ---


@register
class EnsembleDetector(OSRDetector):
    name = "Ensemble"
    requires_logits = True

    def __init__(self):
        self.mahalanobis_ = MahalanobisDetector()
        self.knn_ = KNNDetector()
        self.energy_ = EnergyDetector()

    def fit(self, embeddings, labels):
        self.mahalanobis_.fit(embeddings, labels)
        self.knn_.fit(embeddings, labels)

    def score(self, embeddings, logits=None):
        def to_rank(s):
            return rankdata(s) / len(s)

        maha = to_rank(self.mahalanobis_.score(embeddings))
        knn = to_rank(self.knn_.score(embeddings))
        energy = to_rank(self.energy_.score(embeddings, logits=logits))
        return (maha + knn + energy) / 3.0
