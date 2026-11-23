"""CLI phase dispatch: strategy pattern for experiment phases."""

from __future__ import annotations


def detect_phase(phases: list[str] | None, aggregate_only: bool) -> str | None:
    if aggregate_only:
        return "aggregate"
    if phases:
        if "embeddings" in phases:
            return "embeddings"
        if "ncd" in phases:
            return "ncd"
        if "embeddings_missing" in phases:
            return "embeddings_missing"
    return "train"
