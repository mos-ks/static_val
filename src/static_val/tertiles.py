"""Uncertainty tertiles of the in-domain validation set.

``global`` cuts give one LOW/MID/HIGH partition shared by every AL method:
the per-image score is the mean, over methods, of each method's per-image
mean uncertainty. A shared partition is required so that all methods are
ranked on the same images. ``per_method`` cuts use each method's own
uncertainty and are kept for comparison.

Tier codes: 0 LOW, 1 MID, 2 HIGH, -1 image without detections.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

SIGNAL_COLUMNS = {"entropy": "mean_entropy", "score": "mean_one_minus_score"}


def _index_cut(stems: list[str]) -> dict[str, int]:
    n = len(stems)
    tiers = np.empty(n, dtype=np.int8)
    tiers[: n // 3] = 0
    tiers[n // 3 : 2 * n // 3] = 1
    tiers[2 * n // 3 :] = 2
    return dict(zip(stems, tiers.tolist()))


def per_method_cut(features: pd.DataFrame, column: str) -> dict[str, int]:
    sub = features[features["num_dets"] > 0].sort_values(
        by=[column, "image_stem"], kind="mergesort"
    )
    return _index_cut(sub["image_stem"].tolist())


def global_cut(features_by_method: dict[str, pd.DataFrame], column: str) -> dict[str, int]:
    values: dict[str, list[float]] = defaultdict(list)
    for features in features_by_method.values():
        sub = features[features["num_dets"] > 0]
        for stem, val in zip(sub["image_stem"], sub[column]):
            values[stem].append(float(val))
    consensus = {s: float(np.mean(v)) for s, v in values.items()}
    return _index_cut(sorted(consensus, key=consensus.__getitem__))


def _as_array(tiers: dict[str, int], stems: list[str]) -> np.ndarray:
    return np.asarray([tiers.get(s, -1) for s in stems], dtype=np.int8)


def assign_tertiles(
    features_by_method: dict[str, pd.DataFrame], stems: list[str]
) -> dict[str, dict[str, np.ndarray]]:
    """Return mode -> signal -> (n_methods, n_images) tier codes, rows in method order."""
    methods = list(features_by_method)
    out: dict[str, dict[str, np.ndarray]] = {"global": {}, "per_method": {}}
    for signal, column in SIGNAL_COLUMNS.items():
        shared = _as_array(global_cut(features_by_method, column), stems)
        out["global"][signal] = np.tile(shared, (len(methods), 1))
        out["per_method"][signal] = np.stack(
            [_as_array(per_method_cut(features_by_method[m], column), stems) for m in methods]
        )
    return out
