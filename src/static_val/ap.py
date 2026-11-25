"""COCO mAP@[.50:.95] from cached true-positive tables under image weights.

An image weight of 0 removes the image, 1 keeps it, and k > 1 counts it k
times (bootstrap resampling). Tertile restriction is a 0/1 weight mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from static_val.constants import IOU_THRESHOLDS, NUM_CLASSES, RECALL_GRID
from static_val.matching import TP_COLUMNS


@dataclass(frozen=True)
class MethodTable:
    """Per-class detection arrays of one method, indexed against a fixed image order."""

    scores: tuple[np.ndarray, ...]
    image_idx: tuple[np.ndarray, ...]
    tps: tuple[np.ndarray, ...]
    gt_counts: np.ndarray  # (n_images, n_classes)

    @classmethod
    def from_frames(
        cls, detections: pd.DataFrame, gt_counts: np.ndarray, stems: list[str]
    ) -> MethodTable:
        stem_index = {s: i for i, s in enumerate(stems)}
        image_idx = detections["image_stem"].map(stem_index).to_numpy(dtype=np.int32)
        scores, idx, tps = [], [], []
        for c in range(NUM_CLASSES):
            mask = (detections["class_idx"] == c).to_numpy()
            scores.append(detections.loc[mask, "score"].to_numpy(dtype=np.float32))
            idx.append(image_idx[mask])
            tps.append(detections.loc[mask, TP_COLUMNS].to_numpy(dtype=bool))
        return cls(tuple(scores), tuple(idx), tuple(tps), gt_counts)


def _ap_per_iou(tp: np.ndarray, weight: np.ndarray, num_gt: int) -> np.ndarray:
    """101-point interpolated AP per IoU column for score-sorted detections."""
    w = weight.reshape(-1, 1)
    cum_tp = np.cumsum(tp.astype(np.float32) * w, axis=0)
    cum_fp = np.cumsum((~tp).astype(np.float32) * w, axis=0)
    recall = cum_tp / num_gt
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    precision = np.maximum.accumulate(precision[::-1], axis=0)[::-1]
    n_det = tp.shape[0]
    out = np.zeros(tp.shape[1], dtype=np.float64)
    for j in range(tp.shape[1]):
        idx = np.searchsorted(recall[:, j], RECALL_GRID, side="left")
        valid = idx < n_det
        interp = np.zeros_like(RECALL_GRID)
        interp[valid] = precision[idx[valid], j]
        out[j] = interp.mean()
    return out


def class_ap(table: MethodTable, weights: np.ndarray, c: int) -> np.ndarray | None:
    """AP per IoU threshold for class ``c``; None if the class has no ground truth."""
    num_gt = int(np.dot(weights, table.gt_counts[:, c]))
    if num_gt == 0:
        return None
    det_weight = weights[table.image_idx[c]]
    keep = det_weight > 0
    if not keep.any():
        return np.zeros(len(IOU_THRESHOLDS))
    scores = table.scores[c][keep]
    order = np.argsort(-scores, kind="stable")
    tp = table.tps[c][keep][order]
    weight = det_weight[keep][order].astype(np.float32)
    return _ap_per_iou(tp, weight, num_gt)


def mean_ap(table: MethodTable, weights: np.ndarray, classes: tuple[int, ...]) -> float:
    """Mean over (class, IoU) APs of the classes present in the weighted image set."""
    aps = [ap for c in classes if (ap := class_ap(table, weights, c)) is not None]
    return float(np.mean(np.concatenate(aps))) if aps else float("nan")
