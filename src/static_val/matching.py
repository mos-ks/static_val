"""Per-detection true-positive tables for the in-domain validation set.

Each detection of a method is matched once against the ground truth at all
ten COCO IoU thresholds. Every later mAP (full set, uncertainty tertile,
bootstrap resample) then only re-aggregates these cached outcomes.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from static_val.constants import IOU_THRESHOLDS, NUM_CLASSES

TP_COLUMNS = [f"tp_iou{int(round(t * 100)):02d}" for t in IOU_THRESHOLDS]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of (N, 4) and (M, 4) boxes in (x1, y1, x2, y2)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a.T
    bx1, by1, bx2, by2 = b.T
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None]), 0, None)
    inter = iw * ih
    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = area_a[:, None] + area_b[None] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def match_detections(
    boxes: np.ndarray, scores: np.ndarray, gt_boxes: np.ndarray, thresholds=IOU_THRESHOLDS
) -> np.ndarray:
    """Greedy COCO matching for one (image, class), highest score first.

    Returns an (N, len(thresholds)) boolean TP array in the input order.
    """
    tp = np.zeros((boxes.shape[0], len(thresholds)), dtype=bool)
    if boxes.shape[0] == 0 or gt_boxes.shape[0] == 0:
        return tp
    ious = iou_matrix(boxes, gt_boxes)
    order = np.argsort(-scores, kind="stable")
    for ti, thr in enumerate(thresholds):
        matched = np.zeros(gt_boxes.shape[0], dtype=bool)
        for i in order:
            cand = ious[i].copy()
            cand[matched] = -1.0
            j = int(np.argmax(cand))
            if cand[j] >= thr:
                tp[i, ti] = True
                matched[j] = True
    return tp


def read_predictions(path: Path) -> list[dict]:
    """Parse a prediction file: one Python dict literal per detection.

    Used keys: ``image_name``, ``class`` (1-based), ``bbox`` as
    [y1, x1, y2, x2] in pixels, ``det_score`` and ``entropy``.
    """
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(ast.literal_eval(line))
    return rows


def build_tables(
    predictions: list[dict], gt: dict[str, dict[int, np.ndarray]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (detections, image_features) for one method.

    detections: one row per detection with its image, class, score,
    uncertainty signals and the TP outcome at each IoU threshold.
    image_features: per image, the mean entropy and mean (1 - score) of its
    detections and the detection count.
    """
    by_image_class: dict[str, dict[int, list[dict]]] = {}
    n_rows = 0
    for d in predictions:
        cls = int(d["class"]) - 1
        if not 0 <= cls < NUM_CLASSES:
            continue
        stem = Path(d["image_name"]).stem
        y1, x1, y2, x2 = (float(v) for v in d["bbox"])
        by_image_class.setdefault(stem, {}).setdefault(cls, []).append(
            {
                "row": n_rows,
                "score": float(d["det_score"]),
                "entropy": float(d["entropy"]),
                "bbox": (x1, y1, x2, y2),
            }
        )
        n_rows += 1

    # Rows keep the prediction-file order, which fixes how equal scores are ranked.
    records: list[dict] = [{} for _ in range(n_rows)]
    tp_array = np.zeros((n_rows, len(TP_COLUMNS)), dtype=bool)
    for stem, by_cls in by_image_class.items():
        gt_cls = gt.get(stem, {})
        for cls, dets in by_cls.items():
            boxes = np.asarray([d["bbox"] for d in dets], dtype=np.float32)
            scores = np.asarray([d["score"] for d in dets], dtype=np.float32)
            gt_boxes = gt_cls.get(cls, np.zeros((0, 4), dtype=np.float32))
            tp = match_detections(boxes, scores, gt_boxes)
            for d, tp_row in zip(dets, tp):
                records[d["row"]] = {
                    "image_stem": stem,
                    "class_idx": cls,
                    "score": d["score"],
                    "entropy": d["entropy"],
                    "one_minus_score": 1.0 - d["score"],
                }
                tp_array[d["row"]] = tp_row

    detections = pd.DataFrame(records)
    for i, col in enumerate(TP_COLUMNS):
        detections[col] = tp_array[:, i]

    feature_rows = []
    for stem in sorted(set(gt) | set(by_image_class)):
        dets = [d for ds in by_image_class.get(stem, {}).values() for d in ds]
        feature_rows.append(
            {
                "image_stem": stem,
                "mean_entropy": float(np.mean([d["entropy"] for d in dets])) if dets else np.nan,
                "mean_one_minus_score": (
                    float(np.mean([1.0 - d["score"] for d in dets])) if dets else np.nan
                ),
                "num_dets": len(dets),
            }
        )
    return detections, pd.DataFrame(feature_rows)


def gt_count_matrix(gt: dict[str, dict[int, np.ndarray]], stems: list[str]) -> np.ndarray:
    """(n_images, n_classes) ground-truth box counts in ``stems`` order."""
    counts = np.zeros((len(stems), NUM_CLASSES), dtype=np.int32)
    for i, stem in enumerate(stems):
        for cls, boxes in gt.get(stem, {}).items():
            counts[i, cls] = boxes.shape[0]
    return counts
