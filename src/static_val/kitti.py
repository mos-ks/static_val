"""KITTI ground truth, random validation subsets and depth tertiles."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from static_val.constants import KITTI_CLASSES

_CLASS_INDEX = {name: i for i, name in enumerate(KITTI_CLASSES)}
DEPTH_TIERS = ("near", "medium", "far")


def image_stem(image_id: int) -> str:
    return f"{image_id:06d}"


def _label_rows(label_dir: Path, image_id: int) -> list[list[str]]:
    with open(label_dir / f"{image_stem(image_id)}.txt") as f:
        return [line.split() for line in f if line.strip()]


def load_ground_truth(label_dir: Path, image_ids: list[int]) -> dict[str, dict[int, np.ndarray]]:
    """Return image stem -> class index -> (n, 4) boxes as (x1, y1, x2, y2).

    Only the seven evaluated classes are kept (DontCare and Misc are dropped),
    matching the COCO ground truth the detector is scored against.
    """
    gt: dict[str, dict[int, np.ndarray]] = {}
    for image_id in image_ids:
        per_class: dict[int, list[tuple[float, float, float, float]]] = {}
        for row in _label_rows(label_dir, image_id):
            cls = _CLASS_INDEX.get(row[0])
            if cls is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in row[4:8])
            per_class.setdefault(cls, []).append((x1, y1, x2, y2))
        gt[image_stem(image_id)] = {
            c: np.asarray(boxes, dtype=np.float32) for c, boxes in per_class.items()
        }
    return gt


def random_subsets(ids: list[int], count: int, fraction: float, seed: int) -> list[list[int]]:
    """Draw ``count`` subsets of ``int(len(ids) * fraction)`` ids without replacement each."""
    rng = np.random.default_rng(seed)
    pool = np.asarray(ids)
    size = int(len(pool) * fraction)
    return [sorted(rng.choice(pool, size=size, replace=False).tolist()) for _ in range(count)]


def depth_signatures(label_dir: Path, depth_dir: Path, image_ids: list[int]) -> dict[int, float]:
    """Per-image mean of the per-object mean depth-map value inside each labelled box.

    Every labelled object except DontCare contributes. Pixels with value 0 are
    treated as invalid. Images without a valid object are left out.
    """
    signatures: dict[int, float] = {}
    for image_id in image_ids:
        depth = np.load(depth_dir / f"{image_stem(image_id)}_depth.npy")
        h, w = depth.shape
        object_means = []
        for row in _label_rows(label_dir, image_id):
            if row[0] == "DontCare":
                continue
            x1 = max(0, int(float(row[4])))
            y1 = max(0, int(float(row[5])))
            x2 = min(w, int(float(row[6])))
            y2 = min(h, int(float(row[7])))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = depth[y1:y2, x1:x2]
            valid = crop[crop > 0]
            if valid.size:
                object_means.append(float(valid.mean()))
        if object_means:
            signatures[image_id] = float(np.mean(object_means))
    return signatures


def depth_tertiles(signatures: dict[int, float], inverse_depth: bool) -> dict[str, list[int]]:
    """Split images into equal-count near/medium/far tertiles of their depth signature.

    By default a smaller value is closer, as in the DepthPrior depth maps.
    Set ``inverse_depth`` for maps that hold inverse depth (disparity), where a
    larger value is closer.
    """
    ordered = sorted(signatures, key=signatures.__getitem__)
    n = len(ordered)
    low, mid, high = ordered[: n // 3], ordered[n // 3 : 2 * n // 3], ordered[2 * n // 3 :]
    if inverse_depth:
        return {"near": high, "medium": mid, "far": low}
    return {"near": low, "medium": mid, "far": high}
