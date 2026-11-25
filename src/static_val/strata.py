"""Image subsets evaluated by the detector, written as source-id lists.

Layout under ``<output_dir>/strata``::

    KITTI/subset_<i>.txt            random subsets of the KITTI validation set
    <target>/subset_<i>.txt         random subsets of each cross-domain target
    KITTI/depth_<near|medium|far>.txt
    KITTI/entropy_<low|mid|high>/<method>.txt   each method's own entropy tertiles

The full sets need no list. KITTI source ids are the original image indices.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from static_val.config import Config
from static_val.kitti import depth_signatures, depth_tertiles, random_subsets
from static_val.matching import read_predictions

UNC_TIERS = ("low", "mid", "high")


def entropy_tertiles(predictions: list[dict]) -> dict[str, list[int]]:
    """Tertiles of the per-image mean detection entropy of one method."""
    per_image: dict[int, list[float]] = defaultdict(list)
    for d in predictions:
        image_id = int(Path(d["image_name"]).stem)
        per_image[image_id].append(float(d["entropy"]))
    means = {k: float(np.mean(v)) for k, v in per_image.items()}
    ordered = sorted(means, key=means.__getitem__)
    n = len(ordered)
    return {
        "low": ordered[: n // 3],
        "mid": ordered[n // 3 : 2 * n // 3],
        "high": ordered[2 * n // 3 :],
    }


def _write(path: Path, ids: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(f"{i}\n" for i in ids)


def build_strata(cfg: Config, inverse_depth: bool = False) -> Path:
    root = cfg.output_dir / "strata"
    val_ids = cfg.kitti_val_ids()

    for i, ids in enumerate(
        random_subsets(val_ids, cfg.subset_count, cfg.subset_fraction, cfg.subset_seed)
    ):
        _write(root / "KITTI" / f"subset_{i}.txt", ids)
    for name, target in cfg.targets.items():
        subsets = random_subsets(
            target.source_ids(), cfg.subset_count, cfg.subset_fraction, cfg.subset_seed
        )
        for i, ids in enumerate(subsets):
            _write(root / name / f"subset_{i}.txt", ids)

    signatures = depth_signatures(cfg.kitti_label_dir, cfg.kitti_depth_dir, val_ids)
    tiers = depth_tertiles(signatures, inverse_depth=inverse_depth)
    for tier, ids in tiers.items():
        _write(root / "KITTI" / f"depth_{tier}.txt", ids)
    ordered = sorted(signatures.values())
    n = len(ordered)
    with open(root / "KITTI" / "depth_cuts.json", "w") as f:
        json.dump(
            {
                "inverse_depth": inverse_depth,
                "images": n,
                "cut_lower": ordered[n // 3],
                "cut_upper": ordered[2 * n // 3],
            },
            f,
            indent=2,
        )

    for method in cfg.methods:
        for tier, ids in entropy_tertiles(read_predictions(cfg.predictions(method))).items():
            _write(root / "KITTI" / f"entropy_{tier}" / f"{method}.txt", ids)
    return root
