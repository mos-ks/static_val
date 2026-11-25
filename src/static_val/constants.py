"""Class taxonomy and evaluation constants shared by all stages."""

import numpy as np

KITTI_CLASSES = ("Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram")
NUM_CLASSES = len(KITTI_CLASSES)

# Class indices are 0-based positions in KITTI_CLASSES (COCO category_id - 1).
CLASS_GROUPS = {
    "all": tuple(range(NUM_CLASSES)),
    "head": (0, 1, 2, 3, 5),
    "tail": (4, 6),
    "shared": (0, 2, 3, 5),
}

IOU_THRESHOLDS = np.round(np.linspace(0.50, 0.95, 10), 2)
RECALL_GRID = np.linspace(0.0, 1.0, 101)

SIGNALS = ("entropy", "score")
CUT_MODES = ("global", "per_method")
TERTILES = ("LOW", "MID", "HIGH")
SLICES = ("FULL",) + TERTILES
